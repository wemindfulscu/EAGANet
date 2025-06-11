import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelSSM(nn.Module):
    def __init__(self, d_inner, d_state=16, A_matrix=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.A = nn.Parameter(nn.init.xavier_normal_(torch.randn(d_state, d_state)))  # 改用Xavier初始化控制参数范围
        self.delta = nn.Parameter(torch.randn(self.d_inner))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_state, self.d_inner))

    def forward(self, x):
        B, L, D = x.shape
        assert D == self.d_inner

        delta = F.softplus(self.delta).clamp(max=10.0) 
        A = self.A  # (N, N)
        B = self.B  # (D, N)
        C = self.C  # (N, D)

        A_bar = torch.exp(delta[:, None, None] * A)  # (D, N, N)
        B_bar = delta[:, None] * B  # (D, N)

        if A.shape[0] == A.diag().shape[0]: 
            A_diag = A.diag()  # (N,)
            A_powers = A_diag ** torch.arange(L, device=x.device)[:, None]  # (L, N)
            A_powers = A_powers[:, None, :]  # (L, 1, N)
            B_bar_expanded = B_bar[None, :, :]  # (1, D, N)
            A_powers_B_parts = A_powers * B_bar_expanded  # (L, D, N)
        else:
            A_powers_B_parts = torch.zeros(L, D, self.d_state, device=x.device, dtype=x.dtype)
            current = B_bar  # (D, N)
            A_powers_B_parts[0] = current
            for i in range(1, L):
                current = torch.einsum('dnn,dn->dn', A_bar, current)
                A_powers_B_parts[i] = current

        K = torch.einsum('nd, ldn -> ld', C, A_powers_B_parts).transpose(0, 1)  # (D, L)

        fft_len = 2 * L
        k_f = torch.fft.rfft(K, n=fft_len)  # (D, fft_len//2 + 1)
        x_f = torch.fft.rfft(x.transpose(1, 2), n=fft_len)  # (B, D, fft_len//2 + 1)
        y_f = k_f[None] * x_f  # (B, D, fft_len//2 + 1)
        y = torch.fft.irfft(y_f, n=fft_len)[..., :L]  # (B, D, L)

        return y.transpose(1, 2) + x

class MlpForMamba(nn.Module):
    def __init__(self, in_features, hidden_mult=4):
        super().__init__()
        hidden_features = in_features * hidden_mult
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LWMMamba(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        d_state = 16
        d_conv = 4
        self.d_model = out_features
        if self.d_model == 0 or in_features % self.d_model != 0:
            raise ValueError(f"in_features ({in_features})  out_features ({out_features})")
        self.seq_len = in_features // self.d_model
        self.d_inner = self.d_model

        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.silu = nn.SiLU()

        self.linear_b_gate = nn.Linear(self.d_model, self.d_inner)
        self.mlp_b = MlpForMamba(self.d_inner)
        
        self.linear_b_val = nn.Linear(self.d_model, self.d_inner)
        
        self.conv1d_b_1 = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.ssm_b_1 = ParallelSSM(d_inner=self.d_inner, d_state=d_state, A_matrix=self.A)
        
        self.conv1d_b_2 = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.ssm_b_2 = ParallelSSM(d_inner=self.d_inner, d_state=d_state, A_matrix=self.A)

        self.linear_d_gate = nn.Linear(self.d_model, self.d_inner)
        self.mlp_d = MlpForMamba(self.d_inner)
        
        self.linear_d_val = nn.Linear(self.d_model, self.d_inner)
       
        self.conv1d_d_1 = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.ssm_d_1 = ParallelSSM(d_inner=self.d_inner, d_state=d_state, A_matrix=self.A)
       
        self.conv1d_d_2 = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv, bias=True, groups=self.d_inner, padding=d_conv - 1)
        self.ssm_d_2 = ParallelSSM(d_inner=self.d_inner, d_state=d_state, A_matrix=self.A)

        self.final_linear = nn.Linear(self.d_inner, self.d_model)
        self.ln = nn.LayerNorm(self.d_model)

    def _process_parallel_branch(self, x, conv_layer, ssm_layer):
        x_conv = x.transpose(1, 2)
        x_conv = conv_layer(x_conv)
        x_conv = x_conv[:, :, :self.seq_len]
        x_conv = x_conv.transpose(1, 2)
        return ssm_layer(x_conv)

    def forward(self, x_breast, x_density):
        B = x_breast.shape[0]
        x_breast_seq = x_breast.view(B, self.seq_len, self.d_model)
        x_density_seq = x_density.view(B, self.seq_len, self.d_model)
        residual_b = x_breast_seq
        residual_d = x_density_seq

        gate_b = self.silu(self.mlp_b(self.linear_b_gate(x_breast_seq)))
        val_b = self.linear_b_val(x_breast_seq)
        
        val_b_1_ssm = self._process_parallel_branch(val_b, self.conv1d_b_1, self.ssm_b_1)
        val_b_2_ssm = self._process_parallel_branch(val_b, self.conv1d_b_2, self.ssm_b_2)
        
        out_b = (gate_b * val_b_1_ssm) + (gate_b * val_b_2_ssm)

        gate_d = self.silu(self.mlp_d(self.linear_d_gate(x_density_seq)))
        val_d = self.linear_d_val(x_density_seq)
        
        val_d_1_ssm = self._process_parallel_branch(val_d, self.conv1d_d_1, self.ssm_d_1)
        val_d_2_ssm = self._process_parallel_branch(val_d, self.conv1d_d_2, self.ssm_d_2)

        out_d = (gate_d * val_d_1_ssm) + (gate_d * val_d_2_ssm)

        combined = out_b + out_d
        fused = self.final_linear(combined)
        fused = self.ln(fused)
        
        output_seq = fused + residual_b + residual_d
        output_vec = output_seq.mean(dim=1)
        
        return output_vec
    
class SMLPBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(SMLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        return x

class SSSMBlock(nn.Module):
    def __init__(self, channels):
        super(SSSMBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.state_matrix = nn.Parameter(torch.randn(channels, channels))

    def forward(self, x):
        x_conv = self.conv(x)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).transpose(1, 2)
        x_ssm = torch.matmul(x_flat, self.state_matrix).transpose(1, 2).view(b, c, h, w)
        return x_ssm + x_conv

class SingleInputLWMMamba(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SingleInputLWMMamba, self).__init__()
        d_model = in_features
        d_inner = hidden_features

        # --- Gate Path (Left side of diagram) ---
        self.linear_gate = nn.Linear(d_model, d_inner)
        # SMLPBlock is used as the 'MLP' in the diagram
        self.mlp = SMLPBlock(d_inner, d_inner) 
        self.silu = nn.SiLU()

        # --- Value Path (Right side of diagram) ---
        # Initial Linear layer for the value path
        self.linear_value = nn.Linear(d_model, d_inner)
        
        # Sub-branch 'a' (top-right Conv->SSM)
        self.conv_a = nn.Linear(d_inner, d_inner)
        self.ssm_a = nn.Linear(d_inner, d_inner) 
        self.w_a = nn.Parameter(torch.randn(d_inner)) 

        # Sub-branch 'b' (bottom-right Conv->SSM)
        self.conv_b = nn.Linear(d_inner, d_inner) 
        self.ssm_b = nn.Linear(d_inner, d_inner)  
        self.w_b = nn.Parameter(torch.randn(d_inner)) 

        self.final_linear = nn.Linear(d_inner, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        identity = x

        # --- Gate Path ---
        gate = self.linear_gate(x)
        gate = self.mlp(gate)
        gate = self.silu(gate)

        # --- Value Path ---
        value = self.linear_value(x)
        
        val_a = self.conv_a(value)
        val_a = self.ssm_a(val_a)
        
        val_b = self.conv_b(value)
        val_b = self.ssm_b(val_b)
        
        # --- Gating, Weighting, and Combining ---
        gated_a = gate * (val_a * self.w_a)
        gated_b = gate * (val_b * self.w_b)
        combined = gated_a + gated_b
        
        output = self.final_linear(combined)
        output = self.ln(output)

        return output + identity

class MomentChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(MomentChannelAttention, self).__init__()
        self.fc_mean = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc_var = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc_out = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        
        # Calculate variance safely
        var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
        var = torch.where(torch.isnan(var), torch.zeros_like(var), var)
        
        mean_attention = F.relu(self.fc_mean(mean))
        var_attention = F.relu(self.fc_var(var))
        attention = self.fc_out(mean_attention + var_attention)
        attention = self.sigmoid(attention)

        return x * attention


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EdgeNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(EdgeNetBlock, self).__init__()
        self.laplacian_kernel = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=in_channels # Make it a depthwise filter
        )
        laplacian_kernel = torch.tensor(
            [[[[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]]], dtype=torch.float32
        )
        # Initialize weights correctly for groups
        weight = laplacian_kernel.repeat(in_channels, 1, 1, 1)
        self.laplacian_kernel.weight = nn.Parameter(weight, requires_grad=False)


    def forward(self, x):
        edge_features = self.laplacian_kernel(x)
        return edge_features

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1_depth = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2_depth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv2_point = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        
        x = self.conv1_depth(x)
        x = self.conv1_point(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2_depth(x)
        x = self.conv2_point(x)
        x = self.bn2(x)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        x += residual
        return self.relu(x)

import torchvision.models as models

class LWMMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model 
        
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.silu = nn.SiLU()

        
        self.linear_b_gate = nn.Linear(self.d_model, self.d_inner)
        self.mlp_b = MlpForMamba(self.d_inner)
        self.linear_b_val = nn.Linear(self.d_model, self.d_inner)
        self.conv1d_b_1 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.ssm_b_1 = ParallelSSM(self.d_inner, d_state, A_matrix=self.A)
        self.conv1d_b_2 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.ssm_b_2 = ParallelSSM(self.d_inner, d_state, A_matrix=self.A)

       
        self.linear_d_gate = nn.Linear(self.d_model, self.d_inner)
        self.mlp_d = MlpForMamba(self.d_inner)
        self.linear_d_val = nn.Linear(self.d_model, self.d_inner)
        self.conv1d_d_1 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.ssm_d_1 = ParallelSSM(self.d_inner, d_state, A_matrix=self.A)
        self.conv1d_d_2 = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.ssm_d_2 = ParallelSSM(self.d_inner, d_state, A_matrix=self.A)

    def _process_parallel_branch(self, x, conv_layer, ssm_layer):
        L = x.shape[1]
        x_conv = x.transpose(1, 2)
        x_conv = conv_layer(x_conv)[:, :, :L].transpose(1, 2)
        return ssm_layer(x_conv)

    def forward(self, x_breast_seq, x_density_seq):
        residual_b, residual_d = x_breast_seq, x_density_seq
        
        gate_b = self.silu(self.mlp_b(self.linear_b_gate(x_breast_seq)))
        val_b = self.linear_b_val(x_breast_seq)
        val_b_1_ssm = self._process_parallel_branch(val_b, self.conv1d_b_1, self.ssm_b_1)
        val_b_2_ssm = self._process_parallel_branch(val_b, self.conv1d_b_2, self.ssm_b_2)
        
        out_b = gate_b * (val_b_1_ssm + val_b_2_ssm) 

        gate_d = self.silu(self.mlp_d(self.linear_d_gate(x_density_seq)))
        val_d = self.linear_d_val(x_density_seq)
        val_d_1_ssm = self._process_parallel_branch(val_d, self.conv1d_d_1, self.ssm_d_1)
        val_d_2_ssm = self._process_parallel_branch(val_d, self.conv1d_d_2, self.ssm_d_2)
        
        out_d = gate_d * (val_d_1_ssm + val_d_2_ssm)

       
        return out_b + residual_b, out_d + residual_d

class MambaModel(nn.Module):
    def __init__(self, in_channels, feature_dim=32, num_lwm_mamba_blocks=3):
        super(MambaModel, self).__init__()
        self.feature_dim = feature_dim
        stem_channels = feature_dim
        self.feature_dim = feature_dim; self.num_lwm_mamba_blocks = num_lwm_mamba_blocks

        self.conv_stem_breast = ConvStem(3, stem_channels)
        self.conv_stem_density = ConvStem(1, stem_channels)

        self.edge_net_block = EdgeNetBlock(stem_channels)
        self.res_block = ResBlock(stem_channels, stem_channels)
        
        self.lwm_mamba_blocks = nn.ModuleList(
            [LWMMambaBlock(d_model=stem_channels) for _ in range(self.num_lwm_mamba_blocks)]
        )
        self.fusion_linear = nn.Linear(stem_channels, self.feature_dim)
        self.fusion_ln = nn.LayerNorm(self.feature_dim)

        channels = [stem_channels, 64,128,self.feature_dim]
        self.res_blocks_attention = nn.Sequential(
            *[ResBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        )
        self.mca = MomentChannelAttention(self.feature_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.single_input_lwm_mamba = SingleInputLWMMamba(self.feature_dim, self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, 28)

    def get_grad_cam(self, x):
        return torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    def forward(self, x):
        if x.size(1) == 3:
            x_breast, x_density = x, self.get_grad_cam(x)
        elif x.size(1) == 4:
            x_breast, x_density = x[:, :3, :, :], x[:, 3:4, :, :]
        else:
            raise ValueError(f"Input channel size {x.size(1)} not supported.")

        x_breast = self.conv_stem_breast(x_breast)
        x_density = self.conv_stem_density(x_density)

        x_breast = self.edge_net_block(x_breast)
        x_density = self.res_block(x_density)

        B, C, H, W = x_breast.shape
        #  (B, C, H, W) ->  (B, L, D) , L=H*W, D=C
        x_breast_seq = x_breast.flatten(2).transpose(1, 2)
        x_density_seq = x_density.flatten(2).transpose(1, 2)

        for block in self.lwm_mamba_blocks:
            x_breast_seq, x_density_seq = block(x_breast_seq, x_density_seq)

        combined_seq = x_breast_seq + x_density_seq
        fused_seq = self.fusion_linear(combined_seq)
        fused_seq = self.fusion_ln(fused_seq)
        
        combined_features = fused_seq.mean(dim=1)

        x_attention = self.res_blocks_attention(x_breast + x_density)
        x_attention = self.mca(x_attention)
        x_attention = self.gap(x_attention).view(x_attention.size(0), -1)
        x_attention = self.single_input_lwm_mamba(x_attention)

        final_features = combined_features + x_attention
        output = self.fc(final_features)
        
        return output
    
    def freeze_parameters(self, freeze_list=None):
        if freeze_list is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer in name for layer in freeze_list):
                    param.requires_grad = False

    def unfreeze_parameters(self, unfreeze_list=None):
        if unfreeze_list is None:
           for param in self.parameters():
                    param.requires_grad = True     
        else:
             for name, param in self.named_parameters():
                    if any(layer in name for layer in unfreeze_list):
                        param.requires_grad = True
    
    def freeze_parameters(self, freeze_list=None):
        if freeze_list is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer in name for layer in freeze_list):
                    param.requires_grad = False

    def unfreeze_parameters(self, unfreeze_list=None):
        if unfreeze_list is None:
           for param in self.parameters():
                    param.requires_grad = True     
        else:
             for name, param in self.named_parameters():
                    if any(layer in name for layer in unfreeze_list):
                        param.requires_grad = True
               
def load_pretrained_weights(model, weights_path):
    try:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")

# Usage example:
def fine_tune_model(model, weights_path):
    # Load pre-trained weights
    load_pretrained_weights(model, weights_path)

    # Freeze layers
    freeze_layers = [
        'conv_stem_breast', 'conv_stem_density',
        'edge_net_block', 'res_block',
        'lwm_mamba', 'res_blocks_attention'
    ]
    model.freeze_parameters(freeze_layers)

    # Unfreeze the last few layers for fine-tuning
    unfreeze_layers = ['mca', 'single_input_lwm_mamba', 'fc']
    model.unfreeze_parameters(unfreeze_layers)

    return model

