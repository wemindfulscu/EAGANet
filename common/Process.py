import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.detach().cpu()[0]  # [C, H_feat, W_feat]
        activations = self.activations.detach().cpu()[0] # [C, H_feat, W_feat]

        weights = torch.mean(gradients, dim=[1, 2])  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = F.relu(cam) 
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        H, W = input_tensor.shape[2:]
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)
        
        return cam.squeeze()

class GradCAMPreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Model_PATH = ""
        self.resnet = models.resnet50(weights=Model_PATH).to(self.device)
        self.resnet.eval()

        self.grad_cam = GradCAM(self.resnet, self.resnet.layer4)

        self.preprocess_for_resnet = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.preprocess_for_concat = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(), 
        ])
    
    def __call__(self, pil_image):
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        input_tensor_resnet = self.preprocess_for_resnet(pil_image).unsqueeze(0).to(self.device)
        
        cam_map = self.grad_cam(input_tensor_resnet).cpu() # Shape: [H, W]
        
        original_image_tensor = self.preprocess_for_concat(pil_image) # Shape: [3, H, W]

        combined_tensor = torch.cat([original_image_tensor, cam_map.unsqueeze(0)], dim=0)
        
        return combined_tensor

if __name__ == '__main__':
    preprocessor = GradCAMPreprocessor(image_size=(224, 224))
    print("Grad-CAM Preprocessor is ready.")

    try:
        url = "xxx.jpg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        print("Successfully loaded example image from URL.")
    except Exception as e:
        print(f"Failed to load image from URL: {e}. Creating a dummy image.")
        img = Image.new('RGB', (300, 300), color = 'red')

    four_channel_tensor = preprocessor(img)

    print(f"\nOutput tensor shape: {four_channel_tensor.shape}")
    assert four_channel_tensor.shape == (4, 224, 224)

    original_img_ch = four_channel_tensor[:3, :, :]
    cam_ch = four_channel_tensor[3, :, :]

    original_img_display = original_img_ch.permute(1, 2, 0).numpy()
    cam_display = cam_ch.numpy()

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img.resize((224, 224)))
    plt.title("Original Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam_display, cmap='jet')
    plt.title("Generated Grad-CAM (Channel 4)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original_img_display)
    plt.imshow(cam_display, cmap='jet', alpha=0.5) 
    plt.title("Overlay View")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # model = MambaModel(in_channels=3, feature_dim=32)
    # output = model(four_channel_tensor.unsqueeze(0)) # unsqueeze(0) to add batch dimension
    # print(f"\nMock model output shape: {output.shape}")