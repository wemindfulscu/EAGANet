try:
    from thop import profile
    x = torch.randn(1, 4, 224, 224)  # Batch size 1, 4 channels, 64x64 size
    model = MambaModel(in_channels=3, feature_dim=32)

    # 计算FLOPS
    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"Model FLOPS: {flops / 1e9:.2f} GFLOPS")  # 转换为GFLOPS
    print(f"Model Params: {params / 1e6:.2f} MParams")

    output = model(x)
    print(f"Output shape: {output.shape}")  # Should output [2, 28]
except ImportError:
    print("thop is not installed. Skipping FLOPS calculation.")
    x = torch.randn(1, 4, 224, 224)
    model = MambaModel(in_channels=3, feature_dim=32)
    output = model(x)
    print(f"Output shape: {output.shape}")