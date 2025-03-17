import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.randn(4, 4).to(mps_device)
    print("Running on MPS device (GPU):", x)
else:
    print("MPS device (GPU) is not available. Running on CPU.")
    x = torch.randn(4, 4)
    print("Running on CPU:", x)

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
