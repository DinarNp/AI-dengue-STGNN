# test_mps.py
import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Test tensor operations on MPS
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.mps.synchronize()  # Wait for MPS operations to complete
    end = time.time()
    
    print(f"✅ MPS test successful! Matrix multiplication took {end-start:.4f} seconds")
    print(f"Result shape: {z.shape}")
else:
    print("❌ MPS not available")