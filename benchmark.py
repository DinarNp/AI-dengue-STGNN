# realistic_benchmark.py
import torch
import time

def benchmark_neural_network():
    print("=== Neural Network-like Benchmark ===")
    
    # Simulate LSTM + Linear layers (like our model)
    batch_size, seq_len, hidden_size = 16, 4, 128
    
    # CPU Test
    device_cpu = torch.device('cpu')
    
    # Create typical neural network tensors
    x_cpu = torch.randn(batch_size, seq_len, hidden_size, device=device_cpu)
    lstm_cpu = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device_cpu)
    linear_cpu = torch.nn.Linear(hidden_size, 1).to(device_cpu)
    
    # Warmup
    for _ in range(5):
        out_cpu, _ = lstm_cpu(x_cpu)
        result_cpu = linear_cpu(out_cpu[:, -1, :])
    
    # Benchmark CPU
    start = time.time()
    for _ in range(100):  # Multiple iterations
        out_cpu, _ = lstm_cpu(x_cpu)
        result_cpu = linear_cpu(out_cpu[:, -1, :])
    cpu_time = time.time() - start
    
    # MPS Test
    device_mps = torch.device('mps')
    
    x_mps = torch.randn(batch_size, seq_len, hidden_size, device=device_mps)
    lstm_mps = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device_mps)
    linear_mps = torch.nn.Linear(hidden_size, 1).to(device_mps)
    
    # Warmup MPS
    for _ in range(5):
        out_mps, _ = lstm_mps(x_mps)
        result_mps = linear_mps(out_mps[:, -1, :])
        torch.mps.synchronize()
    
    # Benchmark MPS
    start = time.time()
    for _ in range(100):
        out_mps, _ = lstm_mps(x_mps)
        result_mps = linear_mps(out_mps[:, -1, :])
    torch.mps.synchronize()  # Important!
    mps_time = time.time() - start
    
    print(f"CPU time (100 iterations): {cpu_time:.4f} seconds")
    print(f"MPS time (100 iterations): {mps_time:.4f} seconds")
    print(f"Speedup: {cpu_time/mps_time:.2f}x")
    
    if cpu_time < mps_time:
        print("❌ MPS is slower - use CPU for this model size")
        return 'cpu'
    else:
        print("✅ MPS is faster - use MPS for training")
        return 'mps'

if __name__ == "__main__":
    faster_device = benchmark_neural_network()
    
    # Test larger model
    print("\n=== Larger Model Test ===")
    batch_size, seq_len, hidden_size = 32, 8, 256
    
    # Repeat test with larger model
    device_cpu = torch.device('cpu')
    device_mps = torch.device('mps')
    
    # Large tensors
    x_cpu = torch.randn(batch_size, seq_len, hidden_size, device=device_cpu)
    x_mps = torch.randn(batch_size, seq_len, hidden_size, device=device_mps)
    
    # CPU
    start = time.time()
    for _ in range(10):
        result = torch.matmul(x_cpu, x_cpu.transpose(-2, -1))
        result = torch.nn.functional.softmax(result, dim=-1)
    cpu_large_time = time.time() - start
    
    # MPS
    start = time.time()
    for _ in range(10):
        result = torch.matmul(x_mps, x_mps.transpose(-2, -1))
        result = torch.nn.functional.softmax(result, dim=-1)
    torch.mps.synchronize()
    mps_large_time = time.time() - start
    
    print(f"Large model - CPU: {cpu_large_time:.4f}s, MPS: {mps_large_time:.4f}s")
    print(f"Large model speedup: {cpu_large_time/mps_large_time:.2f}x")