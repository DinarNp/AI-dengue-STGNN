# test_model.py
from config.config import Config
from models.stgnn import STGNNDenguePredictor

config = Config()
input_dim = 25  # Sesuaikan dengan jumlah features Anda
num_nodes = 1   # Sesuai dengan data Anda

model = STGNNDenguePredictor(config, input_dim, num_nodes)
print(f"Model created successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")