import torch
import numpy as np
from config.config import Config
from data.dataset import DengueDataset, collate_fn
from torch.utils.data import DataLoader

# Test data flow
config = Config()

# Create dummy data
n_samples = 100
n_features = 26
n_nodes = 5

features = np.random.randn(n_samples, n_features)
targets = np.random.randn(n_samples)

metadata = {
    'n_nodes': n_nodes,
    'feature_cols': [f'feat_{i}' for i in range(n_features)]
}

# Create dataset
dataset = DengueDataset(
    features, targets, metadata,
    window_size=config.WINDOW_SIZE,
    forecast_horizon=config.FORECAST_HORIZON
)

print(f"Dataset size: {len(dataset)}")

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collate_fn,
    drop_last=True
)

# Test one batch
for batch_features, batch_targets in loader:
    print(f"Batch features shape: {batch_features.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
    print(f"Expected: ({8}, {config.WINDOW_SIZE}, {n_nodes}, {n_features})")
    break

print("✅ Data flow test passed!")