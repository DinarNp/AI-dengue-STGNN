# debug_dataset.py
from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from data.dataset import DengueDataset
import torch

config = Config()
preprocessor = DengueDataPreprocessor(config)

# Load dan preprocess data
df = preprocessor.load_data("data/test2.csv")
features, targets, metadata = preprocessor.preprocess_data(df)

print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Number of nodes: {metadata['n_nodes']}")
print(f"Node IDs: {metadata['node_ids']}")

# Test dataset creation
dataset = DengueDataset(features, targets, metadata, 
                       window_size=config.WINDOW_SIZE, 
                       forecast_horizon=config.FORECAST_HORIZON)

print(f"Dataset length: {len(dataset)}")

# Test one sample
if len(dataset) > 0:
    sample_features, sample_target, node_idx = dataset[0]
    print(f"Sample features shape: {sample_features.shape}")
    print(f"Sample target: {sample_target}")
    print(f"Node index: {node_idx}")