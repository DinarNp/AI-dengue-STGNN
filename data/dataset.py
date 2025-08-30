import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple

class DengueDataset(Dataset):
    """Dataset class for dengue prediction"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, metadata: Dict,
                 window_size: int = 4, forecast_horizon: int = 1):
        self.features = features
        self.targets = targets
        self.metadata = metadata
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_nodes = metadata['n_nodes']
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Dict]:
        """Create sequences for training - FIXED VERSION"""
        sequences = []
        
        try:
            # Karena data Anda berdasarkan Region, bukan node index
            # Kita perlu group by region dulu
            n_samples = len(self.features)
            
            # Asumsi data sudah diurutkan berdasarkan Region dan Week
            samples_per_node = n_samples // self.n_nodes
            
            for node_idx in range(self.n_nodes):
                start_idx = node_idx * samples_per_node
                end_idx = start_idx + samples_per_node
                
                if end_idx > len(self.features):
                    end_idx = len(self.features)
                
                node_features = self.features[start_idx:end_idx]
                node_targets = self.targets[start_idx:end_idx]
                
                # Create sliding windows
                for i in range(len(node_features) - self.window_size - self.forecast_horizon + 1):
                    seq_features = node_features[i:i + self.window_size]
                    seq_target = node_targets[i + self.window_size:i + self.window_size + self.forecast_horizon]
                    
                    # Safety check for target
                    target_value = seq_target[0] if len(seq_target) > 0 else 0.0
                    
                    sequences.append({
                        'features': seq_features,
                        'target': target_value,
                        'node_idx': node_idx
                    })
            
            print(f"✅ Created {len(sequences)} sequences for {self.n_nodes} nodes")
            return sequences
            
        except Exception as e:
            print(f"❌ Error creating sequences: {e}")
            # Fallback: create simple sequences
            sequences = []
            for i in range(len(self.features) - self.window_size):
                seq_features = self.features[i:i + self.window_size]
                target_value = self.targets[i + self.window_size] if i + self.window_size < len(self.targets) else 0.0
                
                sequences.append({
                    'features': seq_features,
                    'target': target_value,
                    'node_idx': 0  # Default node
                })
            
            print(f"✅ Created {len(sequences)} fallback sequences")
            return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq = self.sequences[idx]
        
        features = torch.FloatTensor(seq['features'])
        target = torch.FloatTensor([seq['target']])
        node_idx = seq['node_idx']
        
        return features, target, node_idx

def collate_fn(batch):
    """Custom collate function for batch processing - FIXED VERSION"""
    try:
        features_list, targets_list, node_indices = zip(*batch)
        
        # Group by time steps and nodes
        batch_size = len(batch)
        time_steps = features_list[0].shape[0]
        feature_dim = features_list[0].shape[1]
        
        # Get max nodes from all samples (should be consistent)
        max_nodes = max(node_indices) + 1
        
        # Create batch tensor with ALL possible nodes
        batch_features = torch.zeros(1, time_steps, max_nodes, feature_dim)
        batch_targets = torch.zeros(1, max_nodes)
        
        # Fill only the nodes that exist in this batch
        for i, (features, target, node_idx) in enumerate(batch):
            if node_idx < max_nodes:  # Safety check
                batch_features[0, :, node_idx, :] = features
                batch_targets[0, node_idx] = target
        
        return batch_features, batch_targets
        
    except Exception as e:
        print(f"❌ Error in collate_fn: {e}")
        print(f"   Batch size: {len(batch)}")
        print(f"   Batch sample: {batch[0] if batch else 'Empty batch'}")
        raise e