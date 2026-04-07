# data/dataset.py - COMPLETE REWRITE

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple, List

class DengueDataset(Dataset):
    """Dataset for dengue prediction - FIXED to ensure all nodes in each sample"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 metadata: Dict, window_size: int, forecast_horizon: int):
        """
        Args:
            features: (n_samples, n_features) array
            targets: (n_samples,) array
            metadata: dict with 'n_nodes' and other info
            window_size: sequence length
            forecast_horizon: prediction steps ahead
        """
        self.features = features
        self.targets = targets
        self.metadata = metadata
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_nodes = metadata['n_nodes']
        
        # Create sequences where each sample contains ALL nodes
        # self.sequences = self._create_sequences()
        self.sequences = self._create_sequences_fixed()
        
        print(f"✅ Created {len(self.sequences)} complete sequences (each with all {self.n_nodes} nodes)")
        
    def _create_sequences(self) -> List[Dict]:
<<<<<<< Updated upstream
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
=======
        """
        Create sequences ensuring each sample has data for ALL nodes
        
        Returns list of dicts, each containing:
            - features: (window_size, n_nodes, n_features)
            - targets: (n_nodes,)
        """
        sequences = []
        
        # Calculate samples per node
        n_total = len(self.features)
        samples_per_node = n_total // self.n_nodes
        
        print(f"🔧 Creating sequences:")
        print(f"   Total samples: {n_total}")
        print(f"   Nodes: {self.n_nodes}")
        print(f"   Samples per node: {samples_per_node}")
        
        # Maximum number of complete sequences
        max_sequences = samples_per_node - self.window_size - self.forecast_horizon + 1
        
        # Create sequences
        for seq_idx in range(max_sequences):
            # Initialize arrays for this sequence
            seq_features = np.zeros((self.window_size, self.n_nodes, self.features.shape[1]))
            seq_targets = np.zeros(self.n_nodes)
            
            # Fill in data for each node
            for node_idx in range(self.n_nodes):
                # Get data range for this node
                node_start = node_idx * samples_per_node
                
                # Extract sequence for this node
                start_idx = node_start + seq_idx
                end_idx = start_idx + self.window_size
                target_idx = start_idx + self.window_size + self.forecast_horizon - 1
                
                # Store features and target
                seq_features[:, node_idx, :] = self.features[start_idx:end_idx]
                seq_targets[node_idx] = self.targets[target_idx]
            
            sequences.append({
                'features': seq_features,
                'targets': seq_targets
            })
        
        return sequences
>>>>>>> Stashed changes
    
    def _create_sequences_fixed(self) -> List[Dict]:
        """
        ✅ FIXED: Create sequences that work with split data
        
        Strategy:
        - Data is still organized by location (maintained by location-aware split)
        - For each location, create sequences from its contiguous samples
        - Each sequence contains data for ALL nodes at the same time window
        """
        
        sequences = []
        n_total = len(self.features)
        samples_per_node = n_total // self.n_nodes
        
        if n_total % self.n_nodes != 0:
            print(f"   ⚠️  Data not evenly divisible: {n_total} samples, {self.n_nodes} nodes")
            # Trim to make it divisible
            n_total = (n_total // self.n_nodes) * self.n_nodes
            samples_per_node = n_total // self.n_nodes
        
        # Maximum sequences per node
        max_seq_per_node = samples_per_node - self.window_size - self.forecast_horizon + 1
        
        if max_seq_per_node <= 0:
            print(f"   ⚠️  Not enough samples per node to create sequences!")
            print(f"      Samples per node: {samples_per_node}")
            print(f"      Need: {self.window_size + self.forecast_horizon}")
            return []
        
        # Create sequences
        for seq_idx in range(max_seq_per_node):
            # Initialize arrays for this sequence (all nodes)
            seq_features = np.zeros((self.window_size, self.n_nodes, self.features.shape[1]))
            seq_targets = np.zeros(self.n_nodes)
            
            # Fill data for each node
            for node_idx in range(self.n_nodes):
                # Get this node's data range
                node_start = node_idx * samples_per_node
                
                # Extract sequence window
                start_idx = node_start + seq_idx
                end_idx = start_idx + self.window_size
                target_idx = start_idx + self.window_size + self.forecast_horizon - 1
                
                # Safety check
                if target_idx < node_start + samples_per_node:
                    seq_features[:, node_idx, :] = self.features[start_idx:end_idx]
                    seq_targets[node_idx] = self.targets[target_idx]
            
            sequences.append({
                'features': seq_features,
                'targets': seq_targets
            })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Return: (window_size, n_nodes, n_features), (n_nodes,)
        return (
            torch.FloatTensor(seq['features']),
            torch.FloatTensor(seq['targets'])
        )


def collate_fn(batch):
<<<<<<< Updated upstream
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
=======
    """
    Collate function - SIMPLIFIED since each sample already has all nodes
    
    Input: List of tuples (features, targets)
        - features: (window_size, n_nodes, n_features)
        - targets: (n_nodes,)
    
    Output:
        - batch_features: (batch_size, window_size, n_nodes, n_features)
        - batch_targets: (batch_size, n_nodes)
    """
    if len(batch) == 0:
        return None, None
    
    # Stack all samples
    features_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    
    # Stack into batches
    batch_features = torch.stack(features_list, dim=0)  # (batch, window, nodes, features)
    batch_targets = torch.stack(targets_list, dim=0)    # (batch, nodes)
    
    return batch_features, batch_targets
>>>>>>> Stashed changes
