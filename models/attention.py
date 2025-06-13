import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatioTemporalAttention(nn.Module):
    """Spatio-temporal attention mechanism"""
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_heads: int = 4):
        super(SpatioTemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Remove these debug lines:
        # print(f"DEBUG Attention init:")
        # print(f"  feature_dim: {feature_dim}")
        # print(f"  hidden_dim: {hidden_dim}")
        # print(f"  num_heads: {num_heads}")
        # print(f"  head_dim: {self.head_dim}")
        
        # ... rest of __init__
        
        # Temporal attention
        self.temporal_query = nn.Linear(feature_dim, hidden_dim)
        self.temporal_key = nn.Linear(feature_dim, hidden_dim)
        self.temporal_value = nn.Linear(feature_dim, hidden_dim)
        
        # Spatial attention
        self.spatial_query = nn.Linear(feature_dim, hidden_dim)
        self.spatial_key = nn.Linear(feature_dim, hidden_dim)
        self.spatial_value = nn.Linear(feature_dim, hidden_dim)
        
        # Output projection
        self.temporal_out = nn.Linear(hidden_dim, feature_dim)
        self.spatial_out = nn.Linear(hidden_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention across time steps"""
        batch_size, time_steps, num_nodes, feature_dim = x.shape
        
        # Reshape for attention computation
        x_reshaped = x.view(batch_size * num_nodes, time_steps, feature_dim)
        
        # Compute Q, K, V
        Q = self.temporal_query(x_reshaped)
        K = self.temporal_key(x_reshaped)
        V = self.temporal_value(x_reshaped)
        
        # Multi-head attention
        Q = Q.view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * num_nodes, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size * num_nodes, time_steps, self.hidden_dim)
        
        # Output projection
        output = self.temporal_out(attended)
        output = output.view(batch_size, time_steps, num_nodes, feature_dim)
        
        return output
    
    def spatial_attention(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, feature_dim = x.shape
        
        # print(f"DEBUG spatial_attention:")
        # print(f"  x shape: {x.shape}")
        # print(f"  adj_matrix shape: {adj_matrix.shape}")
        
        # Fix adjacency matrix size to match actual batch nodes
        if adj_matrix.shape[0] != num_nodes:
            # print(f"  Resizing adjacency from {adj_matrix.shape} to {num_nodes}x{num_nodes}")
            # Take subset or pad adjacency matrix
            if adj_matrix.shape[0] > num_nodes:
                # Take first num_nodes x num_nodes
                adj_matrix = adj_matrix[:num_nodes, :num_nodes]
            else:
                # Pad with zeros (shouldn't happen but handle it)
                pad_size = num_nodes - adj_matrix.shape[0]
                adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size), value=0)
        
        # print(f"  Fixed adj_matrix shape: {adj_matrix.shape}")
        
        # Continue with normal attention...
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            
            Q = self.spatial_query(x_t)
            K = self.spatial_key(x_t)
            V = self.spatial_value(x_t)
            
            # Multi-head attention
            Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            
            # Now adjacency mask should match
            adj_mask = adj_matrix.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores = scores.masked_fill(adj_mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attended = torch.matmul(attention_weights, V)
            attended = attended.transpose(1, 2).contiguous().view(
                batch_size, num_nodes, self.hidden_dim)
            
            output_t = self.spatial_out(attended)
            outputs.append(output_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass with both temporal and spatial attention"""
        # Apply temporal attention
        temp_attended = self.temporal_attention(x)
        temp_attended = self.layer_norm(temp_attended + x)
        
        # Apply spatial attention
        spat_attended = self.spatial_attention(temp_attended, adj_matrix)
        spat_attended = self.layer_norm(spat_attended + temp_attended)
        
        return spat_attended