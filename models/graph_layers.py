import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer with attention"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.attention = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            nn.LeakyReLU(0.2),
            nn.Linear(out_features, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_nodes, in_features = x.shape
        
        # Fix adjacency matrix size to match batch
        if adj_matrix.shape[0] != num_nodes:
            if adj_matrix.shape[0] > num_nodes:
                adj_matrix = adj_matrix[:num_nodes, :num_nodes]
            else:
                pad_size = num_nodes - adj_matrix.shape[0]
                adj_matrix = F.pad(adj_matrix, (0, pad_size, 0, pad_size), value=0)
        
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # [batch_size, num_nodes, in_features]
            
            # Linear transformation
            h = torch.matmul(x_t, self.weight) + self.bias
            
            # Attention-based aggregation
            h_expanded = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            h_transposed = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            
            # Concatenate for attention
            attention_input = torch.cat([h_expanded, h_transposed], dim=-1)
            attention_scores = self.attention(attention_input).squeeze(-1)
            
            # Apply adjacency matrix mask
            adj_mask = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            attention_scores = attention_scores.masked_fill(adj_mask == 0, -1e9)
            
            # Softmax attention weights
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Aggregate features
            aggregated = torch.matmul(attention_weights, h)
            
            # Layer normalization and residual connection
            if self.in_features == self.out_features:
                aggregated = self.layer_norm(aggregated + x_t)
            else:
                aggregated = self.layer_norm(aggregated)
            
            outputs.append(aggregated.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return F.relu(output)