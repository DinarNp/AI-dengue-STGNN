# models/stgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import traceback

class SpatioTemporalAttention(nn.Module):
    """Multi-head attention for spatio-temporal sequences - FIXED"""
    
    def __init__(self, config, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_heads = num_heads
        # ✅ CRITICAL FIX: Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            # Adjust hidden_dim to nearest multiple
            adjusted_hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
            print(f"⚠️ Adjusted hidden_dim from {hidden_dim} to {adjusted_hidden_dim} (divisible by {num_heads})")
            hidden_dim = adjusted_hidden_dim

        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
        Returns:
            out: (batch_size, seq_len, num_nodes, hidden_dim)
        """
        try:
            batch_size, seq_len, num_nodes, _ = x.shape
            
            # Reshape for multi-head attention: (batch * nodes, seq_len, input_dim)
            x_reshaped = x.permute(0, 2, 1, 3).contiguous()
            x_reshaped = x_reshaped.view(batch_size * num_nodes, seq_len, self.input_dim)
            
            # Project to Q, K, V
            Q = self.query(x_reshaped)  # (batch*nodes, seq_len, hidden_dim)
            K = self.key(x_reshaped)
            V = self.value(x_reshaped)
            
            # Reshape for multi-head: (batch*nodes, seq_len, num_heads, head_dim)
            Q = Q.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim)
            K = K.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim)
            V = V.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim)
            
            # Transpose for attention: (batch*nodes, num_heads, seq_len, head_dim)
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            
            # Store for interpretability
            self.last_attention_weights = attention.detach()
            
            # Apply attention to values
            out = torch.matmul(attention, V)  # (batch*nodes, num_heads, seq_len, head_dim)
            
            # Reshape back
            out = out.transpose(1, 2).contiguous()  # (batch*nodes, seq_len, num_heads, head_dim)
            out = out.view(batch_size * num_nodes, seq_len, self.hidden_dim)
            
            # Output projection
            out = self.out(out)
            
            # Reshape to original: (batch_size, seq_len, num_nodes, hidden_dim)
            out = out.view(batch_size, num_nodes, seq_len, self.hidden_dim)
            out = out.permute(0, 2, 1, 3).contiguous()
            
            return out
            
        except Exception as e:
            print(f"❌ Error in SpatioTemporalAttention: {e}")
            print(f"   Input shape: {x.shape}")
            traceback.print_exc()
            raise


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer - FIXED"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_nodes, features)
            adj: (num_nodes, num_nodes)
        Returns:
            out: (batch_size, num_nodes, out_features)
        """
        try:
            # # x: (batch, nodes, in_features)
            # # Transform features
            # support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
            
            # # Apply graph convolution
            # # adj: (nodes, nodes)
            # # We want: adj @ support
            # output = torch.matmul(adj, support)  # (batch, nodes, out_features)
            
            # output = output + self.bias
            
            batch_size, num_nodes, in_features = x.shape
            
            # Transform features: (batch, nodes, in_feat) x (in_feat, out_feat)
            support = torch.matmul(x, self.weight)  # (batch, nodes, out_feat)
            
            # Expand adjacency for batch dimension
            # adj: (num_nodes, num_nodes) -> (batch, num_nodes, num_nodes)
            adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply graph convolution: (batch, nodes, nodes) x (batch, nodes, features)
            # bmm expects (batch, n, m) x (batch, m, p) -> (batch, n, p)
            output = torch.bmm(adj_expanded, support)  # (batch, nodes, out_feat)
            
            # Add bias
            output = output + self.bias
            
            return self.dropout(output)
            
            
            
        except Exception as e:
            print(f"❌ Error in GraphConvLayer: {e}")
            print(f"   x shape: {x.shape}, adj shape: {adj.shape}")
            traceback.print_exc()
            raise


class STGNNDenguePredictor(nn.Module):
    """Spatio-Temporal Graph Neural Network - COMPLETELY FIXED"""
    
    def __init__(self, config, input_dim: int, hidden_dim: int = 128, 
                 output_dim: int = 1, num_layers: int = 3):
        super(STGNNDenguePredictor, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        print(f"🔧 Initializing STGNN:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Output dim: {output_dim}")
        print(f"   Num layers: {num_layers}")
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # Spatio-temporal attention
        self.st_attention = SpatioTemporalAttention(config, hidden_dim, hidden_dim, num_heads=4)
        
        # Graph convolutional layers
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, config.DROPOUT)
            for _ in range(num_layers)
        ])
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output heads for zero-inflation
        lstm_output_dim = hidden_dim * 2  # bidirectional
        
        # Zero probability head
        self.zero_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  # Ensure non-negative
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - COMPLETELY FIXED
        
        Args:
            x: (batch_size, time_steps, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
            
        Returns:
            Dictionary with predictions, zero_probs, and intermediate outputs
        """
        try:
            # Verify input shape
            if len(x.shape) != 4:
                raise ValueError(f"Expected 4D input, got shape {x.shape}")
            
            batch_size, time_steps, num_nodes, input_dim = x.shape
            
            print(f"🔍 Forward pass:")
            print(f"   Input: {x.shape}")
            print(f"   Adj: {adj_matrix.shape}")
            
            # 1. Input projection
            # Reshape: (batch, time, nodes, features) -> (batch*time*nodes, features)
            x_flat = x.view(-1, input_dim)
            x_proj = self.input_proj(x_flat)
            # Reshape back: (batch, time, nodes, hidden)
            x_proj = x_proj.view(batch_size, time_steps, num_nodes, self.hidden_dim)
            
            print(f"   After input_proj: {x_proj.shape}")
            
            # 2. Spatio-temporal attention
            x_att = self.st_attention(x_proj)
            print(f"   After attention: {x_att.shape}")
            
            # 3. Graph convolutions at each time step
            graph_outputs = []
            for t in range(time_steps):
                x_t = x_att[:, t, :, :]  # (batch, nodes, hidden)
                
                # Apply graph conv layers with residual
                h = x_t
                for i, graph_conv in enumerate(self.graph_convs):
                    h_new = graph_conv(h, adj_matrix)
                    h_new = F.relu(h_new)
                    h = h + h_new  # Residual connection
                
                graph_outputs.append(h)
            
            # Stack time dimension: (batch, time, nodes, hidden)
            x_graph = torch.stack(graph_outputs, dim=1)
            print(f"   After graph_conv: {x_graph.shape}")
            
            # 4. LSTM over time for each node
            # Reshape: (batch*nodes, time, hidden)
            x_lstm_in = x_graph.permute(0, 2, 1, 3).contiguous()
            x_lstm_in = x_lstm_in.view(batch_size * num_nodes, time_steps, self.hidden_dim)
            
            # LSTM
            lstm_out, _ = self.lstm(x_lstm_in)  # (batch*nodes, time, hidden*2)
            
            # Take last time step
            lstm_final = lstm_out[:, -1, :]  # (batch*nodes, hidden*2)
            
            # Reshape: (batch, nodes, hidden*2)
            lstm_final = lstm_final.view(batch_size, num_nodes, -1)
            print(f"   After LSTM: {lstm_final.shape}")
            
            # 5. Prediction heads
            # Zero probability
            zero_probs = self.zero_head(lstm_final)  # (batch, nodes, 1)
            zero_probs = zero_probs.squeeze(-1)  # (batch, nodes)
            
            # Count prediction
            counts = self.count_head(lstm_final)  # (batch, nodes, 1)
            counts = counts.squeeze(-1)  # (batch, nodes)
            
            # Combined prediction (zero-inflated)
            predictions = (1 - zero_probs) * counts
            
            print(f"   Predictions: {predictions.shape}")
            print(f"   Zero probs: {zero_probs.shape}")
            
            return {
                'predictions': predictions,
                'zero_probs': zero_probs,
                'counts': counts,
                'attention_weights': self.st_attention.last_attention_weights
            }
            
        except Exception as e:
            print(f"❌ Error in STGNN forward: {e}")
            print(f"   Input shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
            traceback.print_exc()
            raise


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)