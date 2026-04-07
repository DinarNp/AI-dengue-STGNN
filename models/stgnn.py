# models/stgnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import traceback


class SpatioTemporalAttention(nn.Module):
    """Multi-head attention for spatio-temporal sequences"""
    
    def __init__(self, config, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Ensure divisibility
        if hidden_dim % num_heads != 0:
            adjusted_hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
            print(f"⚠️ Adjusted hidden_dim from {hidden_dim} to {adjusted_hidden_dim}")
            hidden_dim = adjusted_hidden_dim
        
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.last_attention_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
        Returns:
            out: (batch_size, seq_len, num_nodes, hidden_dim)
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Reshape: (batch*nodes, seq_len, input_dim)
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * num_nodes, seq_len, self.input_dim)
        
        # Project to Q, K, V
        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)
        
        # Reshape for multi-head
        Q = Q.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        self.last_attention_weights = attention.detach()
        
        # Apply attention
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size * num_nodes, seq_len, self.hidden_dim)
        out = self.out(out)
        
        # Reshape back
        out = out.view(batch_size, num_nodes, seq_len, self.hidden_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        
        return out


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer - FIXED"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        ✅ FIXED: Ensure adj is on same device as x
        
        Args:
            x: (batch_size, num_nodes, in_features)
            adj: (num_nodes, num_nodes)
        Returns:
            out: (batch_size, num_nodes, out_features)
        """
        # ✅ FIX 1: Move adj to same device as x
        if adj.device != x.device:
            adj = adj.to(x.device)
        
        # ✅ FIX 2: Ensure adj is contiguous
        adj = adj.contiguous()
        
        # Transform features
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        
        # ✅ FIX 3: Use matmul instead of einsum for better MPS compatibility
        # OLD (causes MPS error): output = torch.einsum('ij,bjf->bif', adj, support)
        # NEW: Manual matrix multiplication
        batch_size = support.shape[0]
        num_nodes = support.shape[1]
        
        # Expand adj for batch dimension: (1, num_nodes, num_nodes)
        adj_expanded = adj.unsqueeze(0)
        
        # Matrix multiplication: (batch, num_nodes, num_nodes) @ (batch, num_nodes, features)
        output = torch.bmm(adj_expanded.expand(batch_size, -1, -1), support)
        
        # Add bias
        output = output + self.bias
        
        return self.dropout(output)


class STGNNDenguePredictor(nn.Module):
    """Spatio-Temporal Graph Neural Network - COMPLETE FIX"""
    
    def __init__(self, config, input_dim: int, hidden_dim: int = None, 
                 output_dim: int = 1, num_layers: int = None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Get dimensions from config
        if hidden_dim is None:
            hidden_dim = getattr(config, 'HIDDEN_DIM', 128)
        if num_layers is None:
            num_layers = getattr(config, 'NUM_LAYERS', 3)
        
        num_heads = getattr(config, 'NUM_HEADS', 4)
        
        # Ensure divisibility
        if hidden_dim % num_heads != 0:
            adjusted_hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
            print(f"⚠️ Adjusting hidden_dim from {hidden_dim} to {adjusted_hidden_dim}")
            hidden_dim = adjusted_hidden_dim
        
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
        self.st_attention = SpatioTemporalAttention(config, hidden_dim, hidden_dim, num_heads=num_heads)

        # Graph convolutional layers
        self.graph_convs = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, config.DROPOUT)
            for _ in range(num_layers)
        ])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.DROPOUT if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2  # bidirectional

        # Output heads
        self.zero_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.count_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        self._initialize_weights()
        
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    
    def _initialize_weights(self):
        """Better weight initialization to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Small positive bias
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (batch_size, time_steps, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        batch_size, time_steps, num_nodes, input_dim = x.shape

        # Check input for NaN
        if torch.isnan(x).any():
            print(f"⚠️ NaN detected in INPUT")
            return {
                'predictions': torch.zeros(batch_size, num_nodes, device=x.device),
                'zero_probs': torch.ones(batch_size, num_nodes, device=x.device) * 0.5,
                'counts': torch.zeros(batch_size, num_nodes, device=x.device),
                'attention_weights': None
            }

        # 1. Input projection
        x_flat = x.view(-1, input_dim)
        x_proj = self.input_proj(x_flat)

        if torch.isnan(x_proj).any():
            print(f"⚠️ NaN after input_proj")
            return {
                'predictions': torch.zeros(batch_size, num_nodes, device=x.device),
                'zero_probs': torch.ones(batch_size, num_nodes, device=x.device) * 0.5,
                'counts': torch.zeros(batch_size, num_nodes, device=x.device),
                'attention_weights': None
            }

        x_proj = x_proj.view(batch_size, time_steps, num_nodes, self.hidden_dim)

        # 2. Spatio-temporal attention
        x_att = self.st_attention(x_proj)

        if torch.isnan(x_att).any():
            print(f"⚠️ NaN after attention")
            return {
                'predictions': torch.zeros(batch_size, num_nodes, device=x.device),
                'zero_probs': torch.ones(batch_size, num_nodes, device=x.device) * 0.5,
                'counts': torch.zeros(batch_size, num_nodes, device=x.device),
                'attention_weights': None
            }

        # 3. Graph convolutions at each time step
        graph_outputs = []
        for t in range(time_steps):
            x_t = x_att[:, t, :, :]  # (batch, nodes, hidden)

            h = x_t
            for graph_conv in self.graph_convs:
                h_new = graph_conv(h, adj_matrix)
                if torch.isnan(h_new).any():
                    print(f"⚠️ NaN in graph_conv at time {t}")
                    h_new = torch.zeros_like(h)

                h_new = F.relu(h_new)
                h = h + h_new  # Residual

            if torch.isnan(h).any():
                print(f"⚠️ NaN in graph output at time {t}")
                h = torch.zeros_like(h)
            graph_outputs.append(h)

        x_graph = torch.stack(graph_outputs, dim=1)  # (batch, time, nodes, hidden)

        # 4. LSTM
        x_lstm_in = x_graph.permute(0, 2, 1, 3).contiguous()
        x_lstm_in = x_lstm_in.view(batch_size * num_nodes, time_steps, self.hidden_dim)

        lstm_out, _ = self.lstm(x_lstm_in)
        if torch.isnan(lstm_out).any():
            print(f"⚠️ NaN after LSTM")
            lstm_out = torch.zeros_like(lstm_out)

        lstm_final = lstm_out[:, -1, :]  # Last time step
        lstm_final = lstm_final.view(batch_size, num_nodes, -1)

        # 5. Predictions
        zero_probs = self.zero_head(lstm_final).squeeze(-1)
        counts = self.count_head(lstm_final).squeeze(-1)
        if torch.isnan(zero_probs).any() or torch.isnan(counts).any():
            print(f"⚠️ NaN in output heads")
            zero_probs = torch.ones(batch_size, num_nodes, device=x.device) * 0.5
            counts = torch.zeros(batch_size, num_nodes, device=x.device)

        predictions = (1 - zero_probs) * counts

        if torch.isnan(predictions).any():
            print(f"⚠️ NaN in final predictions, replacing with zeros")
            predictions = torch.zeros(batch_size, num_nodes, device=x.device)

        return {
            'predictions': predictions,
            'zero_probs': zero_probs,
            'counts': counts,
            'attention_weights': self.st_attention.last_attention_weights
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
