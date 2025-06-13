import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .attention import SpatioTemporalAttention
from .graph_layers import GraphConvLayer

class STGNNDenguePredictor(nn.Module):
    """Spatio-Temporal GNN for Dengue Case Prediction"""
    
    def __init__(self, config, input_dim: int, num_nodes: int):
        super(STGNNDenguePredictor, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.HIDDEN_SIZE)
        
        # Spatio-temporal attention
        self.st_attention = SpatioTemporalAttention(
            config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.ATTENTION_HEADS
        )
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.DROPOUT)
            for _ in range(config.GNN_LAYERS)
        ])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, 1)
        )
        
        # Zero-inflation handling
        self.zero_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.DROPOUT)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the STGNN model"""
        batch_size, time_steps, num_nodes, input_dim = x.shape
        
        # Input projection
        h = self.input_projection(x)
        
        # Spatio-temporal attention
        h = self.st_attention(h, adj_matrix)
        
        # Graph convolutional layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj_matrix)
            h = self.dropout(h)
        
        # LSTM for temporal dependencies
        # Reshape for LSTM: [batch_size * num_nodes, time_steps, hidden_size]
        h_lstm = h.view(batch_size * num_nodes, time_steps, self.config.HIDDEN_SIZE)
        lstm_out, _ = self.lstm(h_lstm)
        
        # Take the last time step output
        final_hidden = lstm_out[:, -1, :]  # [batch_size * num_nodes, hidden_size]
        final_hidden = final_hidden.view(batch_size, num_nodes, self.config.HIDDEN_SIZE)
        
        # Predictions
        case_counts = self.output_projection(final_hidden).squeeze(-1)  # [batch_size, num_nodes]
        zero_probs = self.zero_classifier(final_hidden).squeeze(-1)     # [batch_size, num_nodes]
        
        # Apply zero-inflation
        final_predictions = case_counts * (1 - zero_probs)
        
        return {
            'predictions': final_predictions,
            'case_counts': case_counts,
            'zero_probs': zero_probs
        }