import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .attention import SpatioTemporalAttention
from .graph_layers import GraphConvLayer

class STGNNDenguePredictor(nn.Module):
    """Enhanced Spatio-Temporal GNN for Dengue Case Prediction with improved architecture"""
    
    def __init__(self, config, input_dim: int, num_nodes: int):
        super(STGNNDenguePredictor, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        
        # Enhanced input projection with batch normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_SIZE),
            nn.BatchNorm1d(config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # Enhanced spatio-temporal attention
        self.st_attention = SpatioTemporalAttention(
            config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.ATTENTION_HEADS
        )
        
        # Enhanced graph convolutional layers with residual connections
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.DROPOUT)
            for _ in range(config.GNN_LAYERS)
        ])
        
        # Enhanced LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.HIDDEN_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM
        )
        
        # Enhanced output layers with skip connections
        self.output_projection = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE),  # *2 for bidirectional
            nn.BatchNorm1d(config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, 1)
        )
        
        # Enhanced zero-inflation handling
        self.zero_classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE),  # *2 for bidirectional
            nn.BatchNorm1d(config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE // 2, 1),
            nn.Sigmoid()
        )
        
        # Additional regularization layers
        self.dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm = nn.LayerNorm(config.HIDDEN_SIZE)
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass of the STGNN model - FIXED VERSION"""
        try:
            # Debug print
            print(f"🔍 Model forward input shape: {x.shape}, adj shape: {adj_matrix.shape}")
            
            # Expected: x shape = (batch_size, time_steps, num_nodes, input_dim)
            if len(x.shape) != 4:
                raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
            
            batch_size, time_steps, num_nodes, input_dim = x.shape
            
            # Verify dimensions
            expected_time_steps = self.config.WINDOW_SIZE
            if time_steps != expected_time_steps:
                print(f"⚠️ WARNING: Expected {expected_time_steps} time steps, got {time_steps}")
                
            # Enhanced input projection
            x_reshaped = x.view(-1, input_dim)
            h = self.input_projection(x_reshaped)
            h = h.view(batch_size, time_steps, num_nodes, self.config.HIDDEN_SIZE)
            
            # Store initial features for residual connection
            h_initial = h.clone()
            
            # Enhanced spatio-temporal attention
            h = self.st_attention(h, adj_matrix)
            h = self.layer_norm(h + h_initial)  # Residual connection
            
            # Enhanced graph convolutional layers with residual connections
            for i, gnn_layer in enumerate(self.gnn_layers):
                h_residual = h.clone()
                h = gnn_layer(h, adj_matrix)
                if i > 0:  # Add residual connection for deeper layers
                    h = self.layer_norm(h + h_residual)
                h = self.dropout(h)
            
            # Enhanced LSTM for temporal dependencies
            # Reshape for LSTM: [batch_size * num_nodes, time_steps, hidden_size]
            h_lstm = h.view(batch_size * num_nodes, time_steps, self.config.HIDDEN_SIZE)
            lstm_out, _ = self.lstm(h_lstm)
            
            # Take the last time step output from bidirectional LSTM
            final_hidden = lstm_out[:, -1, :]  # [batch_size * num_nodes, hidden_size * 2]
            final_hidden = final_hidden.view(batch_size, num_nodes, self.config.HIDDEN_SIZE * 2)
            
            # Enhanced predictions with better regularization
            case_counts = self.output_projection(final_hidden.view(-1, self.config.HIDDEN_SIZE * 2))
            case_counts = case_counts.view(batch_size, num_nodes, 1).squeeze(-1)
            
            zero_probs = self.zero_classifier(final_hidden.view(-1, self.config.HIDDEN_SIZE * 2))
            zero_probs = zero_probs.view(batch_size, num_nodes, 1).squeeze(-1)
            
            # Enhanced zero-inflation with better scaling
            final_predictions = case_counts * (1 - zero_probs)
            
            # Apply additional regularization
            final_predictions = torch.clamp(final_predictions, min=0.0)  # Ensure non-negative
            
            return {
                'predictions': final_predictions,
                'case_counts': case_counts,
                'zero_probs': zero_probs
            }
            
        except Exception as e:
            print(f"❌ Error in model forward pass: {e}")
            print(f"   Input shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
            print(f"   Adj matrix shape: {adj_matrix.shape if hasattr(adj_matrix, 'shape') else 'unknown'}")
            import traceback
            traceback.print_exc()
            raise e