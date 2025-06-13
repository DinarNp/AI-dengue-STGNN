import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .stgnn import STGNNDenguePredictor
from .graph_constructor import GraphConstructor

class DenguePredictor:
    """Interface for making predictions with trained model"""

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model with weights_only=False for older checkpoints
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Loading with weights_only=False due to: {e}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.metadata = checkpoint['metadata']
    
        
        # Initialize model
        input_dim = len(self.metadata['feature_cols'])
        self.model = STGNNDenguePredictor(self.config, input_dim, self.metadata['n_nodes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create adjacency matrix
        graph_constructor = GraphConstructor(self.config)
        spatial_adj = graph_constructor.build_spatial_adjacency(self.metadata['location_coords'])
        self.adj_matrix = torch.FloatTensor(spatial_adj).to(self.device)
        
        print("Model loaded successfully!")
        print(f"Test metrics from training: {checkpoint['test_metrics']}")
    
    def predict(self, input_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions on new data"""
        
        # Prepare input tensor
        if len(input_features.shape) == 3:
            # Add batch dimension
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
        else:
            input_tensor = torch.FloatTensor(input_features)
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, self.adj_matrix)
            
            predictions = outputs['predictions'].cpu().numpy()
            case_counts = outputs['case_counts'].cpu().numpy()
            zero_probs = outputs['zero_probs'].cpu().numpy()
        
        return {
            'predictions': predictions,
            'case_counts': case_counts,
            'zero_probabilities': zero_probs,
            'node_ids': self.metadata['node_ids']
        }