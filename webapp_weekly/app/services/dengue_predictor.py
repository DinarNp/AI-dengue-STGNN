"""
Live inference wrapper around the canonical revision_experiments STGNN.

Unlike the old webapp's models/predictor.py, this always has real
multi-region weekly feature data available (the prediction service builds a
(window_size, n_nodes, n_features) tensor directly from the weekly DB), so
only the real-data code path is kept -- no single-location repeat/zero-fill
fallbacks.

Imports models.stgnn / models.graph_constructor / config.config from
revision_experiments (inserted at sys.path[0]) rather than duplicating that
code here, so this stays in sync with whatever is canonical there.
"""
import os
import sys
import traceback
from typing import Dict

import numpy as np
import torch

REVISION_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    'revision_experiments',
)
if REVISION_EXPERIMENTS not in sys.path:
    sys.path.insert(0, REVISION_EXPERIMENTS)


class DenguePredictor:
    """Interface for making predictions with the trained STGNNDenguePredictor model."""

    def __init__(self, model_path: str):
        print(f"\nInitializing DenguePredictor from {model_path}...")

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"   Using device: {self.device}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"   Error loading checkpoint: {e}")
            traceback.print_exc()
            raise

        self.config = checkpoint.get('config')
        self.metadata = checkpoint.get('metadata', {})

        if self.config is None:
            from config.config import Config
            self.config = Config()
            print("   No config in checkpoint, using default")

        # Apply the adaptive config actually used during training (e.g. the
        # winning hidden_dim=64/num_layers=2 capacity, not the class defaults).
        adaptive_config = checkpoint.get('adaptive_config_used') or self.metadata.get('adaptive_config', {})
        for key, value in adaptive_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.n_nodes = self.metadata.get('n_nodes', 5)
        self.node_ids = self.metadata.get('node_ids', [f'Node_{i}' for i in range(self.n_nodes)])
        self.window_size = getattr(self.config, 'WINDOW_SIZE', 8)
        self.forecast_horizon = getattr(self.config, 'FORECAST_HORIZON', 4)
        print(f"   Nodes ({self.n_nodes}): {self.node_ids}")
        print(f"   Window size: {self.window_size} weeks, horizon: {self.forecast_horizon} weeks")

        from models.stgnn import STGNNDenguePredictor

        input_dim = len(self.metadata.get('feature_cols', [])) or self.metadata.get('input_dim', 26)
        hidden_dim = getattr(self.config, 'HIDDEN_DIM', 64)
        print(f"   input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={getattr(self.config, 'NUM_LAYERS', 2)}")

        self.model = STGNNDenguePredictor(self.config, input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        from models.graph_constructor import GraphConstructor

        location_coords = self.metadata.get('location_coords')
        if location_coords is not None:
            graph_constructor = GraphConstructor(self.config)
            k_neighbors = getattr(self.config, 'K_NEIGHBORS_OVERRIDE', None) or 3
            spatial_adj = graph_constructor.build_spatial_adjacency(location_coords, k_neighbors=k_neighbors)
        else:
            print("   No location coords in checkpoint metadata, using identity adjacency")
            spatial_adj = np.eye(self.n_nodes)
        self.adj_matrix = torch.FloatTensor(spatial_adj).to(self.device)

        self.test_metrics = checkpoint.get('test_metrics', {})
        print("   Predictor ready.")

    def _find_node_index(self, location_name: str) -> int:
        location_clean = location_name.strip().upper()
        for i, node_id in enumerate(self.node_ids):
            if node_id.upper() == location_clean:
                return i
        for i, node_id in enumerate(self.node_ids):
            if location_clean in node_id.upper() or node_id.upper() in location_clean:
                return i
        raise ValueError(f"No matching node for location '{location_name}' among {self.node_ids}")

    def predict_with_all_locations(self, window_features: np.ndarray, target_location_name: str) -> Dict:
        """
        Args:
            window_features: (window_size, n_nodes, n_features) real weekly feature
                sequence for every region, aligned to self.node_ids order.
            target_location_name: which region's prediction to extract.

        Returns a dict with the target region's predicted cases (already
        inverse-transformed out of log1p space if the model was trained that
        way) plus the raw all-region prediction vector for reference.
        """
        node_index = self._find_node_index(target_location_name)

        expected_shape = (self.window_size, self.n_nodes)
        if window_features.shape[:2] != expected_shape:
            raise ValueError(
                f"window_features shape {window_features.shape[:2]} does not match "
                f"expected (window_size={self.window_size}, n_nodes={self.n_nodes})"
            )

        batch = np.ascontiguousarray(np.expand_dims(window_features, axis=0))
        input_tensor = torch.FloatTensor(batch).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, self.adj_matrix)

        predictions = outputs['predictions'].cpu().numpy()
        zero_probs = outputs['zero_probs'].cpu().numpy()
        counts = outputs.get('counts', outputs['predictions']).cpu().numpy()

        if self.metadata.get('target_transform') == 'log1p':
            predictions = np.expm1(np.maximum(predictions, 0))
            counts = np.expm1(np.maximum(counts, 0))

        predictions = np.maximum(predictions, 0)
        counts = np.maximum(counts, 0)

        return {
            'predicted_cases': float(predictions[0, node_index]),
            'zero_probability': float(zero_probs[0, node_index]),
            'count_estimate': float(counts[0, node_index]),
            'node_id': self.node_ids[node_index],
            'all_predictions': predictions[0].tolist(),
            'all_node_ids': self.node_ids,
        }

    def get_model_info(self) -> Dict:
        return {
            'device': str(self.device),
            'n_nodes': self.n_nodes,
            'node_ids': self.node_ids,
            'input_dim': len(self.metadata.get('feature_cols', [])),
            'feature_cols': self.metadata.get('feature_cols', []),
            'window_size': self.window_size,
            'forecast_horizon': self.forecast_horizon,
            'test_metrics': self.test_metrics,
            'target_transform': self.metadata.get('target_transform'),
        }
