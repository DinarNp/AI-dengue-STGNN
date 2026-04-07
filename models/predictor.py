import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import traceback

class DenguePredictor:
    """Interface for making predictions with trained STGNNDenguePredictor model"""

    def __init__(self, model_path: str):
        """Initialize predictor with trained model"""
        
        print(f"\n🔧 Initializing DenguePredictor from {model_path}...")
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"   Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print(f"   Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print(f"   Using CPU")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            print(f"   ✅ Checkpoint loaded")
        except Exception as e:
            print(f"   ⚠️ Error loading checkpoint: {e}")
            traceback.print_exc()
            raise
        
        # Extract components
        self.config = checkpoint.get('config')
        self.metadata = checkpoint.get('metadata', {})
        
        if self.config is None:
            # Create default config if missing
            from config.config import Config
            self.config = Config()
            print(f"   ⚠️ No config in checkpoint, using default")
        
        # ✅ CRITICAL FIX: Apply adaptive config that was used during training
        adaptive_config = checkpoint.get('adaptive_config_used') or self.metadata.get('adaptive_config', {})
        if adaptive_config:
            print(f"   📋 Applying adaptive config from training:")
            for key, value in adaptive_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    print(f"      {key}: {value}")
        
        # ✅ CRITICAL: Get the number of nodes the model expects
        self.n_nodes = self.metadata.get('n_nodes', 5)
        self.node_ids = self.metadata.get('node_ids', [f'Node_{i}' for i in range(self.n_nodes)])
        
        print(f"   Model expects: {self.n_nodes} nodes")
        print(f"   Node IDs: {self.node_ids}")

        # Initialize model with STGNNDenguePredictor
        try:
            from models.stgnn import STGNNDenguePredictor
            
            input_dim = len(self.metadata.get('feature_cols', []))
            if input_dim == 0:
                input_dim = self.metadata.get('input_dim', 26)
            
            n_nodes = self.metadata.get('n_nodes', 5)
            
            print(f"   Initializing STGNNDenguePredictor:")
            print(f"      Input dim: {input_dim}")
            print(f"      Nodes: {n_nodes}")
            print(f"      Hidden dim: {self.config.HIDDEN_DIM}")
            
            # Initialize model - DON'T pass n_nodes, it's not a parameter!
            # The model will use config.HIDDEN_DIM which we already set from adaptive_config
            self.model = STGNNDenguePredictor(self.config, input_dim)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"   ✅ Model initialized and weights loaded")
            
        except Exception as e:
            print(f"   ❌ Error initializing model: {e}")
            traceback.print_exc()
            raise
        
        # Create adjacency matrix
        try:
            from models.graph_constructor import GraphConstructor
            
            location_coords = self.metadata.get('location_coords')
            
            if location_coords is not None:
                graph_constructor = GraphConstructor(self.config)
                spatial_adj = graph_constructor.build_spatial_adjacency(location_coords)
            else:
                print(f"   ⚠️ No location coords, using identity matrix")
                spatial_adj = np.eye(n_nodes)
            
            self.adj_matrix = torch.FloatTensor(spatial_adj).to(self.device)
            print(f"   ✅ Adjacency matrix: {self.adj_matrix.shape}")
            
        except Exception as e:
            print(f"   ⚠️ Error creating adjacency: {e}")
            self.adj_matrix = torch.eye(n_nodes).to(self.device)
        
        # Store test metrics if available
        self.test_metrics = checkpoint.get('test_metrics', {})
        if self.test_metrics:
            mae = self.test_metrics.get('mae', 'N/A')
            if isinstance(mae, (int, float)):
                print(f"   ✅ Test metrics: MAE={mae:.4f}")
            else:
                print(f"   ✅ Test metrics available")

        print(f"   ✅ Predictor ready!")
    
    def predict(self, input_features: np.ndarray, node_index: int = 0) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            input_features: Can be various shapes, will be reshaped automatically
        
        Returns:
            Dict with predictions, case_counts, zero_probabilities, node_ids
        """
        
        try:
            print(f"\n🎯 Making prediction...")
            print(f"   Input shape: {input_features.shape}")
            
            # ✅ CRITICAL FIX: Prepare input with ALL nodes the model expects
            input_tensor = self._prepare_input_all_nodes(input_features)
            print(f"   Prepared shape: {input_tensor.shape}")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.adj_matrix)
            
            # Extract outputs
            predictions = outputs['predictions'].cpu().numpy()
            zero_probs = outputs['zero_probs'].cpu().numpy()
            counts = outputs.get('counts', predictions).cpu().numpy()
            
            # Apply inverse transform if needed
            if self.metadata.get('target_transform') == 'log1p':
                print(f"   Applying inverse log1p transform...")
                predictions = np.expm1(np.maximum(predictions, 0))
                counts = np.expm1(np.maximum(counts, 0))
            
            # Ensure non-negative
            predictions = np.maximum(predictions, 0)
            counts = np.maximum(counts, 0)
            
            # ✅ Extract prediction for the requested node
            node_prediction = predictions[0, node_index]
            node_zero_prob = zero_probs[0, node_index]
            node_count = counts[0, node_index]
            
            print(f"   ✅ Node {node_index} prediction: {node_prediction:.2f}")
        
            return {
                'predictions': np.array([[node_prediction]]),  # Keep 2D for compatibility
                'case_counts': np.array([[node_count]]),
                'zero_probabilities': np.array([[node_zero_prob]]),
                'node_ids': [self.node_ids[node_index]],
                'all_predictions': predictions[0],  # All nodes for reference
                'all_node_ids': self.node_ids
            }
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            traceback.print_exc()
            raise
    
    def _prepare_input_all_nodes(self, input_features: np.ndarray) -> torch.Tensor:
        """
        Prepare input with ALL nodes the model expects
        
        ✅ KEY FIX: Model was trained with n_nodes (e.g., 5) always present
        So we need to provide data for all nodes, even if predicting for just one
        
        Args:
            input_features: Features for ONE location (1D array of features)
        
        Returns:
            Tensor of shape (1, window_size, n_nodes, num_features)
        """
        
        window_size = getattr(self.config, 'WINDOW_SIZE', 8)
        
        # Get expected feature dimension
        expected_features = len(self.metadata.get('feature_cols', []))
        if expected_features == 0:
            expected_features = self.metadata.get('input_dim', 26)
        
        # Ensure input is 1D array of features
        if len(input_features.shape) > 1:
            input_features = input_features.flatten()
        
        if len(input_features) != expected_features:
            print(f"   ⚠️ Feature count mismatch: got {len(input_features)}, expected {expected_features}")
            # Pad or truncate
            if len(input_features) < expected_features:
                input_features = np.pad(input_features, (0, expected_features - len(input_features)))
            else:
                input_features = input_features[:expected_features]
        
        # ✅ Create input for ALL nodes
        # Strategy: Repeat the same features for all nodes
        # (In production, you'd have actual data for each node)
        
        # Shape: (window_size, n_nodes, num_features)
        # Repeat features across time and nodes
        node_features = np.tile(input_features, (window_size, self.n_nodes, 1))
        
        # Add batch dimension: (1, window_size, n_nodes, num_features)
        batch_input = np.expand_dims(node_features, axis=0)
        
        print(f"   Created input for all {self.n_nodes} nodes")
        print(f"   Shape: (batch={batch_input.shape[0]}, time={batch_input.shape[1]}, nodes={batch_input.shape[2]}, features={batch_input.shape[3]})")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(batch_input).to(self.device)
        
        return input_tensor
    
    def predict_all_nodes(self, input_features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions for ALL nodes at once
        
        Args:
            input_features: Features for ONE location (will be used for all nodes)
        
        Returns:
            Dict with predictions for all nodes
        """
        
        try:
            print(f"\n🎯 Making predictions for ALL nodes...")
            
            # Prepare input
            input_tensor = self._prepare_input_all_nodes(input_features)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.adj_matrix)
            
            # Extract outputs
            predictions = outputs['predictions'].cpu().numpy()[0]  # Remove batch dim
            zero_probs = outputs['zero_probs'].cpu().numpy()[0]
            counts = outputs.get('counts', outputs['predictions']).cpu().numpy()[0]
            
            # Apply inverse transform if needed
            if self.metadata.get('target_transform') == 'log1p':
                predictions = np.expm1(np.maximum(predictions, 0))
                counts = np.expm1(np.maximum(counts, 0))
            
            predictions = np.maximum(predictions, 0)
            counts = np.maximum(counts, 0)
            
            print(f"   ✅ All predictions: {predictions}")
            
            return {
                'predictions': predictions,
                'case_counts': counts,
                'zero_probabilities': zero_probs,
                'node_ids': self.node_ids
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
            raise
    
    # models/predictor.py - FIXED VERSION with location-specific input

    def predict_for_location(self, input_features: np.ndarray, location_name: str) -> Dict[str, np.ndarray]:
        """
        Make prediction for a specific location with proper node matching
        
        Args:
            input_features: Features for the specific location
            location_name: Name of the location (regency)
        
        Returns:
            Dict with prediction for that location
        """
        
        try:
            print(f"\n🎯 Predicting for location: {location_name}")
            print(f"   Input features shape: {input_features.shape}")
            
            # ✅ Find which node index corresponds to this location
            node_index = self._find_node_index(location_name)
            print(f"   Matched to node {node_index}: {self.node_ids[node_index]}")
            
            # ✅ Create input with location-specific features
            input_tensor = self._prepare_input_with_location(input_features, node_index)
            print(f"   Prepared tensor shape: {input_tensor.shape}")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor, self.adj_matrix)
            
            # Extract outputs
            predictions = outputs['predictions'].cpu().numpy()
            zero_probs = outputs['zero_probs'].cpu().numpy()
            counts = outputs.get('counts', predictions).cpu().numpy()
            
            print(f"   Raw predictions (all nodes): {predictions[0]}")
            
            # Apply inverse transform if needed
            if self.metadata.get('target_transform') == 'log1p':
                print(f"   Applying inverse log1p transform...")
                predictions = np.expm1(np.maximum(predictions, 0))
                counts = np.expm1(np.maximum(counts, 0))
            
            predictions = np.maximum(predictions, 0)
            counts = np.maximum(counts, 0)
            
            # Extract prediction for the requested location
            node_prediction = predictions[0, node_index]
            node_zero_prob = zero_probs[0, node_index]
            node_count = counts[0, node_index]
            
            print(f"   ✅ Prediction for {location_name}: {node_prediction:.2f} cases")
            
            return {
                'predictions': np.array([[node_prediction]]),
                'case_counts': np.array([[node_count]]),
                'zero_probabilities': np.array([[node_zero_prob]]),
                'node_ids': [self.node_ids[node_index]],
                'all_predictions': predictions[0],
                'all_node_ids': self.node_ids
            }
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            traceback.print_exc()
            raise

    def _find_node_index(self, location_name: str) -> int:
        """Find node index for a given location name"""
        
        # Clean location name
        location_clean = location_name.strip().upper()
        
        # Try exact match first
        for i, node_id in enumerate(self.node_ids):
            if node_id.upper() == location_clean:
                return i
        
        # Try partial match
        for i, node_id in enumerate(self.node_ids):
            if location_clean in node_id.upper() or node_id.upper() in location_clean:
                return i
        
        # Default to first node if no match
        print(f"   ⚠️ No match found for '{location_name}', using node 0")
        return 0

    def _prepare_input_with_location(self, input_features: np.ndarray, target_node_index: int) -> torch.Tensor:
        """
        Prepare input with location-specific features
        
        Strategy: Put the actual features at target_node_index, 
        use zeros or mean values for other nodes
        """
        
        window_size = getattr(self.config, 'WINDOW_SIZE', 8)
        expected_features = len(self.metadata.get('feature_cols', []))
        if expected_features == 0:
            expected_features = self.metadata.get('input_dim', 26)
        
        # Ensure input is 1D
        if len(input_features.shape) > 1:
            input_features = input_features.flatten()
        
        # Pad or truncate if needed
        if len(input_features) != expected_features:
            if len(input_features) < expected_features:
                input_features = np.pad(input_features, (0, expected_features - len(input_features)))
            else:
                input_features = input_features[:expected_features]
        
        # ✅ Create input array: (window_size, n_nodes, num_features)
        batch_input = np.zeros((window_size, self.n_nodes, expected_features), dtype=np.float32)
        
        # Put actual features at the target node index
        for t in range(window_size):
            batch_input[t, target_node_index, :] = input_features
        
        # For other nodes, use small random values or zeros
        # This helps the model distinguish between nodes
        for t in range(window_size):
            for n in range(self.n_nodes):
                if n != target_node_index:
                    # Use zeros or very small values for other nodes
                    batch_input[t, n, :] = input_features * 0.1  # 10% of actual values
        
        # Add batch dimension: (1, window_size, n_nodes, num_features)
        batch_input = np.expand_dims(batch_input, axis=0)
        
        print(f"   Created input: node {target_node_index} has actual features")
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(batch_input).to(self.device)
        
        return input_tensor

    def predict_with_all_locations(self, all_location_features: np.ndarray, 
                               target_location_name: str) -> Dict[str, np.ndarray]:
        """
        Predict with REAL features for all locations
        
        Args:
            all_location_features: (n_nodes, n_features) array with features for each location
            target_location_name: Name of location to extract prediction for
        
        Returns:
            Dict with prediction for target location
        """
        
        try:
            print(f"\n🎯 Predicting with all location features for: {target_location_name}")
            
            # Find target node index
            node_index = self._find_node_index(target_location_name)
            print(f"   Target node: {node_index} ({self.node_ids[node_index]})")
            
            # Prepare input: (1, window_size, n_nodes, n_features)
            window_size = getattr(self.config, 'WINDOW_SIZE', 8)
            
            # Tile across time: (window_size, n_nodes, n_features)
            input_sequence = np.tile(all_location_features, (window_size, 1, 1))
            
            # Add batch dimension: (1, window_size, n_nodes, n_features)
            input_batch = np.expand_dims(input_sequence, axis=0)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_batch).to(self.device)
            
            print(f"   Input shape: {input_tensor.shape}")
            
            # Predict with ACTUAL adjacency matrix (nodes can interact properly)
            with torch.no_grad():
                outputs = self.model(input_tensor, self.adj_matrix)
            
            # Extract outputs
            predictions = outputs['predictions'].cpu().numpy()
            zero_probs = outputs['zero_probs'].cpu().numpy()
            counts = outputs.get('counts', predictions).cpu().numpy()
            
            print(f"   Raw predictions (all nodes): {predictions[0]}")
            
            # Apply inverse transform
            if self.metadata.get('target_transform') == 'log1p':
                predictions = np.expm1(np.maximum(predictions, 0))
                counts = np.expm1(np.maximum(counts, 0))
            
            predictions = np.maximum(predictions, 0)
            counts = np.maximum(counts, 0)
            
            print(f"   Final predictions (all nodes): {predictions[0]}")
            
            # Extract for target location
            node_prediction = predictions[0, node_index]
            node_zero_prob = zero_probs[0, node_index]
            node_count = counts[0, node_index]
            
            print(f"   ✅ Prediction for {target_location_name}: {node_prediction:.2f} cases")
            
            return {
                'predictions': np.array([[node_prediction]]),
                'case_counts': np.array([[node_count]]),
                'zero_probabilities': np.array([[node_zero_prob]]),
                'node_ids': [self.node_ids[node_index]],
                'all_predictions': predictions[0],
                'all_node_ids': self.node_ids
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
            raise

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'device': str(self.device),
            'n_nodes': self.n_nodes,
            'node_ids': self.node_ids,
            'input_dim': len(self.metadata.get('feature_cols', [])),
            'feature_cols': self.metadata.get('feature_cols', []),
            'window_size': getattr(self.config, 'WINDOW_SIZE', 8),
            'test_metrics': self.test_metrics,
            'target_transform': self.metadata.get('target_transform')
        }