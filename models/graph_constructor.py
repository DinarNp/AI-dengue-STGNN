# =============================================================================
# FIXED MODELS FILES
# =============================================================================

# ===== 1. models/graph_constructor.py =====
import numpy as np
from typing import Dict

class GraphConstructor:
    """Constructs spatial and spatio-temporal graphs"""
    
    def __init__(self, config):
        self.config = config
    
    def haversine_distance(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate Haversine distance between coordinates"""
        lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
        lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
        
        dlat = lat1[:, None] - lat2[None, :]
        dlon = lon1[:, None] - lon2[None, :]
        
        a = np.sin(dlat/2)**2 + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth's radius in km
        
        return r * c
    
    def build_spatial_adjacency(self, location_coords: np.ndarray) -> np.ndarray:
        """Build spatial adjacency matrix based on geographical distance"""
        n_nodes = len(location_coords)
        distances = self.haversine_distance(location_coords, location_coords)
        
        # k-NN approach: connect each node to k nearest neighbors
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            # Get k nearest neighbors (excluding self)
            distances_i = distances[i].copy()
            distances_i[i] = np.inf  # Exclude self
            nearest_indices = np.argsort(distances_i)[:self.config.K_NEAREST]
            
            for j in nearest_indices:
                # Inverse distance weighting
                if distances[i, j] > 0:
                    weight = 1.0 / (1.0 + distances[i, j])
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight  # Symmetric
        
        return adj_matrix
    
    def build_environmental_similarity(self, features: np.ndarray, node_mapping: Dict) -> np.ndarray:
        """Build environmental similarity matrix"""
        n_nodes = len(node_mapping)
        env_adj = np.zeros((n_nodes, n_nodes))
        
        # Extract environmental features (NDVI, temperature, humidity, etc.)
        env_feature_indices = [2, 3, 4, 5, 6, 7]  # Adjust based on feature order
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Get environmental features for both nodes
                node_i_features = features[features[:, -1] == i]  # Assuming last column is node ID
                node_j_features = features[features[:, -1] == j]
                
                if len(node_i_features) > 0 and len(node_j_features) > 0:
                    # Calculate cosine similarity
                    env_i = np.mean(node_i_features[:, env_feature_indices], axis=0)
                    env_j = np.mean(node_j_features[:, env_feature_indices], axis=0)
                    
                    similarity = np.dot(env_i, env_j) / (np.linalg.norm(env_i) * np.linalg.norm(env_j) + 1e-8)
                    
                    if similarity > self.config.ENV_SIMILARITY_THRESHOLD:
                        env_adj[i, j] = similarity
                        env_adj[j, i] = similarity
        
        return env_adj
    
    def combine_adjacency_matrices(self, spatial_adj: np.ndarray, env_adj: np.ndarray, 
                                  alpha: float = 0.7, beta: float = 0.3) -> np.ndarray:
        """Combine spatial and environmental adjacency matrices"""
        combined_adj = alpha * spatial_adj + beta * env_adj
        return combined_adj