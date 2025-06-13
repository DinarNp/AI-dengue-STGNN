import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict  # ← Tambahkan Dict di sini
plt.style.use('seaborn-v0_8')

class DengueVisualizer:
    """Visualization utilities for dengue prediction results"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_training_history(self, history: Dict):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(history['val_mae'], label='Validation MAE', color='green')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(history['val_rmse'], label='Validation RMSE', color='orange')
        axes[1, 0].set_title('Validation RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R²
        axes[1, 1].plot(history['val_r2'], label='Validation R²', color='purple')
        axes[1, 1].set_title('Validation R²')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, predictions: np.ndarray, actuals: np.ndarray,
                                  node_names: List[str] = None):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot
        axes[0, 0].scatter(actuals, predictions, alpha=0.6, color='blue')
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Cases')
        axes[0, 0].set_ylabel('Predicted Cases')
        axes[0, 0].set_title('Predictions vs Actual Cases')
        axes[0, 0].grid(True)
        
        # Residuals
        residuals = predictions - actuals
        axes[0, 1].scatter(actuals, residuals, alpha=0.6, color='red')
        axes[0, 1].axhline(y=0, color='black', linestyle='--')
        axes[0, 1].set_xlabel('Actual Cases')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True)
        
        # Distribution of predictions
        axes[1, 0].hist(predictions, bins=30, alpha=0.7, label='Predictions', color='blue')
        axes[1, 0].hist(actuals, bins=30, alpha=0.7, label='Actual', color='red')
        axes[1, 0].set_xlabel('Cases')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Cases')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Time series for first few nodes
        if node_names:
            n_nodes_to_plot = min(3, len(node_names))
            for i in range(n_nodes_to_plot):
                node_indices = np.arange(i, len(predictions), len(node_names))
                if len(node_indices) > 0:
                    axes[1, 1].plot(node_indices, actuals[node_indices], 
                                   label=f'{node_names[i]} (Actual)', linestyle='-')
                    axes[1, 1].plot(node_indices, predictions[node_indices], 
                                   label=f'{node_names[i]} (Pred)', linestyle='--')
        
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Cases')
        axes[1, 1].set_title('Time Series Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_spatial_heatmap(self, predictions: np.ndarray, location_coords: np.ndarray,
                           node_names: List[str] = None):
        """Plot spatial heatmap of predictions"""
        plt.figure(figsize=(12, 8))
        
        # Average predictions per node
        n_nodes = len(location_coords)
        avg_predictions = np.zeros(n_nodes)
        
        for i in range(n_nodes):
            node_indices = np.arange(i, len(predictions), n_nodes)
            if len(node_indices) > 0:
                avg_predictions[i] = np.mean(predictions[node_indices])
        
        # Create scatter plot
        scatter = plt.scatter(location_coords[:, 1], location_coords[:, 0], 
                            c=avg_predictions, s=200, cmap='YlOrRd', 
                            alpha=0.8, edgecolors='black')
        
        plt.colorbar(scatter, label='Average Predicted Cases')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Distribution of Predicted Dengue Cases')
        
        # Add node labels if provided
        if node_names:
            for i, name in enumerate(node_names):
                plt.annotate(name, (location_coords[i, 1], location_coords[i, 0]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.show()