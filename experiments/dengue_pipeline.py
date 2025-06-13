import torch
import numpy as np
from typing import Dict, Tuple
from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor
from models.stgnn import STGNNDenguePredictor
from training.trainer import DengueTrainer
from utils.visualization import DengueVisualizer

class DenguePredictionSystem:
    """Main system for dengue prediction using STGNN"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DengueDataPreprocessor(config)
        self.graph_constructor = GraphConstructor(config)
        self.trainer = DengueTrainer(config)
        self.visualizer = DengueVisualizer()
        
    def run_complete_pipeline(self, data_path: str = None):
        """Run the complete dengue prediction pipeline"""
        
        print("=" * 80)
        print("DENGUE PREDICTION USING SPATIO-TEMPORAL GNN")
        print("=" * 80)
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        if data_path:
            df = self.preprocessor.load_data(data_path)
        else:
            df = self.preprocessor.load_data("dummy_path")  # Will generate synthetic data
        
        features, targets, metadata = self.preprocessor.preprocess_data(df)
        
        # Step 2: Construct graph
        print("\n2. Constructing spatial graph...")
        location_coords = metadata['location_coords']
        spatial_adj = self.graph_constructor.build_spatial_adjacency(location_coords)
        
        # Convert to tensor
        adj_matrix = torch.FloatTensor(spatial_adj).to(self.trainer.device)
        
        print(f"Graph constructed with {metadata['n_nodes']} nodes")
        print(f"Adjacency matrix density: {np.mean(spatial_adj > 0):.3f}")
        
        # Step 3: Create data loaders
        print("\n3. Creating data loaders...")
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(
            features, targets, metadata)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Step 4: Initialize model
        print("\n4. Initializing STGNN model...")
        input_dim = len(metadata['feature_cols'])
        model = STGNNDenguePredictor(self.config, input_dim, metadata['n_nodes'])
        model = model.to(self.trainer.device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        
        # Step 5: Train model
        print("\n5. Training model...")
        trained_model, history = self.trainer.train(model, train_loader, val_loader, adj_matrix)
        
        # Step 6: Evaluate on test set
        print("\n6. Evaluating on test set...")
        test_metrics = self.trainer.evaluate(trained_model, test_loader, adj_matrix)
        
        print("\nFinal Test Results:")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")  # Remove ² symbol
        print(f"Zero Accuracy: {test_metrics['zero_accuracy']:.4f}")
        print(f"Non-zero MAE: {test_metrics['non_zero_mae']:.4f}")
        
        # Step 7: Generate predictions for visualization
        print("\n7. Generating visualizations...")
        trained_model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.trainer.device)
                outputs = trained_model(batch_features, adj_matrix)
                predictions = outputs['predictions'].cpu().numpy()
                targets = batch_targets.cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Step 8: Visualizations
        self.visualizer.plot_training_history(history)
        self.visualizer.plot_predictions_vs_actual(all_predictions, all_targets, 
                                                  metadata['node_ids'])
        self.visualizer.plot_spatial_heatmap(all_predictions, location_coords, 
                                           metadata['node_ids'])
        
        # Step 9: Save model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': self.config,
            'metadata': metadata,
            'test_metrics': test_metrics
        }, 'dengue_stgnn_model.pth')
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("Model saved as 'dengue_stgnn_model.pth'")
        print("=" * 80)
        
        return trained_model, test_metrics, metadata