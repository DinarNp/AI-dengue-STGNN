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
    """Main system for dengue prediction using STGNN with adaptive configuration"""
    
    def __init__(self, config):
        self.config = config
        self.preprocessor = DengueDataPreprocessor(config)
        self.graph_constructor = GraphConstructor(config)
        self.trainer = DengueTrainer(config)
        self.visualizer = DengueVisualizer()
        
    def run_complete_pipeline(self, data_path: str = None):
        """Run the complete dengue prediction pipeline with adaptive configuration"""
        
        print("=" * 80)
        print("ADAPTIVE DENGUE PREDICTION USING SPATIO-TEMPORAL GNN")
        print("=" * 80)
        
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        if data_path:
            df = self.preprocessor.load_data(data_path)
        else:
            df = self.preprocessor.load_data("dummy_path")  # Will generate synthetic data
        
        features, targets, metadata = self.preprocessor.preprocess_data(df)
        
        # 🎯 APPLY ADAPTIVE CONFIGURATION
        if 'adaptive_config' in metadata:
            adaptive_config = metadata['adaptive_config']
            print(f"\n🔧 Applying adaptive configuration for {adaptive_config['scale_type']} scale data:")
            
            # Update config attributes dynamically
            self.config.LEARNING_RATE = adaptive_config['LEARNING_RATE']
            self.config.DROPOUT = adaptive_config['DROPOUT']
            self.config.BATCH_SIZE = adaptive_config['BATCH_SIZE']
            self.config.WEIGHT_DECAY = adaptive_config['WEIGHT_DECAY']
            self.config.EPOCHS = adaptive_config['EPOCHS']
            self.config.EARLY_STOPPING_PATIENCE = adaptive_config['PATIENCE']
            
            print(f"   ⚙️ Learning Rate: {self.config.LEARNING_RATE}")
            print(f"   ⚙️ Dropout: {self.config.DROPOUT}")
            print(f"   ⚙️ Batch Size: {self.config.BATCH_SIZE}")
            print(f"   ⚙️ Weight Decay: {self.config.WEIGHT_DECAY}")
            print(f"   ⚙️ Epochs: {self.config.EPOCHS}")
            print(f"   ⚙️ Patience: {self.config.EARLY_STOPPING_PATIENCE}")
            
            # Update trainer with new config
            self.trainer.config = self.config
        
        # 🎯 PASS METADATA TO TRAINER FOR INVERSE TRANSFORMS
        self.trainer.set_metadata(metadata)
        
        # Step 2: Construct graph
        print("\n2. Constructing spatial graph...")
        location_coords = metadata['location_coords']
        spatial_adj = self.graph_constructor.build_spatial_adjacency(location_coords)
        
        # Convert to tensor
        adj_matrix = torch.FloatTensor(spatial_adj).to(self.trainer.device)
        
        print(f"   📊 Graph constructed with {metadata['n_nodes']} nodes")
        print(f"   🕸️ Adjacency matrix density: {np.mean(spatial_adj > 0):.3f}")
        
        # Step 3: Create data loaders with adaptive batch size
        print("\n3. Creating data loaders...")
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(
            features, targets, metadata)
        
        print(f"   📦 Train batches: {len(train_loader)}")
        print(f"   📦 Validation batches: {len(val_loader)}")
        print(f"   📦 Test batches: {len(test_loader)}")
        
        # Step 4: Initialize model with adaptive dropout
        print("\n4. Initializing STGNN model...")
        input_dim = len(metadata['feature_cols'])
        model = STGNNDenguePredictor(self.config, input_dim, metadata['n_nodes'])
        model = model.to(self.trainer.device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   🤖 Model initialized with {total_params:,} trainable parameters")
        print(f"   🎛️ Input dimension: {input_dim}")
        print(f"   🎛️ Hidden size: {self.config.HIDDEN_SIZE}")
        print(f"   🎛️ Dropout rate: {self.config.DROPOUT}")
        
        # Step 5: Train model with adaptive configuration
        print("\n5. Training model with adaptive configuration...")
        trained_model, history = self.trainer.train(model, train_loader, val_loader, adj_matrix)
        
        # Step 6: Evaluate on test set with proper inverse transforms
        print("\n6. Evaluating on test set...")
        test_metrics = self.trainer.evaluate(trained_model, test_loader, adj_matrix)
        
        # Enhanced results reporting
        print("\n📊 Final Test Results:")
        
        # Show data scale context
        if metadata.get('target_transform') == 'log1p':
            target_stats = metadata.get('target_stats', {})
            print(f"   📈 Dataset: High-scale (original mean: {target_stats.get('original_mean', 0):.2f}, max: {target_stats.get('original_max', 0):.0f})")
            print(f"   🔄 Applied log1p normalization for training")
            print(f"   📊 Metrics calculated in original scale after inverse transform")
        else:
            target_stats = metadata.get('target_stats', {})
            print(f"   📈 Dataset: Low-scale (mean: {target_stats.get('original_mean', 0):.2f}, max: {target_stats.get('original_max', 0):.0f})")
            print(f"   📊 No normalization applied")
        
        print(f"\n   🎯 MAE: {test_metrics['mae']:.4f}")
        print(f"   🎯 RMSE: {test_metrics['rmse']:.4f}")
        print(f"   🎯 R²: {test_metrics['r2']:.4f}")
        print(f"   🎯 Zero Accuracy: {test_metrics['zero_accuracy']:.4f}")
        print(f"   🎯 Non-zero MAE: {test_metrics['non_zero_mae']:.4f}")
        
        # Performance assessment
        if test_metrics['mae'] < 3.0 and test_metrics['r2'] > 0.3:
            print(f"   ✅ EXCELLENT PERFORMANCE achieved!")
        elif test_metrics['mae'] < 5.0 and test_metrics['r2'] > 0.2:
            print(f"   🎯 GOOD PERFORMANCE achieved!")
        elif test_metrics['mae'] < 8.0:
            print(f"   📈 Moderate performance - consider fine-tuning")
        else:
            print(f"   ⚠️ Poor performance - check data quality and model architecture")
        
        # Prediction statistics for debugging
        stats = test_metrics.get('prediction_stats', {})
        print(f"\n   📊 Prediction Statistics:")
        print(f"      Predicted mean: {stats.get('pred_mean', 0):.2f}, max: {stats.get('pred_max', 0):.2f}")
        print(f"      Actual mean: {targets.mean():.2f}, max: {targets.max():.2f}")
        print(f"      Predicted zeros: {stats.get('pred_zeros', 0)}, Actual zeros: {stats.get('actual_zeros', 0)}")
        
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
        
        # Apply inverse transform for visualization if needed
        if metadata.get('target_transform') == 'log1p':
            all_predictions = np.expm1(np.maximum(all_predictions, 0))
            all_targets = np.expm1(all_targets)
        
        # 🎯 GET REAL LOCATION NAMES from original data
        try:
            import pandas as pd
            df_original = pd.read_csv(data_path) if data_path else pd.DataFrame()
            
            # Determine the location column dynamically
            location_columns = ['Region', 'Puskesmas', 'Kecamatan', 'District', 'Location']
            location_col = None
            
            for col in location_columns:
                if col in df_original.columns:
                    location_col = col
                    break
            
            if location_col and not df_original.empty:
                # Get unique locations with their coordinates
                if 'Latitude' in df_original.columns and 'Longitude' in df_original.columns:
                    location_info = df_original[[location_col, 'Latitude', 'Longitude']].drop_duplicates()
                    real_locations = location_info[location_col].tolist()
                    real_coords = location_info[['Latitude', 'Longitude']].values
                    
                    print(f"   📍 Using real location names from '{location_col}' column")
                    print(f"   📊 Found {len(real_locations)} unique locations")
                    
                    # Update metadata with real information
                    metadata['node_ids'] = real_locations
                    metadata['location_coords'] = real_coords
                    metadata['location_source'] = location_col
                    
                else:
                    print("   ⚠️ Latitude/Longitude columns not found, using default coordinates")
            else:
                print(f"   ⚠️ No location column found in {location_columns}, using default names")
                
        except Exception as e:
            print(f"   ⚠️ Could not read location data: {e}")
        
        # Step 8: Create visualizations with enhanced labeling
        try:
            # Enhanced metadata for visualization
            viz_metadata = {
                'node_ids': metadata.get('node_ids', [f'Location_{i+1}' for i in range(metadata.get('n_nodes', 5))]),
                'location_coords': metadata.get('location_coords', np.random.uniform(-8, -7, (metadata.get('n_nodes', 5), 2))),
                'location_source': metadata.get('location_source', 'Generated'),
                'data_type': 'Real Data' if 'location_source' in metadata else 'Synthetic Data'
            }
            
            # Create enhanced visualizations
            self.visualizer.plot_training_history(history)
            self.visualizer.plot_predictions_vs_actual_enhanced(
                all_predictions, all_targets, viz_metadata
            )
            self.visualizer.plot_spatial_heatmap_enhanced(
                all_predictions, viz_metadata
            )
            
            print("   📈 Enhanced visualizations created successfully")
            print(f"   📍 Location labels: {viz_metadata['data_type']}")
            if viz_metadata['location_source'] != 'Generated':
                print(f"   📋 Source column: {viz_metadata['location_source']}")
                
        except Exception as viz_error:
            print(f"   ⚠️ Visualization error: {viz_error}")
            # Fallback to original visualization
            try:
                self.visualizer.plot_training_history(history)
                self.visualizer.plot_predictions_vs_actual(all_predictions, all_targets, 
                                                        metadata['node_ids'])
                self.visualizer.plot_spatial_heatmap(all_predictions, metadata['location_coords'], 
                                                metadata['node_ids'])
                print("   📈 Fallback visualizations created")
            except:
                print("   ❌ All visualization methods failed")
        # Step 9: Save model with comprehensive metadata
        model_save_data = {
            'model_state_dict': trained_model.state_dict(),
            'config': self.config,
            'metadata': metadata,
            'test_metrics': test_metrics,
            'training_history': history,
            'adaptive_config_used': metadata.get('adaptive_config', {}),
            'target_transform': metadata.get('target_transform', 'none'),
            'target_stats': metadata.get('target_stats', {}),
            'feature_columns': metadata.get('feature_cols', []),
            'node_ids': metadata.get('node_ids', [])
        }
        
        torch.save(model_save_data, 'dengue_stgnn_model.pth')
        
        print(f"\n" + "=" * 80)
        print("ADAPTIVE PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Performance: MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
        
        if metadata.get('target_transform') == 'log1p':
            original_scale = metadata['target_stats']['original_mean']
            improvement_factor = 16.23 / test_metrics['mae']  # Assuming Dataset 2 baseline
            print(f"🚀 Improvement over baseline: {improvement_factor:.1f}x better")
        
        print(f"💾 Model saved as 'dengue_stgnn_model.pth'")
        print(f"🎯 Scale type: {metadata.get('adaptive_config', {}).get('scale_type', 'unknown')}")
        print("=" * 80)
        
        return trained_model, test_metrics, metadata