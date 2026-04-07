import torch
import numpy as np
from typing import Dict, Tuple
from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor
from models.stgnn import STGNNDenguePredictor
from training.trainer import DengueTrainer
from utils.visualization import DengueVisualizer
from analysis.paper_results import generate_paper_results 

class DenguePredictionSystem:
    """Main system for dengue prediction using STGNN with adaptive configuration"""
    
    def __init__(self, config):
        self.config = config

        self.model = None
        self.adj_matrix = None
        self.metadata = None
        self.trainer = None
        
        self.preprocessor = DengueDataPreprocessor(config)
        self.graph_constructor = GraphConstructor(config)
        self.visualizer = DengueVisualizer()
        
    def run_complete_pipeline(self, data_path: str = None, generate_paper_analysis: bool = False):
        """Run the complete dengue prediction pipeline with adaptive configuration"""
        
        print("=" * 80)
        print("ADAPTIVE DENGUE PREDICTION USING SPATIO-TEMPORAL GNN")
        print("=" * 80)
        
        # ================================================================
        # STEP 1: LOAD AND PREPROCESS DATA
        # ================================================================
        print("\n1. Loading and preprocessing data...")
        if data_path:
            df = self.preprocessor.load_data(data_path)
        else:
            df = self.preprocessor.load_data("dummy_path")
        
        features, targets, metadata = self.preprocessor.preprocess_data(df)
        
        # ✅ Store metadata
        self.metadata = metadata

        print(f"✅ Data loaded: {features.shape}")
        print(f"   Nodes: {metadata['n_nodes']}")
        print(f"   Features: {len(metadata['feature_cols'])}")
        
        # ================================================================
        # STEP 2: CONSTRUCT GRAPH
        # ================================================================
        print("\n2. Constructing spatial graph...")
        location_coords = metadata['location_coords']
        spatial_adj = self.graph_constructor.build_spatial_adjacency(
            location_coords, k_neighbors=3
        )
        
        # ✅ Store adjacency matrix
        self.adj_matrix = torch.FloatTensor(spatial_adj)
        
        print(f"✅ Graph built:")
        print(f"   Nodes: {spatial_adj.shape[0]}")
        print(f"   Density: {(spatial_adj > 0).sum() / spatial_adj.size:.3f}")
        
        # ================================================================
        # STEP 3: INITIALIZE TRAINER
        # ================================================================
        print("\n3. Initializing trainer...")
        
        # ✅ Create new trainer instance (don't use __init__ one)
        self.trainer = DengueTrainer(self.config)
        self.trainer.set_metadata(metadata)
        
        print(f"✅ Trainer initialized on device: {self.trainer.device}")
        
        # ================================================================
        # STEP 4: CREATE DATA LOADERS
        # ================================================================
        print("\n4. Creating data loaders...")
        
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(
            features, targets, metadata
        )
        
        print(f"   📦 Train batches: {len(train_loader)}")
        print(f"   📦 Validation batches: {len(val_loader)}")
        print(f"   📦 Test batches: {len(test_loader)}")
        
        # ================================================================
        # STEP 5: APPLY ADAPTIVE CONFIGURATION
        # ================================================================
        print("\n5. Applying adaptive configuration...")
        
        # ✅ Get adaptive config from metadata
        if 'adaptive_config' in metadata:
            adaptive_config = metadata['adaptive_config']
            print(f"   Using {adaptive_config['scale_type']} scale configuration")
            
            # Update config
            self.config.LEARNING_RATE = adaptive_config['LEARNING_RATE']
            self.config.DROPOUT = adaptive_config['DROPOUT']
            self.config.BATCH_SIZE = adaptive_config['BATCH_SIZE']
            self.config.WEIGHT_DECAY = adaptive_config['WEIGHT_DECAY']
            self.config.EPOCHS = adaptive_config['EPOCHS']
            self.config.EARLY_STOPPING_PATIENCE = adaptive_config['PATIENCE']
            
            # ✅ Get hidden_dim and num_layers
            hidden_dim = adaptive_config.get('HIDDEN_DIM', self.config.HIDDEN_DIM)
            num_layers = adaptive_config.get('NUM_LAYERS', self.config.NUM_LAYERS)
            
            print(f"   ⚙️ Learning Rate: {self.config.LEARNING_RATE}")
            print(f"   ⚙️ Dropout: {self.config.DROPOUT}")
            print(f"   ⚙️ Batch Size: {self.config.BATCH_SIZE}")
            print(f"   ⚙️ Hidden Dim: {hidden_dim}")
            print(f"   ⚙️ Num Layers: {num_layers}")
            
            # Update trainer config
            self.trainer.config = self.config
        else:
            # ✅ Use default config
            hidden_dim = self.config.HIDDEN_DIM
            num_layers = self.config.NUM_LAYERS
            print(f"   Using default configuration")
        
        # ================================================================
        # STEP 6: INITIALIZE MODEL
        # ================================================================
        print("\n6. Initializing STGNN model...")
        
        input_dim = len(metadata['feature_cols'])
        
        model = STGNNDenguePredictor(
            config=self.config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers
        )
        
        # ✅ Store model BEFORE training
        self.model = model
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   🤖 Model initialized: {total_params:,} parameters")
        print(f"   🎛️ Input dimension: {input_dim}")
        print(f"   🎛️ Hidden dimension: {hidden_dim}")
        
        # ================================================================
        # STEP 7: TRAIN MODEL
        # ================================================================
        print("\n7. Training model...")
        
        trained_model, history = self.trainer.train(
            model, train_loader, val_loader, self.adj_matrix
        )
        
        # ✅ Update with trained model
        self.model = trained_model
        
        print(f"✅ Training completed:")
        print(f"   Epochs: {len(history['train_loss'])}")
        print(f"   Best val loss: {min(history['val_loss']):.4f}")
        
        # ================================================================
        # STEP 8: EVALUATE ON TEST SET
        # ================================================================
        print("\n8. Evaluating on test set...")
        
        test_metrics = self.trainer.evaluate(
            trained_model, test_loader, self.adj_matrix
        )
        
        print(f"✅ Test evaluation completed:")
        print(f"   MAE:  {test_metrics['mae']:.3f} cases/week")
        print(f"   RMSE: {test_metrics['rmse']:.3f} cases/week")
        print(f"   R²:   {test_metrics['r2']:.3f}")
        
        # ================================================================
        # STEP 9: ENHANCED RESULTS REPORTING
        # ================================================================
        print("\n📊 Final Test Results:")
        
        # Show data scale context
        if metadata.get('target_transform') == 'log1p':
            target_stats = metadata.get('target_stats', {})
            print(f"   📈 Dataset: High-scale (mean: {target_stats.get('original_mean', 0):.2f})")
            print(f"   🔄 Applied log1p normalization")
        else:
            print(f"   📈 Dataset: Low-scale")
            print(f"   📊 No normalization applied")
        
        print(f"\n   🎯 MAE: {test_metrics['mae']:.4f}")
        print(f"   🎯 RMSE: {test_metrics['rmse']:.4f}")
        print(f"   🎯 R²: {test_metrics['r2']:.4f}")
        
        # Performance assessment
        if test_metrics['mae'] < 3.0 and test_metrics['r2'] > 0.3:
            print(f"   ✅ EXCELLENT PERFORMANCE!")
        elif test_metrics['mae'] < 5.0 and test_metrics['r2'] > 0.2:
            print(f"   🎯 GOOD PERFORMANCE!")
        elif test_metrics['mae'] < 8.0:
            print(f"   📈 Moderate performance")
        else:
            print(f"   ⚠️ Check data quality")
        
        # ================================================================
        # STEP 10: GENERATE PAPER ANALYSIS (OPTIONAL)
        # ================================================================
        paper_results = None
        
        if generate_paper_analysis:
            print("\n" + "="*80)
            print("📝 GENERATING PAPER RESULTS")
            print("="*80)
            
            try:
                from analysis.paper_results import generate_paper_results
                
                # ✅ All instance variables now exist
                paper_results = generate_paper_results(
                    test_loader=test_loader,
                    model=self.model,
                    adj_matrix=self.adj_matrix,
                    trainer=self.trainer,
                    metadata=self.metadata,
                    save_dir='paper_outputs'
                )
                
                print("\n✅ Paper results generated!")
                print("📁 Check 'paper_outputs/' directory")
                
            except Exception as e:
                print(f"\n⚠️ Paper generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ================================================================
        # STEP 11: GENERATE VISUALIZATIONS
        # ================================================================
        print("\n11. Generating visualizations...")
        
        try:
            # Get predictions for visualization
            trained_model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_features, batch_targets in test_loader:
                    batch_features = batch_features.to(self.trainer.device)
                    outputs = trained_model(batch_features, self.adj_matrix)
                    predictions = outputs['predictions'].cpu().numpy()
                    targets_batch = batch_targets.cpu().numpy()
                    
                    all_predictions.extend(predictions.flatten())
                    all_targets.extend(targets_batch.flatten())
            
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            # Apply inverse transform if needed
            if metadata.get('target_transform') == 'log1p':
                all_predictions = np.expm1(np.maximum(all_predictions, 0))
                all_targets = np.expm1(all_targets)
            
            # ✅ Store predictions and actuals in metadata for later use
            metadata['test_predictions'] = all_predictions.tolist()
            metadata['test_actuals'] = all_targets.tolist()
            
            # Create visualizations
            self.visualizer.plot_training_history(history)
            print("   📈 Training history saved")
            
            self.visualizer.plot_predictions_vs_actual(
                all_predictions, all_targets,
                metadata.get('node_ids', [f'Node_{i}' for i in range(metadata['n_nodes'])])
            )
            print("   📊 Predictions vs actual saved")
            
            self.visualizer.plot_spatial_heatmap(
                all_predictions,
                metadata.get('location_coords'),
                metadata.get('node_ids', [f'Node_{i}' for i in range(metadata['n_nodes'])])
            )
            print("   🗺️ Spatial heatmap saved")
            
        except Exception as viz_error:
            print(f"   ⚠️ Visualization error: {viz_error}")
        
        # ================================================================
        # STEP 12: SAVE MODEL
        # ================================================================
        print("\n12. Saving model...")
        
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
        print(f"   💾 Model saved: dengue_stgnn_model.pth")
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print(f"\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Final: MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
        print("=" * 80)
        
        # ✅ Return in correct format for app3.py
        return trained_model, test_metrics, metadata, history

# ================================================================
# CONVENIENCE FUNCTION FOR PAPER GENERATION FROM CHECKPOINT
# ================================================================
def generate_paper_results_from_checkpoint(
    checkpoint_path='dengue_stgnn_model.pth',
    data_path='data/fix.csv'
):
    """
    Generate paper results from a saved checkpoint
    
    Usage:
        from experiments.dengue_pipeline import generate_paper_results_from_checkpoint
        results = generate_paper_results_from_checkpoint()
    """
    
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import DengueDataset, collate_fn
    
    print("\n📄 Loading checkpoint and generating paper results...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    metadata = checkpoint['metadata']
    
    print(f"✅ Checkpoint loaded")
    
    # Load and preprocess data
    preprocessor = DengueDataPreprocessor(config)
    df = preprocessor.load_data(data_path)
    features, targets, metadata = preprocessor.preprocess_data(df)
    
    # Create trainer
    trainer = DengueTrainer(config)
    trainer.set_metadata(metadata)
    
    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        features, targets, metadata
    )
    
    # ✅ Infer model architecture from checkpoint
    print("\n🔍 Inferring model architecture...")
    state_dict = checkpoint['model_state_dict']
    
    # Get hidden_dim from input projection layer
    if 'input_proj.0.weight' in state_dict:
        hidden_dim = state_dict['input_proj.0.weight'].shape[0]
        input_dim = state_dict['input_proj.0.weight'].shape[1]
    else:
        hidden_dim = config.HIDDEN_DIM if hasattr(config, 'HIDDEN_DIM') else 128
        input_dim = len(metadata['feature_cols'])
    
    # Count graph conv layers
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith('graph_convs.'):
            layer_idx = int(key.split('.')[1])
            num_layers = max(num_layers, layer_idx + 1)
    
    if num_layers == 0:
        num_layers = config.NUM_LAYERS if hasattr(config, 'NUM_LAYERS') else 3
    
    print(f"   Input dim: {input_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Num layers: {num_layers}")
    
    # Reconstruct model
    model = STGNNDenguePredictor(
        config=config,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(trainer.device)
    model.eval()
    
    print(f"✅ Model reconstructed: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Reconstruct adjacency matrix
    graph_constructor = GraphConstructor(config)
    location_coords = metadata['location_coords']
    adj_matrix = torch.FloatTensor(
        graph_constructor.build_spatial_adjacency(location_coords)
    ).to(trainer.device)
    
    # Generate paper results
    from analysis.paper_results import generate_paper_results
    
    print("\n" + "="*80)
    print("📊 Generating paper results...")
    print("="*80)
    
    paper_results = generate_paper_results(
        test_loader=test_loader,
        model=model,
        adj_matrix=adj_matrix,
        trainer=trainer,
        metadata=metadata,
        save_dir='paper_outputs'
    )
    
    return paper_results