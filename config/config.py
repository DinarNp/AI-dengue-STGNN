class Config:
    """Configuration parameters for STGNN Dengue Prediction with adaptive support"""
    
    # Data parameters
    DATA_PATH = "data/fix.csv"
    WINDOW_SIZE = 8  # For weekly data - Increased from 4 to capture longer temporal patterns
    WINDOW_SIZE_MONTHLY = 4  # For monthly data - Reduced to work with smaller datasets (4 months history)
    FORECAST_HORIZON = 4  # Weeks ahead to predict (~1 month), matching the manuscript's "one month ahead" claim
    FORECAST_HORIZON_MONTHLY = 1  # For monthly-cadence data, 1 step = 1 month ahead

    # Chronological (forward-chaining) split fractions, applied per-location.
    # No shuffling: each location's samples are already time-ordered when this
    # is applied (see DengueDataPreprocessor.create_lag_features), so the first
    # TRAIN_FRACTION weeks train, the next VAL_FRACTION validate, and the
    # remaining weeks are held out as test.
    TRAIN_FRACTION = 0.70
    VAL_FRACTION = 0.15

    # Optional caps for quick experiments/CV folds (None = use the scale-based
    # adaptive defaults in DengueDataPreprocessor._get_adaptive_config).
    EPOCHS_OVERRIDE = None
    PATIENCE_OVERRIDE = None
    
    # Model architecture - Optimized configuration
    NODE_FEATURE_DIM = 128  # Increased from 64
    EDGE_FEATURE_DIM = 64   # Increased from 32
    HIDDEN_SIZE = 128       # Increased from 64
    ATTENTION_HEADS = 4     # Increased from 2
    GNN_LAYERS = 4          # Increased from 3
    LSTM_LAYERS = 3         # Increased from 2
    DROPOUT = 0.15           # Reduced from 0.3 for better generalization
    
    # Model architecture - ENSURE DIVISIBILITY
    NUM_HEADS = 4
    HIDDEN_DIM = 256  # 128 / 4 = 32 ✅
    NUM_LAYERS = 4    # Number of graph convolutional layers (matches GNN_LAYERS concept)

    # Graph construction
    SPATIAL_THRESHOLD = 0.1
    ENV_SIMILARITY_THRESHOLD = 0.7
    K_NEAREST = 3           # Increased from 3
    
    # Training parameters - Optimized for better performance
    BATCH_SIZE = 16         # Increased to avoid batch size issues
    NUM_WORKERS = 0
    PIN_MEMORY = False
    LEARNING_RATE = 0.0005  # Reduced for more stable training
    EPOCHS = 1000            # Increased for better convergence
    WEIGHT_DECAY = 1e-5     # Reduced for less regularization
    EARLY_STOPPING_PATIENCE = 100  # Increased patience
    USE_AMP = True
    
    # Loss weights - Optimized for dengue prediction
    REGRESSION_WEIGHT = 0.8      # Increased from 0.7
    TEMPORAL_REG_WEIGHT = 0.05   # Reduced from 0.1
    SPATIAL_REG_WEIGHT = 0.05    # Reduced from 0.1
    ZERO_INFLATION_WEIGHT = 0.2  # Reduced from 0.3
    
    # Adaptive configuration thresholds
    HIGH_SCALE_THRESHOLD = 8.0  # Mean target value threshold for high-scale data
    
    # Performance targets - More realistic targets
    TARGET_MAE_LOW_SCALE = 0.5    # Reduced from 1.0
    TARGET_MAE_HIGH_SCALE = 2.0   # Reduced from 3.0
    TARGET_R2_MINIMUM = 0.3       # Increased from 0.2

    # ADD THIS - Device configuration
    USE_ATTENTION = True  # Enable attention mechanism in STGNN
    
    @classmethod
    def get_adaptive_config(cls, target_mean: float, dataset_characteristics: dict = None):
        """
        Get adaptive configuration based on dataset characteristics
        
        Args:
            target_mean: Mean value of target variable
            dataset_characteristics: Additional dataset info (optional)
        
        Returns:
            Dict with adaptive configuration parameters
        """
        
        if target_mean > cls.HIGH_SCALE_THRESHOLD:
            # High-scale dataset configuration - Optimized
            config = {
                'scale_type': 'high',
                'LEARNING_RATE': 0.0005,     # Further reduced for stability
                'DROPOUT': 0.15,              # Reduced for better generalization
                'BATCH_SIZE': 8,              # Smaller batches for stability
                'WEIGHT_DECAY': 1e-6,         # Reduced L2 regularization
                'EPOCHS': 1000,                # More epochs for convergence
                'PATIENCE': 100,               # More patience
                'HIDDEN_DIM': 128,            # ✅ ADD THIS
                'NUM_LAYERS': 4,              # ✅ ADD THIS
                'target_transform': 'log1p',  # Apply log transformation
                'loss_function': 'huber',     # Robust to outliers
                'scheduler_factor': 0.2,      # More aggressive LR reduction
                'grad_clip_norm': 0.3,        # Tighter gradient clipping
            }
            print(f"🔧 High-scale configuration selected (target_mean={target_mean:.2f})")
            
        else:
            # Low-scale dataset configuration - Optimized
            config = {
                'scale_type': 'low',
                'LEARNING_RATE': 0.0005,      # Reduced for stability
                'DROPOUT': 0.1,               # Reduced dropout
                'BATCH_SIZE': 8,              # Optimal batch size
                'WEIGHT_DECAY': 1e-6,         # Minimal L2 regularization
                'EPOCHS': 600,                # More epochs
                'PATIENCE': 50,               # More patience
                'HIDDEN_DIM': 128,            # ✅ ADD THIS
                'NUM_LAYERS': 3,              # ✅ ADD THIS
                'target_transform': 'none',   # No transformation needed
                'loss_function': 'mse',       # Standard MSE loss
                'scheduler_factor': 0.3,      # Moderate LR reduction
                'grad_clip_norm': 0.5,        # Standard gradient clipping
            }
            print(f"🔧 Low-scale configuration selected (target_mean={target_mean:.2f})")
        
        # Add dataset-specific adjustments if provided
        if dataset_characteristics:
            zero_ratio = dataset_characteristics.get('zero_ratio', 0.0)
            n_locations = dataset_characteristics.get('n_locations', 1)
            
            # Adjust for zero-inflation
            if zero_ratio > 0.5:  # High zero-inflation
                config['ZERO_INFLATION_WEIGHT'] = 0.3
                config['zero_threshold'] = 0.2  # Lower threshold for zero classification
                print(f"   📊 High zero-inflation detected ({zero_ratio:.2f}), adjusting weights")
            else:
                config['ZERO_INFLATION_WEIGHT'] = 0.15
                config['zero_threshold'] = 0.4
            
            # Adjust for number of locations
            if n_locations < 5:  # Few locations
                config['SPATIAL_REG_WEIGHT'] = 0.02  # Reduce spatial regularization
                config['K_NEAREST'] = min(3, n_locations - 1)
                print(f"   🗺️ Few locations ({n_locations}), reducing spatial regularization")
            elif n_locations > 20:  # Many locations
                config['SPATIAL_REG_WEIGHT'] = 0.08  # Moderate spatial regularization
                config['BATCH_SIZE'] = min(config['BATCH_SIZE'] * 2, 32)  # Larger batches
                print(f"   🗺️ Many locations ({n_locations}), increasing batch size")
        
        return config
    
    @classmethod
    def validate_config(cls, config_dict: dict):
        """Validate and sanitize configuration parameters"""
        
        # Ensure reasonable bounds
        config_dict['LEARNING_RATE'] = max(1e-6, min(config_dict['LEARNING_RATE'], 0.1))
        config_dict['DROPOUT'] = max(0.0, min(config_dict['DROPOUT'], 0.8))
        config_dict['BATCH_SIZE'] = max(4, min(config_dict['BATCH_SIZE'], 128))
        config_dict['WEIGHT_DECAY'] = max(0.0, min(config_dict['WEIGHT_DECAY'], 0.01))
        config_dict['EPOCHS'] = max(10, min(config_dict['EPOCHS'], 1000))
        config_dict['PATIENCE'] = max(5, min(config_dict['PATIENCE'], 100))
        
        return config_dict
    
    def update_from_adaptive_config(self, adaptive_config: dict):
        """Update config instance with adaptive parameters"""
        
        # Validate first
        adaptive_config = self.validate_config(adaptive_config)
        
        # Update parameters
        for key, value in adaptive_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        print(f"✅ Config updated with adaptive parameters:")
        print(f"   Scale type: {adaptive_config.get('scale_type', 'unknown')}")
        print(f"   Learning rate: {self.LEARNING_RATE}")
        print(f"   Dropout: {self.DROPOUT}")
        print(f"   Batch size: {self.BATCH_SIZE}")
        
    def get_performance_targets(self, scale_type: str):
        """Get performance targets based on scale type"""
        
        if scale_type == 'high':
            return {
                'mae_target': self.TARGET_MAE_HIGH_SCALE,
                'r2_target': self.TARGET_R2_MINIMUM,
                'zero_accuracy_target': 0.6,
                'improvement_baseline': 16.23  # Baseline MAE for high-scale
            }
        else:
            return {
                'mae_target': self.TARGET_MAE_LOW_SCALE,
                'r2_target': self.TARGET_R2_MINIMUM,
                'zero_accuracy_target': 0.7,
                'improvement_baseline': 2.0    # Baseline MAE for low-scale
            }
    
    def __str__(self):
        """String representation of config"""
        config_str = "DengueConfig:\n"
        config_str += f"  Model: Hidden={self.HIDDEN_SIZE}, Dropout={self.DROPOUT}\n"
        config_str += f"  Training: LR={self.LEARNING_RATE}, Batch={self.BATCH_SIZE}, Epochs={self.EPOCHS}\n"
        config_str += f"  Regularization: Weight_Decay={self.WEIGHT_DECAY}, Patience={self.EARLY_STOPPING_PATIENCE}\n"
        return config_str