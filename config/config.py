class Config:
    """Configuration parameters for STGNN Dengue Prediction with adaptive support"""
    
    # Data parameters
    DATA_PATH = "data/test2.csv"
    WINDOW_SIZE = 4
    FORECAST_HORIZON = 1
    
    # Model architecture - Base configuration
    NODE_FEATURE_DIM = 64
    EDGE_FEATURE_DIM = 32
    HIDDEN_SIZE = 64
    ATTENTION_HEADS = 2
    GNN_LAYERS = 3
    LSTM_LAYERS = 2
    DROPOUT = 0.3  # Will be overridden by adaptive config
    
    # Graph construction
    SPATIAL_THRESHOLD = 0.1
    ENV_SIMILARITY_THRESHOLD = 0.7
    K_NEAREST = 3
    
    # Training parameters - Base configuration (will be overridden by adaptive config)
    BATCH_SIZE = 16          # Will be overridden
    NUM_WORKERS = 0
    PIN_MEMORY = False
    LEARNING_RATE = 0.0005   # Will be overridden
    EPOCHS = 100             # Will be overridden
    WEIGHT_DECAY = 1e-4      # Will be overridden
    EARLY_STOPPING_PATIENCE = 20  # Will be overridden
    USE_AMP = True
    
    # Loss weights
    REGRESSION_WEIGHT = 0.7
    TEMPORAL_REG_WEIGHT = 0.1
    SPATIAL_REG_WEIGHT = 0.1
    ZERO_INFLATION_WEIGHT = 0.3
    
    # Adaptive configuration thresholds
    HIGH_SCALE_THRESHOLD = 8.0  # Mean target value threshold for high-scale data
    
    # Performance targets
    TARGET_MAE_LOW_SCALE = 1.0    # Target MAE for low-scale datasets
    TARGET_MAE_HIGH_SCALE = 3.0   # Target MAE for high-scale datasets
    TARGET_R2_MINIMUM = 0.2       # Minimum acceptable R² score
    
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
            # High-scale dataset configuration
            config = {
                'scale_type': 'high',
                'LEARNING_RATE': 0.0001,      # Lower LR for stability
                'DROPOUT': 0.4,               # Higher dropout for regularization
                'BATCH_SIZE': 32,             # Larger batches for stability
                'WEIGHT_DECAY': 0.001,        # Strong L2 regularization
                'EPOCHS': 200,                # Fewer epochs (faster convergence expected)
                'PATIENCE': 15,               # Less patience (stop early)
                'target_transform': 'log1p',  # Apply log transformation
                'loss_function': 'huber',     # Robust to outliers
                'scheduler_factor': 0.3,      # More aggressive LR reduction
                'grad_clip_norm': 0.5,        # Tighter gradient clipping
            }
            print(f"🔧 High-scale configuration selected (target_mean={target_mean:.2f})")
            
        else:
            # Low-scale dataset configuration  
            config = {
                'scale_type': 'low',
                'LEARNING_RATE': 0.001,       # Higher LR for faster learning
                'DROPOUT': 0.2,               # Lower dropout (less regularization needed)
                'BATCH_SIZE': 16,             # Smaller batches
                'WEIGHT_DECAY': 0.0001,       # Light L2 regularization
                'EPOCHS': 300,                # More epochs (may need more training)
                'PATIENCE': 25,               # More patience
                'target_transform': 'none',   # No transformation needed
                'loss_function': 'mse',       # Standard MSE loss
                'scheduler_factor': 0.5,      # Moderate LR reduction
                'grad_clip_norm': 1.0,        # Standard gradient clipping
            }
            print(f"🔧 Low-scale configuration selected (target_mean={target_mean:.2f})")
        
        # Add dataset-specific adjustments if provided
        if dataset_characteristics:
            zero_ratio = dataset_characteristics.get('zero_ratio', 0.0)
            n_locations = dataset_characteristics.get('n_locations', 1)
            
            # Adjust for zero-inflation
            if zero_ratio > 0.5:  # High zero-inflation
                config['ZERO_INFLATION_WEIGHT'] = 0.4
                config['zero_threshold'] = 0.3  # Lower threshold for zero classification
                print(f"   📊 High zero-inflation detected ({zero_ratio:.2f}), adjusting weights")
            else:
                config['ZERO_INFLATION_WEIGHT'] = 0.2
                config['zero_threshold'] = 0.5
            
            # Adjust for number of locations
            if n_locations < 5:  # Few locations
                config['SPATIAL_REG_WEIGHT'] = 0.05  # Reduce spatial regularization
                config['K_NEAREST'] = min(2, n_locations - 1)
                print(f"   🗺️ Few locations ({n_locations}), reducing spatial regularization")
            elif n_locations > 20:  # Many locations
                config['SPATIAL_REG_WEIGHT'] = 0.15  # Increase spatial regularization
                config['BATCH_SIZE'] = min(config['BATCH_SIZE'] * 2, 64)  # Larger batches
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