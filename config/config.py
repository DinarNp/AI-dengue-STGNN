class Config:
    """Configuration parameters for STGNN Dengue Prediction"""
    
    # Data parameters
    DATA_PATH = "data/test2.csv"
    WINDOW_SIZE = 4  # Restore original window size
    FORECAST_HORIZON = 1
    
    # Model architecture - Adjust for real data
    NODE_FEATURE_DIM = 64
    EDGE_FEATURE_DIM = 32
    HIDDEN_SIZE = 64
    ATTENTION_HEADS = 2
    GNN_LAYERS = 3
    LSTM_LAYERS = 2
    DROPOUT = 0.3
    
    # Graph construction - Adjust for 6 nodes
    SPATIAL_THRESHOLD = 0.1
    ENV_SIMILARITY_THRESHOLD = 0.7
    K_NEAREST = 3  # Max 3 neighbors for 6 nodes total
    
    # Training parameters - Adjust for real data (1248 samples)
    BATCH_SIZE = 16        # Increase batch size for GPU
    NUM_WORKERS = 0        # For DataLoader
    PIN_MEMORY = False      # Speed up GPU transfer
    LEARNING_RATE = 0.0005
    EPOCHS = 100       # Reasonable epochs
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 20
    USE_AMP = True
    
    # Loss weights
    REGRESSION_WEIGHT = 0.7
    TEMPORAL_REG_WEIGHT = 0.1
    SPATIAL_REG_WEIGHT = 0.1
    ZERO_INFLATION_WEIGHT = 0.3