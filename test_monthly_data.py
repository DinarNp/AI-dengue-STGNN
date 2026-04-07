#!/usr/bin/env python3
"""
Test script to verify monthly data loading and processing
"""

import torch
import numpy as np
import pandas as pd
from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from data.dataset import DengueDataset
from torch.utils.data import DataLoader

def main():
    print("="*80)
    print("TESTING MONTHLY DATA PIPELINE")
    print("="*80)
    
    # 1. Load and preprocess data
    print("\n1. Loading monthly data...")
    config = Config()
    preprocessor = DengueDataPreprocessor(config)
    
    df = preprocessor.load_data('data/monthly_data_3kab.csv')
    print(f"   Loaded: {df.shape}")
    print(f"   Regions: {df['Region'].unique().tolist()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Preprocess
    print("\n2. Preprocessing...")
    features, targets, metadata = preprocessor.preprocess_data(df)
    print(f"   Features: {features.shape}")
    print(f"   Targets: {targets.shape}")
    print(f"   Nodes: {metadata['n_nodes']}")
    print(f"   NaN in features: {np.isnan(features).sum()}")
    print(f"   NaN in targets: {np.isnan(targets).sum()}")
    print(f"   Target transform: {metadata.get('target_transform', 'none')}")
    
    # 3. Create dataset
    print("\n3. Creating dataset...")
    try:
        dataset = DengueDataset(
            features=features,
            targets=targets,
            metadata=metadata,
            window_size=config.WINDOW_SIZE,
            forecast_horizon=config.FORECAST_HORIZON
        )
        print(f"   ✅ Dataset created with {len(dataset)} sequences")
        
        # 4. Create dataloader
        print("\n4. Creating dataloader...")
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        print(f"   ✅ Dataloader created with {len(loader)} batches")
        
        # 5. Test batch
        print("\n5. Testing batch...")
        batch_features, batch_targets = next(iter(loader))
        print(f"   Batch features shape: {batch_features.shape}")
        print(f"   Batch targets shape: {batch_targets.shape}")
        print(f"   NaN in batch features: {torch.isnan(batch_features).sum().item()}")
        print(f"   NaN in batch targets: {torch.isnan(batch_targets).sum().item()}")
        
        if torch.isnan(batch_features).any():
            print("\n   ❌ ERROR: NaN detected in batch features!")
            print(f"   NaN locations: {torch.where(torch.isnan(batch_features))}")
        else:
            print("\n   ✅ No NaN in batch - data is clean!")
            
    except Exception as e:
        print(f"\n   ❌ ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    return True

if __name__ == "__main__":
    main()
