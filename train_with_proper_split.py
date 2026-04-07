"""
FIXED Training Script with:
1. Stratified random split (NOT time-based)
2. CPU training (avoid MPS gradient issues)
3. Better validation handling
4. Reduced model complexity for stability
"""

import sys
import os
import traceback
import torch
import numpy as np

print("\n" + "="*80)
print("🚀 DENGUE PREDICTION - FIXED TRAINING")
print("="*80)

# STEP 1: Import and Configuration
try:
    from config.config import Config
    from data.preprocessor import DengueDataPreprocessor
    from models.stgnn import STGNNDenguePredictor
    from models.graph_constructor import GraphConstructor
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

config = Config()

# 🔥 CRITICAL FIXES
print("\n🔧 Applying critical fixes...")
print("   1. Using CPU (not MPS) for stable gradients")
print("   2. Using stratified random split (not time-based)")
print("   3. Reduced complexity for better convergence")

# Override config for stability
config.EPOCHS = 200  # Reduced for faster testing
config.LEARNING_RATE = 0.001  # Higher LR for better learning
config.BATCH_SIZE = 16  # Larger batches for stability
config.DROPOUT = 0.1  # Less dropout
config.HIDDEN_DIM = 64  # Smaller hidden dim for faster convergence


print(f"\n📋 Configuration:")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Learning Rate: {config.LEARNING_RATE}")
print(f"   Batch Size: {config.BATCH_SIZE}")
print(f"   Hidden Dim: {config.HIDDEN_DIM}")
print(f"   Dropout: {config.DROPOUT}")

# STEP 2: Load and Preprocess Data
print("\n📊 Loading data...")
preprocessor = DengueDataPreprocessor(config)
df = preprocessor.load_data("data/fix.csv")
features, targets, metadata = preprocessor.preprocess_data(df)

print(f"✅ Data loaded: {features.shape}")

# STEP 3: Manual Stratified Random Split
print("\n🎲 Creating STRATIFIED RANDOM split...")

from sklearn.model_selection import train_test_split

n_samples = len(features)
n_nodes = metadata['n_nodes']
samples_per_node = n_samples // n_nodes

# Create location labels for stratification
location_labels = np.repeat(np.arange(n_nodes), samples_per_node)
if len(location_labels) < n_samples:
    location_labels = np.concatenate([
        location_labels,
        np.array([n_nodes-1] * (n_samples - len(location_labels)))
    ])

# Split: 70% train, 15% val, 15% test
train_idx, temp_idx = train_test_split(
    np.arange(n_samples),
    test_size=0.30,
    random_state=42,
    stratify=location_labels
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=location_labels[temp_idx]
)

# Verify split quality
train_targets_orig = np.expm1(targets[train_idx]) if metadata.get('target_transform') == 'log1p' else targets[train_idx]
val_targets_orig = np.expm1(targets[val_idx]) if metadata.get('target_transform') == 'log1p' else targets[val_idx]
test_targets_orig = np.expm1(targets[test_idx]) if metadata.get('target_transform') == 'log1p' else targets[test_idx]
overall_mean = np.expm1(targets).mean() if metadata.get('target_transform') == 'log1p' else targets.mean()

print(f"✅ Split created:")
print(f"   Train: {len(train_idx)} samples, mean={train_targets_orig.mean():.2f}")
print(f"   Val:   {len(val_idx)} samples, mean={val_targets_orig.mean():.2f}")
print(f"   Test:  {len(test_idx)} samples, mean={test_targets_orig.mean():.2f}")
print(f"   Overall mean: {overall_mean:.2f}")

test_bias = abs(test_targets_orig.mean() - overall_mean) / overall_mean * 100
print(f"   Test bias: {test_bias:.1f}% {'✅' if test_bias < 20 else '⚠️'}")

# STEP 4: Create Datasets Manually
print("\n🔧 Creating datasets...")

from data.dataset import DengueDataset, collate_fn
from torch.utils.data import DataLoader

train_dataset = DengueDataset(
    features[train_idx], targets[train_idx], metadata,
    config.WINDOW_SIZE, config.FORECAST_HORIZON
)
val_dataset = DengueDataset(
    features[val_idx], targets[val_idx], metadata,
    config.WINDOW_SIZE, config.FORECAST_HORIZON
)
test_dataset = DengueDataset(
    features[test_idx], targets[test_idx], metadata,
    config.WINDOW_SIZE, config.FORECAST_HORIZON
)

print(f"✅ Datasets created:")
print(f"   Train: {len(train_dataset)} sequences")
print(f"   Val:   {len(val_dataset)} sequences")
print(f"   Test:  {len(test_dataset)} sequences")

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=False, collate_fn=collate_fn)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# STEP 5: Build Graph
print("\n🗺️  Building spatial graph...")
graph_constructor = GraphConstructor(config)
spatial_adj = graph_constructor.build_spatial_adjacency(metadata['location_coords'], k_neighbors=3)
adj_matrix = torch.FloatTensor(spatial_adj)

print(f"✅ Graph built: {adj_matrix.shape}")

# STEP 6: Initialize Model
print("\n🧠 Initializing model...")

device = torch.device('cpu')  # FORCE CPU
print(f"   Device: {device}")

model = STGNNDenguePredictor(
    config=config,
    input_dim=len(metadata['feature_cols']),
    hidden_dim=config.HIDDEN_DIM,
    output_dim=1,
    num_layers=3
).to(device)

adj_matrix = adj_matrix.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {total_params:,} parameters")

# STEP 7: Training Setup
print("\n🎯 Training setup...")

from training.trainer import CombinedDengueLoss

criterion = CombinedDengueLoss(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
patience = 50

print(f"   Optimizer: Adam (LR={config.LEARNING_RATE})")
print(f"   Scheduler: ReduceLROnPlateau")
print(f"   Early stopping patience: {patience}")

# STEP 8: Training Loop
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)

history = {'train_loss': [], 'val_loss': []}

for epoch in range(config.EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    for batch_features, batch_targets in train_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features, adj_matrix)
        loss = criterion(outputs, batch_targets)
        
        if torch.isnan(loss):
            print(f"⚠️  NaN loss at epoch {epoch+1}, skipping batch")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_batches += 1
    
    if train_batches == 0:
        print(f"❌ All training batches failed at epoch {epoch+1}")
        break
    
    avg_train_loss = train_loss / train_batches
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_features, adj_matrix)
            loss = criterion(outputs, batch_targets)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                val_loss += loss.item()
                val_batches += 1
    
    if val_batches == 0:
        print(f"⚠️  All validation batches failed at epoch {epoch+1}")
        avg_val_loss = float('inf')
    else:
        avg_val_loss = val_loss / val_batches
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        marker = "⭐"
    else:
        patience_counter += 1
        marker = ""
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f} {marker}")
    
    if patience_counter >= patience:
        print(f"\n⏹️  Early stopping at epoch {epoch+1}")
        break

# Load best model
if best_val_loss < float('inf'):
    model.load_state_dict(best_model_state)
    print(f"✅ Loaded best model (val_loss={best_val_loss:.4f})")

# STEP 9: Evaluate
print("\n📊 Evaluating on test set...")

model.eval()
all_preds = []
all_actuals = []

with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        batch_features = batch_features.to(device)
        outputs = model(batch_features, adj_matrix)
        
        preds = outputs['predictions'].cpu().numpy()
        actuals = batch_targets.cpu().numpy()
        
        # Inverse transform
        if metadata.get('target_transform') == 'log1p':
            preds = np.expm1(np.maximum(preds, 0))
            actuals = np.expm1(actuals)
        
        preds = np.maximum(preds, 0)
        
        all_preds.extend(preds.flatten())
        all_actuals.extend(actuals.flatten())

all_preds = np.array(all_preds)
all_actuals = np.array(all_actuals)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(all_actuals, all_preds)
rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
r2 = r2_score(all_actuals, all_preds)

print("\n" + "="*80)
print("📊 FINAL TEST RESULTS")
print("="*80)
print(f"   MAE:  {mae:.3f} cases/week")
print(f"   RMSE: {rmse:.3f} cases/week")
print(f"   R²:   {r2:.3f}")
print(f"   Pred range: [{all_preds.min():.1f}, {all_preds.max():.1f}]")
print(f"   Actual range: [{all_actuals.min():.1f}, {all_actuals.max():.1f}]")
print("="*80)

# Save model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'metadata': metadata,
    'history': history,
    'test_metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
}

torch.save(checkpoint, 'dengue_stgnn_FIXED.pth')
print(f"\n✅ Model saved: dengue_stgnn_FIXED.pth")

if r2 > 0:
    print(f"\n🎉 SUCCESS! R² = {r2:.3f} (positive, model is learning!)")
else:
    print(f"\n⚠️  R² still negative. Try:")
    print(f"   1. More epochs (currently {epoch+1})")
    print(f"   2. Different learning rate")
    print(f"   3. Simpler model architecture")