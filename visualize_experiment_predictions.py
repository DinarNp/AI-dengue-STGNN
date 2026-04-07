#!/usr/bin/env python3
"""
Visualize Experiment Predictions by Kabupaten for 2025

This script loads a trained model and generates detailed predictions
for each kabupaten in the 2025 test period.

Usage:
    Just change the EXPERIMENT_DIR variable to point to your experiment folder.
    
    EXPERIMENT_DIR = "experiment_results/experiments1"  # Weekly SKDR 5-kab
    EXPERIMENT_DIR = "experiment_results/experiments3"  # Monthly SKDR 5-kab
    etc.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from models.stgnn import STGNNDenguePredictor
from data.preprocessor import DengueDataPreprocessor
from data.dataset import DengueDataset, collate_fn
from torch.utils.data import DataLoader

# ============================================================================
# CONFIGURATION - CHANGE THIS TO YOUR EXPERIMENT FOLDER
# ============================================================================

EXPERIMENT_DIR = "experiment_results/experiments8"  # 👈 CHANGE THIS

# ============================================================================
# CONSTANTS
# ============================================================================

MONTHLY_THRESHOLDS = {
    'low': 30,      # 0-30 cases = Low Risk
    'medium': 80,   # 31-80 cases = Medium Risk  
    'high': 80      # >80 cases = High Risk
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_experiment_config(exp_dir):
    """Load experiment configuration from JSON"""
    json_files = list(Path(exp_dir).glob('*_results.json'))
    if not json_files:
        raise FileNotFoundError(f"No results.json found in {exp_dir}")
    
    with open(json_files[0], 'r') as f:
        return json.load(f)


def load_model_checkpoint(exp_dir):
    """Load trained model from checkpoint"""
    model_path = os.path.join(exp_dir, 'dengue_stgnn_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint


def prepare_test_data(combined_data_path, exp_config):
    """Load and prepare test data"""
    print(f"Loading test data from {combined_data_path}...")
    df = pd.read_csv(combined_data_path)
    
    # Filter only test split
    test_df = df[df['split'] == 'test'].copy()
    print(f"Test data shape: {test_df.shape}")
    print(f"Test period: {test_df['Year'].min()}-{test_df['Year'].max()}")
    print(f"Regions: {sorted(test_df['Region'].unique())}")
    
    return test_df


def reconstruct_model_and_generate_predictions(checkpoint, test_df, exp_config):
    """
    Reconstruct the model and generate predictions for test data
    
    Returns:
        predictions_by_region: Dict mapping region -> DataFrame with predictions
    """
    print("\n" + "="*80)
    print("RECONSTRUCTING MODEL AND GENERATING PREDICTIONS")
    print("="*80)
    
    # Get data type from exp_config
    data_type = exp_config['config']['type']
    
    # Get config from checkpoint
    config = Config()
    metadata = checkpoint['metadata']
    model_config = exp_config['model_config']
    
    # Set config parameters
    config.HIDDEN_DIM = model_config['HIDDEN_DIM']
    config.NUM_LAYERS = model_config['NUM_LAYERS']
    config.NUM_HEADS = model_config['NUM_HEADS']
    config.WINDOW_SIZE = model_config['WINDOW_SIZE']
    
    # Preprocess data
    print("\n1. Preprocessing data...")
    preprocessor = DengueDataPreprocessor(config)
    
    # Combine train and test for preprocessing (to get same scaling)
    features, targets, prep_metadata = preprocessor.preprocess_data(test_df)
    
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Nodes: {prep_metadata['n_nodes']}")
    
    # Build graph
    print("\n2. Building spatial graph...")
    from models.graph_constructor import GraphConstructor
    location_coords = prep_metadata['location_coords']
    graph_constructor = GraphConstructor(config)
    adj_matrix = torch.tensor(
        graph_constructor.build_spatial_adjacency(location_coords), 
        dtype=torch.float32
    )
    print(f"   Graph nodes: {adj_matrix.shape[0]}")
    
    # Initialize model
    print("\n3. Initializing model...")
    input_dim = len(prep_metadata['feature_cols'])
    model = STGNNDenguePredictor(
        config=config,
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=1,
        num_layers=config.NUM_LAYERS
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✅ Model loaded successfully")
    
    # Create dataset and dataloader
    print("\n4. Creating dataset...")
    dataset = DengueDataset(
        features=features,
        targets=targets,
        metadata=prep_metadata,
        window_size=config.WINDOW_SIZE,
        forecast_horizon=1
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"   Total sequences: {len(dataset)}")
    
    # Generate predictions
    print("\n5. Generating predictions...")
    
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_features, batch_targets in dataloader:
            outputs = model(batch_features, adj_matrix)
            predictions = outputs['predictions'].cpu().numpy()
            targets_batch = batch_targets.cpu().numpy()
            
            all_predictions.extend(predictions.flatten())
            all_actuals.extend(targets_batch.flatten())
    
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    # Apply inverse transform if needed
    if prep_metadata.get('target_transform') == 'log1p':
        all_predictions = np.expm1(np.maximum(all_predictions, 0))
        all_actuals = np.expm1(all_actuals)
        print("   ✅ Applied inverse log1p transform")
    
    # Ensure non-negative
    all_predictions = np.maximum(all_predictions, 0)
    
    print(f"   Total predictions: {len(all_predictions)}")
    print(f"   Prediction range: [{all_predictions.min():.2f}, {all_predictions.max():.2f}]")
    print(f"   Actual range: [{all_actuals.min():.2f}, {all_actuals.max():.2f}]")
    
    # Map predictions back to regions and time periods
    print("\n6. Mapping predictions to kabupaten...")
    
    # Since we use sequences, we need to align predictions with original data
    # The dataset creates sequences, so first WINDOW_SIZE-1 samples don't have predictions
    window_size = config.WINDOW_SIZE
    n_nodes = prep_metadata['n_nodes']
    
    # Calculate how many timesteps we have predictions for
    n_predictions_per_node = len(all_predictions) // n_nodes
    
    # Get unique regions and time info
    regions = sorted(test_df['Region'].unique())
    # Determine temporal column based on data type
    if data_type == 'monthly':
        temporal_col = 'Month'
    else:
        temporal_col = 'Week'
    
    # Create prediction DataFrame for each region
    predictions_by_region = {}
    
    for idx, region in enumerate(regions):
        region_data = test_df[test_df['Region'] == region].sort_values(temporal_col).copy()
        
        # Extract predictions for this region (every n_nodes-th prediction starting from idx)
        region_preds = all_predictions[idx::n_nodes]
        region_actuals = all_actuals[idx::n_nodes]
        
        # We can only predict for sequences after the first WINDOW_SIZE-1 samples
        # So skip those in the alignment
        if len(region_preds) < len(region_data):
            # Add NaN for the first few timesteps that don't have predictions
            n_skip = len(region_data) - len(region_preds)
            region_preds = np.concatenate([np.full(n_skip, np.nan), region_preds])
            region_actuals = np.concatenate([np.full(n_skip, np.nan), region_actuals])
        elif len(region_preds) > len(region_data):
            # Truncate if we have too many predictions
            region_preds = region_preds[:len(region_data)]
            region_actuals = region_actuals[:len(region_data)]
        
        # Add to dataframe
        region_data['Predicted'] = region_preds
        region_data['Actual_FromModel'] = region_actuals
        region_data['Error'] = region_data['Actual_FromModel'] - region_data['Predicted']
        region_data['Abs_Error'] = np.abs(region_data['Error'])
        
        predictions_by_region[region] = region_data
        
        # Calculate metrics for this region (excluding NaN)
        valid_mask = ~np.isnan(region_data['Predicted'])
        if valid_mask.sum() > 0:
            mae = np.nanmean(region_data['Abs_Error'])
            rmse = np.sqrt(np.nanmean(region_data['Error']**2))
            print(f"   {region}: MAE={mae:.2f}, RMSE={rmse:.2f} ({valid_mask.sum()} predictions)")
    
    return predictions_by_region, exp_config['config']['type']


def plot_predictions_by_kabupaten(predictions_by_region, exp_config, output_dir):
    """
    Create detailed visualization of predictions for each kabupaten
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    regions = sorted(predictions_by_region.keys())
    n_regions = len(regions)
    data_type = exp_config['config']['type']
    # Determine temporal column based on data type
    temporal_col = 'Month' if data_type == 'monthly' else 'Week'
    target_col = exp_config['config']['target_col']
    exp_name = exp_config['experiment_name']
    
    # Create figure with subplots (one per region)
    fig, axes = plt.subplots(n_regions, 1, figsize=(16, 4.5*n_regions))
    
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'{exp_config["config"]["train_data"].split("/")[-1]} - 2025 Predictions by Kabupaten', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    for idx, (region, ax) in enumerate(zip(regions, axes)):
        region_data = predictions_by_region[region]
        
        # Use ALL data (including NaN predictions for window period)
        full_data = region_data.sort_values(temporal_col)
        
        if len(full_data) == 0:
            ax.text(0.5, 0.5, f'{region}\nNo data available', 
                   ha='center', va='center', fontsize=14)
            continue
        
        # Get all time points
        x_vals_all = full_data[temporal_col].values
        actual_vals_all = full_data[target_col].values
        pred_vals_all = full_data['Predicted'].values
        
        # Separate window period (NaN predictions) from prediction period
        has_prediction = ~np.isnan(pred_vals_all)
        
        # Plot FULL ACTUAL LINE (entire 2025)
        ax.plot(x_vals_all, actual_vals_all, 'o-', color='#d62728', linewidth=2.5, 
                markersize=7, label='Actual Cases', alpha=0.85, markeredgecolor='white', markeredgewidth=0.5)
        
        # Plot PREDICTIONS (only where available)
        if has_prediction.any():
            x_vals_pred = x_vals_all[has_prediction]
            pred_vals = pred_vals_all[has_prediction]
            ax.plot(x_vals_pred, pred_vals, 's--', color='#1f77b4', linewidth=2.5, 
                    markersize=7, label='Predicted Cases', alpha=0.85, markeredgecolor='white', markeredgewidth=0.5)
        
        # Shade the WINDOW PERIOD (where predictions are not available)
        if (~has_prediction).any():
            window_indices = np.where(~has_prediction)[0]
            if len(window_indices) > 0:
                window_start = x_vals_all[window_indices[0]]
                window_end = x_vals_all[window_indices[-1]]
                ax.axvspan(window_start - 0.5, window_end + 0.5, alpha=0.15, color='gray', 
                          label=f'Window Period (No Predictions)', zorder=0)
        
        # Add threshold lines for monthly data
        if data_type == 'monthly':
            ax.axhline(y=MONTHLY_THRESHOLDS['low'], color='green', linestyle=':', 
                      alpha=0.6, linewidth=2, label=f"Low Risk Threshold ({MONTHLY_THRESHOLDS['low']} cases)")
            ax.axhline(y=MONTHLY_THRESHOLDS['high'], color='orange', linestyle=':', 
                      alpha=0.6, linewidth=2, label=f"High Risk Threshold ({MONTHLY_THRESHOLDS['high']} cases)")
            
            # Shade risk zones (only for prediction period)
            if has_prediction.any():
                x_pred_min, x_pred_max = x_vals_pred.min(), x_vals_pred.max()
                ax.fill_between([x_pred_min, x_pred_max], 0, MONTHLY_THRESHOLDS['low'], 
                              alpha=0.1, color='green', label='Low Risk Zone')
                ax.fill_between([x_pred_min, x_pred_max], MONTHLY_THRESHOLDS['low'], MONTHLY_THRESHOLDS['high'], 
                              alpha=0.1, color='yellow', label='Medium Risk Zone')
        
        # Calculate metrics (only for prediction period)
        if has_prediction.any():
            prediction_data = full_data[has_prediction]
            mae = np.nanmean(prediction_data['Abs_Error'])
            rmse = np.sqrt(np.nanmean(prediction_data['Error']**2))
            actual_pred_period = actual_vals_all[has_prediction]
            r2 = 1 - (np.nansum(prediction_data['Error']**2) / np.nansum((actual_pred_period - actual_pred_period.mean())**2))
            
            # Add text box with metrics
            n_window = (~has_prediction).sum()
            window_label = f"Window: {n_window} {temporal_col.lower()}s\n" if n_window > 0 else ""
            textstr = f'{window_label}MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Highlight prediction errors > threshold
            large_errors = prediction_data[prediction_data['Abs_Error'] > mae * 2]
            if len(large_errors) > 0:
                for _, row in large_errors.iterrows():
                    ax.scatter(row[temporal_col], row['Predicted'], 
                             s=150, facecolors='none', edgecolors='red', linewidths=2, alpha=0.7)
        
        # Styling
        ax.set_title(f'{region}', fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel('Dengue Cases', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.set_ylim(bottom=0)
    
    axes[-1].set_xlabel(temporal_col, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'{exp_name}_predictions_by_kabupaten.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    plt.close()
    
    # Also create a comparison summary plot
    create_summary_plot(predictions_by_region, exp_config, output_dir)


def create_summary_plot(predictions_by_region, exp_config, output_dir):
    """Create summary comparison plot"""
    print("\nCreating summary comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    exp_name = exp_config['experiment_name']
    data_type = exp_config['config']['type']
    temporal_col = 'Month' if data_type == 'monthly' else 'Week'
    target_col = exp_config['config']['target_col']
    
    # Plot 1: Scatter plot (all regions combined)
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(predictions_by_region)))
    
    for idx, (region, region_data) in enumerate(predictions_by_region.items()):
        valid_data = region_data.dropna(subset=['Predicted'])
        if len(valid_data) > 0:
            ax.scatter(valid_data[target_col], valid_data['Predicted'], 
                      alpha=0.6, s=60, label=region, color=colors[idx])
    
    # Add diagonal line
    all_actuals = np.concatenate([df[target_col].dropna().values for df in predictions_by_region.values()])
    all_preds = np.concatenate([df['Predicted'].dropna().values for df in predictions_by_region.values()])
    max_val = max(all_actuals.max(), all_preds.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    ax.set_xlabel('Actual Cases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Cases', fontsize=12, fontweight='bold')
    ax.set_title('Predictions vs Actual (All Kabupaten)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    # Plot 2: MAE by region
    ax = axes[1]
    mae_by_region = {}
    for region, region_data in predictions_by_region.items():
        valid_data = region_data.dropna(subset=['Predicted'])
        if len(valid_data) > 0:
            mae_by_region[region] = np.nanmean(valid_data['Abs_Error'])
    
    regions = list(mae_by_region.keys())
    maes = list(mae_by_region.values())
    
    bars = ax.barh(regions, maes, color=colors[:len(regions)], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Kabupaten', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        ax.text(mae + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{mae:.2f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{exp_name}_summary_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Summary plot saved to: {output_file}")
    plt.close()


def save_predictions_csv(predictions_by_region, output_dir, exp_name):
    """Save predictions to CSV files"""
    print("\nSaving prediction CSVs...")
    
    for region, region_data in predictions_by_region.items():
        valid_data = region_data.dropna(subset=['Predicted'])
        if len(valid_data) > 0:
            # Save region-specific file
            region_name = region.replace(' ', '_').replace('/', '_')
            output_file = os.path.join(output_dir, f'{exp_name}_{region_name}_predictions.csv')
            valid_data.to_csv(output_file, index=False)
            print(f"   ✅ {region}: {output_file}")
    
    # Also save combined file
    combined_df = pd.concat([df.dropna(subset=['Predicted']) for df in predictions_by_region.values()])
    combined_file = os.path.join(output_dir, f'{exp_name}_all_regions_predictions.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"   ✅ Combined: {combined_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("DENGUE PREDICTION VISUALIZATION BY KABUPATEN")
    print("="*80)
    print(f"\nExperiment Directory: {EXPERIMENT_DIR}")
    
    # Check if directory exists
    if not os.path.exists(EXPERIMENT_DIR):
        print(f"\n❌ ERROR: Directory not found: {EXPERIMENT_DIR}")
        print("\nPlease update the EXPERIMENT_DIR variable at the top of this script.")
        return
    
    # Load experiment configuration
    print("\n1. Loading experiment configuration...")
    exp_config = load_experiment_config(EXPERIMENT_DIR)
    print(f"   Experiment: {exp_config['config']['train_data']}")
    print(f"   Type: {exp_config['config']['type']}")
    print(f"   Target: {exp_config['config']['target_col']}")
    print(f"   Regions: {exp_config['config']['n_kabupaten']}")
    
    # Load model checkpoint
    print("\n2. Loading model checkpoint...")
    checkpoint = load_model_checkpoint(EXPERIMENT_DIR)
    print(f"   Model parameters: {sum(p.numel() for p in checkpoint['model_state_dict'].values()):,}")
    
    # Load test data
    print("\n3. Loading test data...")
    combined_data_files = list(Path(EXPERIMENT_DIR).glob('*_combined_data.csv'))
    if not combined_data_files:
        print("   ❌ ERROR: No combined_data.csv found")
        return
    
    test_df = prepare_test_data(combined_data_files[0], exp_config)
    
    # Generate predictions
    print("\n4. Generating predictions by kabupaten...")
    predictions_by_region, data_type = reconstruct_model_and_generate_predictions(
        checkpoint, test_df, exp_config
    )
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    plot_predictions_by_kabupaten(predictions_by_region, exp_config, EXPERIMENT_DIR)
    
    # Save predictions to CSV
    print("\n6. Saving predictions...")
    save_predictions_csv(predictions_by_region, EXPERIMENT_DIR, exp_config['experiment_name'])
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput files saved in: {EXPERIMENT_DIR}/")
    print("  - *_predictions_by_kabupaten.png (detailed by region)")
    print("  - *_summary_comparison.png (overall comparison)")
    print("  - *_predictions.csv (prediction data)")


if __name__ == "__main__":
    main()
