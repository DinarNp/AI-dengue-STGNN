#!/usr/bin/env python3
"""
Comprehensive Model Comparison Experiment
==========================================

Compares dengue prediction models across 8 scenarios:
Weekly vs Monthly, NDVI types, 3 vs 5 kabupaten, DINKES vs SKDR

Training: 2021-2024 (train + validation split internally)
Testing: 2025 (held-out test set)

Date: 2026-03-10
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENTS = {
    'weekly_skdr_5': {
        'name': 'Weekly SKDR Cases (5 Kabupaten)',
        'train_data': 'data/fix/data_weekly_5kab_2021_2024.csv',
        'test_data': 'data/fix/data_weekly_5kab_2025.csv',
        'target_col': 'Cases',
        'temporal_col': 'Week',
        'type': 'weekly',
        'has_real_ndvi': False,
        'n_kabupaten': 5
    },
    'weekly_ndvi_5': {
        'name': 'Weekly with Real NDVI (5 Kabupaten)',
        'train_data': 'data/fix/data_weekly_5kab_2021_2024_ndvi.csv',
        'test_data': 'data/fix/data_weekly_5kab_2025_ndvi.csv',
        'target_col': 'Cases',
        'temporal_col': 'Week',
        'type': 'weekly',
        'has_real_ndvi': True,
        'n_kabupaten': 5
    },
    'monthly_skdr_5': {
        'name': 'Monthly SKDR Cases (5 Kabupaten)',
        'train_data': 'data/fix/data_monthly_5kab_2021_2024.csv',
        'test_data': 'data/fix/data_monthly_5kab_2025.csv',
        'target_col': 'Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': False,
        'n_kabupaten': 5
    },
    'monthly_ndvi_5': {
        'name': 'Monthly with Real NDVI (5 Kabupaten)',
        'train_data': 'data/fix/data_monthly_5kab_2021_2024_ndvi.csv',
        'test_data': 'data/fix/data_monthly_5kab_2025_ndvi.csv',
        'target_col': 'Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': True,
        'n_kabupaten': 5
    },
    'monthly_skdr_3': {
        'name': 'Monthly SKDR Cases (3 Kabupaten)',
        'train_data': 'data/fix/data_monthly_3kab_2021_2024.csv',
        'test_data': 'data/fix/data_monthly_3kab_2025.csv',
        'target_col': 'Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': False,
        'n_kabupaten': 3
    },
    'monthly_ndvi_3': {
        'name': 'Monthly with Real NDVI (3 Kabupaten)',
        'train_data': 'data/fix/data_monthly_3kab_2021_2024_ndvi.csv',
        'test_data': 'data/fix/data_monthly_3kab_2025_ndvi.csv',
        'target_col': 'Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': True,
        'n_kabupaten': 3
    },
    'monthly_dinkes_3': {
        'name': 'Monthly DINKES Cases (3 Kabupaten)',
        'train_data': 'data/fix/data_monthly_3kab_2021_2024_DINKES.csv',
        'test_data': 'data/fix/data_monthly_3kab_2025_DINKES.csv',
        'target_col': 'Dinkes_Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': False,
        'n_kabupaten': 3
    },
    'monthly_dinkes_ndvi_3': {
        'name': 'Monthly DINKES + Real NDVI (3 Kabupaten)',
        'train_data': 'data/fix/data_monthly_3kab_2021_2024_ndvi_DINKES.csv',
        'test_data': 'data/fix/data_monthly_3kab_2025_ndvi_DINKES.csv',
        'target_col': 'Dinkes_Cases',
        'temporal_col': 'Month',
        'type': 'monthly',
        'has_real_ndvi': True,
        'n_kabupaten': 3
    }
}

# Output directory
OUTPUT_DIR = 'experiment_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classification thresholds for monthly cases (UPDATED)
MONTHLY_THRESHOLDS = {
    'low': 30,      # 0-30 cases = Low risk
    'medium': 80,   # 31-80 cases = Medium risk  
    'high': 80      # >80 cases = High risk
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_combined_dataset(train_path, test_path, target_col):
    """
    Combine training and test datasets with proper split markers
    """
    print(f"  Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    train_df.columns = train_df.columns.str.strip()
    
    print(f"  Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path, encoding='utf-8-sig')
    test_df.columns = test_df.columns.str.strip()
    
    # Ensure target column exists and rename if needed
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data. Available: {train_df.columns.tolist()}")
    if target_col != 'Cases':
        train_df['Cases'] = train_df[target_col]
        test_df['Cases'] = test_df[target_col]
    
    # Add split marker
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    
    # Combine
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"  ✅ Combined dataset shape: {combined_df.shape}")
    print(f"     Training samples: {len(train_df)} (2021-2024)")
    print(f"     Test samples: {len(test_df)} (2025)")
    
    return combined_df, test_df


def extract_metrics_from_result(metrics_dict):
    """
    Extract and normalize metrics from different possible formats
    
    Args:
        metrics_dict: Dictionary containing metrics
        
    Returns:
        Normalized metrics dict with standard keys
    """
    if not metrics_dict:
        return None
    
    # Try to extract test metrics from various possible keys
    test_mae = (metrics_dict.get('test_mae') or 
                metrics_dict.get('mae') or 
                metrics_dict.get('MAE'))
    
    test_rmse = (metrics_dict.get('test_rmse') or 
                 metrics_dict.get('rmse') or 
                 metrics_dict.get('RMSE'))
    
    test_r2 = (metrics_dict.get('test_r2') or 
               metrics_dict.get('r2') or 
               metrics_dict.get('R2'))
    
    test_mape = (metrics_dict.get('test_mape') or 
                 metrics_dict.get('mape') or 
                 metrics_dict.get('MAPE'))
    
    # Extract training metrics
    train_mae = (metrics_dict.get('train_mae') or 
                 metrics_dict.get('train_loss'))
    
    val_mae = metrics_dict.get('val_mae') or metrics_dict.get('val_loss')
    
    return {
        'test_mae': float(test_mae) if test_mae is not None else None,
        'test_rmse': float(test_rmse) if test_rmse is not None else None,
        'test_r2': float(test_r2) if test_r2 is not None else None,
        'test_mape': float(test_mape) if test_mape is not None else None,
        'train_mae': float(train_mae) if train_mae is not None else None,
        'val_mae': float(val_mae) if val_mae is not None else None
    }


def save_experiment_results(exp_name, results, metrics, config_used, predictions=None, actuals=None):
    """Save experiment results to JSON"""
    output = {
        'experiment_name': exp_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'train_data': EXPERIMENTS[exp_name]['train_data'],
            'test_data': EXPERIMENTS[exp_name]['test_data'],
            'target_col': EXPERIMENTS[exp_name]['target_col'],
            'type': EXPERIMENTS[exp_name]['type'],
            'has_real_ndvi': EXPERIMENTS[exp_name]['has_real_ndvi'],
            'n_kabupaten': EXPERIMENTS[exp_name]['n_kabupaten']
        },
        'model_config': {
            'HIDDEN_DIM': config_used.HIDDEN_DIM,
            'NUM_LAYERS': config_used.NUM_LAYERS,
            'NUM_HEADS': config_used.NUM_HEADS,
            'LEARNING_RATE': config_used.LEARNING_RATE,
            'NUM_EPOCHS': config_used.NUM_EPOCHS,
            'BATCH_SIZE': config_used.BATCH_SIZE,
            'WINDOW_SIZE': getattr(config_used, 'WINDOW_SIZE_MONTHLY', config_used.WINDOW_SIZE)
                           if EXPERIMENTS[exp_name]['type'] == 'monthly' else config_used.WINDOW_SIZE,
        },
        'results': results,
        'metrics': metrics
    }
    
    output_file = os.path.join(OUTPUT_DIR, f'{exp_name}_results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save predictions if available
    if predictions is not None and actuals is not None:
        # Create a simple dataframe with predictions and actuals
        # Note: Length may differ from test_df due to sequence windowing
        pred_df = pd.DataFrame({
            'Predicted_Cases': predictions,
            'Actual_Cases': actuals,
            'Error': np.array(actuals) - np.array(predictions),
            'Abs_Error': np.abs(np.array(actuals) - np.array(predictions)),
            'Pct_Error': ((np.array(actuals) - np.array(predictions)) / (np.array(actuals) + 1)) * 100
        })
        
        pred_file = os.path.join(OUTPUT_DIR, f'{exp_name}_predictions.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f"  ✅ Predictions saved to {pred_file} ({len(pred_df)} samples)")
    
    print(f"  ✅ Results saved to {output_file}")
    return output_file


def plot_predictions_vs_actual(exp_name, predictions, actuals, exp_config):
    """
    Create visualization comparing predictions vs actual values for 2025
    
    Args:
        exp_name: Experiment name
        predictions: Predicted values (flattened array)
        actuals: Actual values (flattened array)
        exp_config: Experiment configuration
    """
    print(f"  📊 Creating prediction visualization...")
    
    # Create simple scatter plot and time series
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fig.suptitle(f'{exp_config["name"]} - 2025 Predictions vs Actual', 
                 fontsize=14, fontweight='bold')
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Subplot 1: Scatter plot
    ax = axes[0]
    ax.scatter(actuals, predictions, alpha=0.6, s=50, color='#1f77b4')
    
    # Add diagonal line (perfect prediction)
    max_val = max(actuals.max(), predictions.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    # Add threshold lines for monthly data
    if exp_config['type'] == 'monthly':
        ax.axvline(x=MONTHLY_THRESHOLDS['low'], color='green', linestyle=':', 
                  alpha=0.5, linewidth=1.5, label=f"Low Threshold")
        ax.axhline(y=MONTHLY_THRESHOLDS['low'], color='green', linestyle=':', 
                  alpha=0.5, linewidth=1.5)
        ax.axvline(x=MONTHLY_THRESHOLDS['high'], color='orange', linestyle=':', 
                  alpha=0.5, linewidth=1.5, label=f"High Threshold")
        ax.axhline(y=MONTHLY_THRESHOLDS['high'], color='orange', linestyle=':', 
                  alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Actual Cases', fontsize=11)
    ax.set_ylabel('Predicted Cases', fontsize=11)
    ax.set_title('Predictions vs Actual (Scatter)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    # Subplot 2: Time series
    ax = axes[1]
    x_vals = np.arange(len(predictions))
    ax.plot(x_vals, actuals, 'o-', color='#d62728', linewidth=2, 
            markersize=5, label='Actual', alpha=0.8)
    ax.plot(x_vals, predictions, 's--', color='#1f77b4', linewidth=2, 
            markersize=5, label='Predicted', alpha=0.8)
    
    # Add threshold lines for monthly data
    if exp_config['type'] == 'monthly':
        ax.axhline(y=MONTHLY_THRESHOLDS['low'], color='green', linestyle=':', 
                  alpha=0.5, linewidth=1.5, label=f"Low (<{MONTHLY_THRESHOLDS['low']})")
        ax.axhline(y=MONTHLY_THRESHOLDS['high'], color='orange', linestyle=':', 
                  alpha=0.5, linewidth=1.5, label=f"High (>{MONTHLY_THRESHOLDS['high']})")
    
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Cases', fontsize=11)
    ax.set_title('Predictions vs Actual (Time Series)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    plot_file = os.path.join(OUTPUT_DIR, f'{exp_name}_prediction_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ Prediction plot saved to {plot_file}")
    plt.close()


def run_single_experiment(exp_name, exp_config):
    """
    Run a single experiment
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_config['name']}")
    print(f"{'='*80}")
    
    # Prepare combined dataset
    combined_df, test_df = prepare_combined_dataset(
        exp_config['train_data'],
        exp_config['test_data'],
        exp_config['target_col']
    )
    
    # Save combined dataset temporarily
    temp_data_file = os.path.join(OUTPUT_DIR, f'{exp_name}_combined_data.csv')
    combined_df.to_csv(temp_data_file, index=False)
    print(f"  Saved combined dataset to {temp_data_file}")
    
    # Configure the model
    config = Config()
    
    # Adjust config based on experiment type
    if exp_config['type'] == 'monthly':
        config.WINDOW_SIZE = getattr(config, 'WINDOW_SIZE_MONTHLY', 4)
        print(f"  Using monthly window size: {config.WINDOW_SIZE}")
    else:
        config.WINDOW_SIZE = 8
        print(f"  Using weekly window size: {config.WINDOW_SIZE}")
    
    # Reduce epochs for faster experimentation
    config.NUM_EPOCHS = 100
    config.EARLY_STOPPING_PATIENCE = 20
    
    # Initialize prediction system
    system = DenguePredictionSystem(config)
    
    # Run the pipeline
    print(f"\n  🚀 Starting training...")
    try:
        model, metrics, metadata, history = system.run_complete_pipeline(
            data_path=temp_data_file,
            generate_paper_analysis=False
        )
        
        print(f"\n  ✅ Training completed!")
        
        # Extract normalized metrics
        results = extract_metrics_from_result(metrics)
        
        if not results:
            print("  ⚠️ Warning: Could not extract metrics properly")
            results = {
                'test_mae': None, 'test_rmse': None, 'test_r2': None,
                'test_mape': None, 'train_mae': None, 'val_mae': None
            }
        
        # Add history info
        if history:
            results['best_epoch'] = history.get('best_epoch')
            results['train_mae'] = history.get('final_train_mae')
            # Get the best validation MAE (minimum across all epochs)
            if 'val_mae' in history and history['val_mae']:
                results['val_mae'] = min(history['val_mae'])
        results['training_time'] = metadata.get('training_time') if metadata else None
        
        # Get predictions for plotting
        predictions = metadata.get('test_predictions')
        actuals = metadata.get('test_actuals')
        
        # Save results with predictions
        save_experiment_results(exp_name, results, metrics, config, 
                              predictions, actuals)
        
        # Create prediction visualization
        if predictions is not None and actuals is not None:
            plot_predictions_vs_actual(exp_name, predictions, actuals, exp_config)
        else:
            print("  ⚠️ Predictions not available for plotting")
        
        return results, metrics, True
        
    except Exception as e:
        print(f"\n  ❌ ERROR in experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False


def compare_results(all_results):
    """
    Create comparison table and visualizations
    """
    print(f"\n{'='*80}")
    print("COMPARISON OF ALL EXPERIMENTS")
    print(f"{'='*80}\n")
    
    # Create comparison dataframe
    comparison_data = []
    for exp_name, (results, metrics, success) in all_results.items():
        if success and results:
            exp_info = EXPERIMENTS[exp_name]
            comparison_data.append({
                'Experiment': exp_info['name'],
                'Type': exp_info['type'].capitalize(),
                'NDVI': 'Real' if exp_info['has_real_ndvi'] else 'Estimated',
                'Kabupaten': exp_info['n_kabupaten'],
                'Test MAE': f"{results['test_mae']:.2f}" if results.get('test_mae') else 'N/A',
                'Test RMSE': f"{results['test_rmse']:.2f}" if results.get('test_rmse') else 'N/A',
                'Test R²': f"{results['test_r2']:.4f}" if results.get('test_r2') else 'N/A',
                'Test MAPE': f"{results['test_mape']:.2f}%" if results.get('test_mape') else 'N/A',
                'Train MAE': f"{results['train_mae']:.2f}" if results.get('train_mae') else 'N/A',
                'Val MAE': f"{results['val_mae']:.2f}" if results.get('val_mae') else 'N/A',
                'Best Epoch': results.get('best_epoch', 'N/A')
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print table
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_file = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✅ Comparison table saved to {comparison_file}")
    
    # Determine best model
    if len(comparison_data) > 0:
        valid_mae = [d for d in comparison_data if d['Test MAE'] != 'N/A']
        if valid_mae:
            mae_values = [float(d['Test MAE']) for d in valid_mae]
            best_mae_idx = mae_values.index(min(mae_values))
            best_model = valid_mae[best_mae_idx]
            
            print(f"\n{'='*80}")
            print("🏆 BEST MODEL (by Test MAE)")
            print(f"{'='*80}")
            print(f"Experiment: {best_model['Experiment']}")
            print(f"Type: {best_model['Type']}")
            print(f"NDVI: {best_model['NDVI']}")
            print(f"Test MAE: {best_model['Test MAE']}")
            print(f"Test RMSE: {best_model['Test RMSE']}")
            print(f"Test R²: {best_model['Test R²']}")
            print(f"{'='*80}")


def main():
    """Main execution function"""
    print("="*80)
    print("DENGUE PREDICTION MODEL COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Training Period: 2021-2024")
    print(f"Test Period: 2025")
    print(f"Number of Experiments: {len(EXPERIMENTS)}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nMonthly Case Thresholds:")
    print(f"  Low Risk: 0-{MONTHLY_THRESHOLDS['low']} cases")
    print(f"  Medium Risk: {MONTHLY_THRESHOLDS['low']+1}-{MONTHLY_THRESHOLDS['medium']} cases")
    print(f"  High Risk: >{MONTHLY_THRESHOLDS['high']} cases")
    print("="*80)
    
    # Ask user which experiments to run
    print("\nAvailable Experiments:")
    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items(), 1):
        print(f"{i}. {exp_config['name']}")
    
    print("\nOptions:")
    print("  Enter 'all' to run all experiments")
    print("  Enter numbers separated by commas (e.g., '1,3,4') to run specific experiments")
    print("  Press Enter to run all")
    
    user_input = input("\nYour choice: ").strip()
    
    if user_input == '' or user_input.lower() == 'all':
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            exp_names = list(EXPERIMENTS.keys())
            experiments_to_run = [exp_names[i-1] for i in indices if 1 <= i <= len(exp_names)]
        except:
            print("Invalid input. Running all experiments.")
            experiments_to_run = list(EXPERIMENTS.keys())
    
    print(f"\n✅ Will run {len(experiments_to_run)} experiment(s)")
    
    # Run experiments
    all_results = {}
    for exp_name in experiments_to_run:
        exp_config = EXPERIMENTS[exp_name]
        results, metrics, success = run_single_experiment(exp_name, exp_config)
        all_results[exp_name] = (results, metrics, success)
    
    # Compare results
    compare_results(all_results)
    
    print(f"\n{'='*80}")
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"\n📂 All results saved in: {OUTPUT_DIR}/")
    print("Files generated:")
    print("  - <experiment>_results.json (detailed results)")
    print("  - <experiment>_predictions.csv (predictions + actual values)")
    print("  - <experiment>_prediction_plot.png (visualization)")
    print("  - comparison_results.csv (comparison table)")
    print("\n🎉 You can now analyze the results!")
    print("="*80)


if __name__ == "__main__":
    main()
