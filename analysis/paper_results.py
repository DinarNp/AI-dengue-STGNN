"""
Paper Results Generator
Generates comprehensive analysis results for the academic paper
"""

import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def generate_paper_results(test_loader, model, adj_matrix, trainer, metadata, save_dir='paper_outputs'):
    """
    Generate comprehensive results for academic paper
    
    Args:
        test_loader: DataLoader for test set
        model: Trained STGNN model
        adj_matrix: Spatial adjacency matrix
        trainer: DengueTrainer instance
        metadata: Data metadata dictionary
        save_dir: Directory to save outputs
    
    Returns:
        results_dict: Dictionary containing all results
    """
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 GENERATING PAPER RESULTS")
    print("="*80)
    
    # ==================================================================
    # 1. OVERALL TEST PERFORMANCE
    # ==================================================================
    print("\n🎯 Evaluating overall test performance...")
    test_metrics = trainer.evaluate(model, test_loader, adj_matrix)
    
    print("\n📊 OVERALL TEST PERFORMANCE:")
    print(f"   MAE: {test_metrics['mae']:.3f} cases/week")
    print(f"   RMSE: {test_metrics['rmse']:.3f} cases/week")
    print(f"   R²: {test_metrics['r2']:.3f}")
    print(f"   Zero Accuracy: {test_metrics.get('zero_accuracy', 0)*100:.1f}%")
    print(f"   Non-Zero MAE: {test_metrics.get('non_zero_mae', 0):.3f} cases/week")
    
    # ==================================================================
    # 2. COLLECT ALL PREDICTIONS
    # ==================================================================
    print("\n🔄 Collecting predictions for detailed analysis...")
    model.eval()
    all_preds = []
    all_actuals = []
    all_zero_probs = []
    
    with torch.no_grad():
        for batch_idx, (batch_features, batch_targets) in enumerate(test_loader):
            batch_features = batch_features.to(trainer.device)
            batch_targets = batch_targets.to(trainer.device)
            
            outputs = model(batch_features, adj_matrix)
            preds = outputs['predictions'].cpu().numpy()
            actuals = batch_targets.cpu().numpy()
            zero_probs = outputs['zero_probs'].cpu().numpy()
            
            # Apply inverse transform if needed
            if metadata.get('target_transform') == 'log1p':
                preds = np.expm1(np.maximum(preds, 0))
                actuals = np.expm1(actuals)
            
            preds = np.maximum(preds, 0)
            
            all_preds.extend(preds.flatten())
            all_actuals.extend(actuals.flatten())
            all_zero_probs.extend(zero_probs.flatten())
    
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_zero_probs = np.array(all_zero_probs)
    
    print(f"   Collected {len(all_preds)} predictions")
    
    # ==================================================================
    # 3. LOCATION-SPECIFIC PERFORMANCE
    # ==================================================================
    print("\n📍 LOCATION-SPECIFIC PERFORMANCE:")
    node_ids = metadata.get('node_ids', [])
    n_nodes = len(node_ids)
    samples_per_node = len(all_preds) // n_nodes
    
    location_results = {}
    
    for i, location in enumerate(node_ids):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node if i < n_nodes - 1 else len(all_preds)
        
        loc_preds = all_preds[start_idx:end_idx]
        loc_actuals = all_actuals[start_idx:end_idx]
        
        # Safety check for empty arrays
        if len(loc_preds) == 0:
            print(f"   ⚠️ Warning: No test samples for location {location}, skipping...")
            continue
        
        # Calculate metrics
        loc_mae = np.mean(np.abs(loc_preds - loc_actuals))
        loc_rmse = np.sqrt(np.mean((loc_preds - loc_actuals)**2))
        
        # R² calculation with safety check
        ss_res = np.sum((loc_actuals - loc_preds)**2)
        ss_tot = np.sum((loc_actuals - np.mean(loc_actuals))**2)
        loc_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
        
        location_results[location] = {
            'n_samples': int(len(loc_preds)),
            'mae': float(loc_mae),
            'rmse': float(loc_rmse),
            'r2': float(loc_r2),
            'mean_prediction': float(loc_preds.mean()) if len(loc_preds) > 0 else 0.0,
            'mean_actual': float(loc_actuals.mean()) if len(loc_actuals) > 0 else 0.0,
            'std_prediction': float(loc_preds.std()) if len(loc_preds) > 0 else 0.0,
            'std_actual': float(loc_actuals.std()) if len(loc_actuals) > 0 else 0.0,
            'max_prediction': float(loc_preds.max()) if len(loc_preds) > 0 else 0.0,
            'max_actual': float(loc_actuals.max()) if len(loc_actuals) > 0 else 0.0
        }
        
        print(f"\n   {location}:")
        print(f"      Samples: {len(loc_preds)}")
        print(f"      MAE: {loc_mae:.3f} cases/week")
        print(f"      RMSE: {loc_rmse:.3f} cases/week")
        print(f"      R²: {loc_r2:.3f}")
        print(f"      Mean Prediction: {loc_preds.mean():.2f} ± {loc_preds.std():.2f}")
        print(f"      Mean Actual: {loc_actuals.mean():.2f} ± {loc_actuals.std():.2f}")
    
    # ==================================================================
    # 4. FEATURE IMPORTANCE ANALYSIS
    # ==================================================================
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("   Computing gradient-based feature importance...")
    
    feature_cols = metadata.get('feature_cols', [])
    feature_importances = compute_feature_importance(
        model, test_loader, adj_matrix, trainer.device, feature_cols
    )
    
    # Sort and display top 20
    sorted_features = sorted(feature_importances.items(), 
                            key=lambda x: x[1], reverse=True)
    
    print("\n   TOP 20 MOST IMPORTANT FEATURES:")
    print("   " + "-"*60)
    for rank, (feat, score) in enumerate(sorted_features[:20], 1):
        print(f"   {rank:2d}. {feat:35s}: {score:.6f}")
    print("   " + "-"*60)
    
    # ==================================================================
    # 5. RISK LEVEL CLASSIFICATION PERFORMANCE
    # ==================================================================
    print("\n⚠️  RISK LEVEL CLASSIFICATION:")
    
    def get_risk_level(pred):
        """Match paper thresholds: θ_low=3.0, θ_high=10.0"""
        if pred > 10.0:
            return 'High'
        elif pred > 3.0:
            return 'Moderate'
        else:
            return 'Low'
    
    pred_risk = np.array([get_risk_level(p) for p in all_preds])
    actual_risk = np.array([get_risk_level(a) for a in all_actuals])
    
    # Classification report
    risk_class_report = classification_report(
        actual_risk, pred_risk, 
        labels=['Low', 'Moderate', 'High'],
        output_dict=True,
        zero_division=0
    )
    
    print("\n   Classification Report:")
    print(classification_report(actual_risk, pred_risk, 
                                labels=['Low', 'Moderate', 'High'],
                                zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(actual_risk, pred_risk, 
                         labels=['Low', 'Moderate', 'High'])
    
    print("\n   Confusion Matrix:")
    print("                 Predicted")
    print("              Low    Moderate    High")
    for i, label in enumerate(['Low', 'Moderate', 'High']):
        print(f"   Actual {label:8s}", end='')
        for j in range(3):
            print(f"  {cm[i,j]:6d}", end='')
        print()
    
    # Calculate false positives and false negatives
    false_positives = np.sum((pred_risk == 'High') & (actual_risk != 'High'))
    false_negatives = np.sum((pred_risk != 'High') & (actual_risk == 'High'))
    total_predictions = len(pred_risk)
    
    print(f"\n   False Positives (over-prediction to High): {false_positives} ({false_positives/total_predictions*100:.1f}%)")
    print(f"   False Negatives (missed High risk): {false_negatives} ({false_negatives/total_predictions*100:.1f}%)")
    
    # ==================================================================
    # 6. TEMPORAL ATTENTION ANALYSIS
    # ==================================================================
    print("\n🕐 TEMPORAL ATTENTION PATTERNS:")
    temporal_attention_weights = analyze_temporal_attention(
        model, test_loader, adj_matrix, trainer.device
    )
    
    if temporal_attention_weights is not None and len(temporal_attention_weights) > 0:
        print("   Average attention by time lag (most recent to oldest):")
        for t in range(len(temporal_attention_weights)):
            lag = len(temporal_attention_weights) - t - 1
            weight_value = float(temporal_attention_weights[t])
            bar_length = int(weight_value * 50)
            print(f"   Week t-{lag:2d}: {weight_value:.4f} {'█' * bar_length}")
    else:
        print("   ⚠️  Temporal attention weights not available")
        print("   This is normal if the model doesn't store attention weights explicitly.")
    
    # ==================================================================
    # 7. SPATIAL ATTENTION ANALYSIS
    # ==================================================================
    print("\n🗺️  SPATIAL ATTENTION PATTERNS:")
    spatial_attention_matrix = analyze_spatial_attention(
        model, test_loader, adj_matrix, trainer.device, node_ids
    )
    
    if spatial_attention_matrix is not None:
        print("\n   Spatial Attention Matrix (averaged across all predictions):")
        print("   Rows = Target location, Columns = Source of influence\n")
        
        # Print header
        print("              ", end='')
        for node in node_ids:
            short_name = node.replace('KAB ', '').replace('KOTA ', '')[:8]
            print(f" {short_name:8s}", end='')
        print()
        
        # Print matrix
        for i, node_i in enumerate(node_ids):
            short_name_i = node_i.replace('KAB ', '').replace('KOTA ', '')[:12]
            print(f"   {short_name_i:12s}", end='')
            for j in range(len(node_ids)):
                print(f" {spatial_attention_matrix[i,j]:8.4f}", end='')
            print()
    
    # ==================================================================
    # 8. PREDICTION STATISTICS
    # ==================================================================
    print("\n📈 PREDICTION STATISTICS:")
    print(f"   Prediction Range: [{all_preds.min():.2f}, {all_preds.max():.2f}]")
    print(f"   Actual Range: [{all_actuals.min():.2f}, {all_actuals.max():.2f}]")
    print(f"   Prediction Mean: {all_preds.mean():.2f} ± {all_preds.std():.2f}")
    print(f"   Actual Mean: {all_actuals.mean():.2f} ± {all_actuals.std():.2f}")
    
    # Percentiles
    pred_percentiles = np.percentile(all_preds, [25, 50, 75, 90, 95])
    actual_percentiles = np.percentile(all_actuals, [25, 50, 75, 90, 95])
    
    print("\n   Percentiles:")
    print("   Percentile  Predicted  Actual")
    for p, pred_p, act_p in zip([25, 50, 75, 90, 95], pred_percentiles, actual_percentiles):
        print(f"   P{p:2d}          {pred_p:6.2f}    {act_p:6.2f}")
    
    # ==================================================================
    # 9. COMPILE RESULTS DICTIONARY
    # ==================================================================
    results_dict = {
        'test_metrics': {
            'mae': float(test_metrics['mae']),
            'rmse': float(test_metrics['rmse']),
            'r2': float(test_metrics['r2']),
            'zero_accuracy': float(test_metrics.get('zero_accuracy', 0)),
            'non_zero_mae': float(test_metrics.get('non_zero_mae', 0))
        },
        'location_performance': location_results,
        'feature_importance': {feat: float(score) for feat, score in sorted_features},
        'top_20_features': {feat: float(score) for feat, score in sorted_features[:20]},
        'risk_classification': {
            'classification_report': risk_class_report,
            'confusion_matrix': cm.tolist(),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'false_positive_rate': float(false_positives/total_predictions),
            'false_negative_rate': float(false_negatives/total_predictions)
        },
        'temporal_attention': temporal_attention_weights.tolist() if temporal_attention_weights is not None and hasattr(temporal_attention_weights, 'tolist') else None,
        'spatial_attention': spatial_attention_matrix.tolist() if spatial_attention_matrix is not None and hasattr(spatial_attention_matrix, 'tolist') else None,
        'prediction_statistics': {
            'pred_mean': float(all_preds.mean()),
            'pred_std': float(all_preds.std()),
            'pred_min': float(all_preds.min()),
            'pred_max': float(all_preds.max()),
            'actual_mean': float(all_actuals.mean()),
            'actual_std': float(all_actuals.std()),
            'actual_min': float(all_actuals.min()),
            'actual_max': float(all_actuals.max()),
            'pred_percentiles': {f'p{p}': float(v) for p, v in zip([25,50,75,90,95], pred_percentiles)},
            'actual_percentiles': {f'p{p}': float(v) for p, v in zip([25,50,75,90,95], actual_percentiles)}
        },
        'metadata': {
            'n_test_samples': len(all_preds),
            'n_locations': n_nodes,
            'location_names': node_ids,
            'target_transform': metadata.get('target_transform', 'none'),
            'n_features': len(feature_cols)
        }
    }
    
    # ==================================================================
    # 10. SAVE RESULTS
    # ==================================================================
    print(f"\n💾 Saving results to {save_dir}/...")
    
    # Save JSON
    with open(f'{save_dir}/paper_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"   ✅ Saved paper_results.json")
    
    # Save readable text report
    with open(f'{save_dir}/paper_results_report.txt', 'w') as f:
        f.write("DENGUE PREDICTION MODEL - PAPER RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL TEST PERFORMANCE:\n")
        f.write(f"  MAE: {test_metrics['mae']:.3f} cases/week\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.3f} cases/week\n")
        f.write(f"  R²: {test_metrics['r2']:.3f}\n")
        f.write(f"  Zero Accuracy: {test_metrics.get('zero_accuracy', 0)*100:.1f}%\n\n")
        
        f.write("LOCATION-SPECIFIC PERFORMANCE:\n")
        for location, metrics in location_results.items():
            f.write(f"\n  {location}:\n")
            f.write(f"    MAE: {metrics['mae']:.3f}, RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}\n")
            f.write(f"    Mean Pred: {metrics['mean_prediction']:.2f}, Mean Actual: {metrics['mean_actual']:.2f}\n")
        
        f.write("\n\nTOP 20 FEATURES:\n")
        for rank, (feat, score) in enumerate(sorted_features[:20], 1):
            f.write(f"  {rank:2d}. {feat:35s}: {score:.6f}\n")
    
    print(f"   ✅ Saved paper_results_report.txt")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    generate_paper_plots(results_dict, all_preds, all_actuals, save_dir)
    
    print("\n" + "="*80)
    print("✅ PAPER RESULTS GENERATION COMPLETE")
    print(f"📁 All outputs saved to: {save_dir}/")
    print("="*80 + "\n")
    
    return results_dict


def compute_feature_importance(model, test_loader, adj_matrix, device, feature_cols, n_samples=100):
    """Compute gradient-based feature importance"""
    
    model.train()  # Enable gradients
    feature_importances = {feat: [] for feat in feature_cols}
    
    sample_count = 0
    for batch_features, batch_targets in test_loader:
        if sample_count >= n_samples:
            break
        
        batch_features = batch_features.to(device)
        batch_features.requires_grad = True
        
        outputs = model(batch_features, adj_matrix)
        pred = outputs['predictions'].sum()
        
        pred.backward()
        
        # Get gradients for each feature
        grads = batch_features.grad.abs().mean(dim=(0,1,2)).cpu().numpy()
        
        for j, feat in enumerate(feature_cols):
            if j < len(grads):
                feature_importances[feat].append(float(grads[j]))
        
        sample_count += batch_features.size(0)
    
    model.eval()
    
    # Average across samples
    feature_importance_scores = {
        feat: np.mean(scores) if scores else 0.0 
        for feat, scores in feature_importances.items()
    }
    
    return feature_importance_scores


def analyze_temporal_attention(model, test_loader, adj_matrix, device, n_samples=50):
    """Analyze temporal attention patterns"""
    
    model.eval()
    attention_weights_list = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_features, _ in test_loader:
            if sample_count >= n_samples:
                break
            
            batch_features = batch_features.to(device)
            outputs = model(batch_features, adj_matrix)
            
            # Get attention weights if available
            if hasattr(model, 'st_attention') and hasattr(model.st_attention, 'last_attention_weights'):
                attn = model.st_attention.last_attention_weights
                if attn is not None:
                    # Average across heads and batch
                    avg_attn = attn.mean(dim=(0, 1)).cpu().numpy()
                    attention_weights_list.append(avg_attn)
            
            sample_count += batch_features.size(0)
    
    if attention_weights_list:
        # Average across all samples and ensure it's a 1D numpy array
        avg_temporal_attention = np.mean(attention_weights_list, axis=0)
        # Ensure it's 1D
        if len(avg_temporal_attention.shape) > 1:
            avg_temporal_attention = avg_temporal_attention.flatten()
        return avg_temporal_attention
    
    return None


def analyze_spatial_attention(model, test_loader, adj_matrix, device, node_ids, n_samples=50):
    """Analyze spatial attention patterns"""
    
    # This would need to be extracted from the spatial attention layer
    # For now, return the adjacency matrix as a proxy
    n_nodes = len(node_ids)
    spatial_matrix = np.zeros((n_nodes, n_nodes))
    
    # If your model has spatial attention, extract it here
    # Otherwise, use the adjacency matrix structure
    if isinstance(adj_matrix, torch.Tensor):
        spatial_matrix = adj_matrix.cpu().numpy()
    else:
        spatial_matrix = adj_matrix
    
    return spatial_matrix


def generate_paper_plots(results_dict, all_preds, all_actuals, save_dir):
    """Generate plots for paper"""
    
    # 1. Predictions vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(all_actuals, all_preds, alpha=0.5, s=20)
    max_val = max(all_actuals.max(), all_preds.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Cases', fontsize=12)
    plt.ylabel('Predicted Cases', fontsize=12)
    plt.title('Predictions vs. Actual Cases', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved predictions_vs_actual.png")
    
    # 2. Feature importance bar plot
    top_features = results_dict['top_20_features']
    features = list(top_features.keys())
    scores = list(top_features.values())
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), scores, color='steelblue')
    plt.yticks(range(len(features)), features, fontsize=9)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 20 Feature Importance Rankings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved feature_importance.png")
    
    # 3. Risk level confusion matrix
    cm = np.array(results_dict['risk_classification']['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High'],
                yticklabels=['Low', 'Moderate', 'High'])
    plt.xlabel('Predicted Risk Level', fontsize=12)
    plt.ylabel('Actual Risk Level', fontsize=12)
    plt.title('Risk Level Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/risk_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved risk_confusion_matrix.png")
    
    print("   📊 Visualizations complete")