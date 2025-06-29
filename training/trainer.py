import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from data.dataset import DengueDataset, collate_fn
import os

class DengueTrainer:
    """Training and evaluation class with fixed stratified data splitting"""
    
    def __init__(self, config):
        self.config = config
        self.metadata = None  # For storing preprocessing metadata
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
    
    def set_metadata(self, metadata: Dict):
        """Set metadata for inverse transforms and adaptive behavior"""
        self.metadata = metadata
        transform_type = metadata.get('target_transform', 'none')
        print(f"🔧 Trainer configured with target transform: {transform_type}")
        
        if transform_type == 'log1p':
            stats = metadata.get('target_stats', {})
            print(f"   Original scale: mean={stats.get('original_mean', 0):.2f}, max={stats.get('original_max', 0):.0f}")
            print(f"   Normalized scale: mean={stats.get('normalized_mean', 0):.2f}, max={stats.get('normalized_max', 0):.2f}")

    def _create_enhanced_time_series_split(self, features: np.ndarray, targets: np.ndarray, 
                                        metadata: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enhanced time series split with seasonal awareness"""
        
        n_samples = len(features)
        
        print(f"📅 Creating ENHANCED time series split for {n_samples} samples...")
        
        # Get original scale targets for better understanding
        original_targets = targets.copy()
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            original_targets = np.expm1(targets)
        
        # For dengue data: Use 80% train, 10% val, 10% test to reduce seasonal bias
        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples) 
        
        train_idx = np.arange(0, train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size, n_samples)
        
        print(f"   Using 80/10/10 split to reduce seasonal bias")
        print(f"   Split points: Train=0-{train_size}, Val={train_size}-{train_size+val_size}, Test={train_size+val_size}-{n_samples}")
        
        # Enhanced verification
        self._enhanced_time_series_verification(original_targets, targets, train_idx, val_idx, test_idx)
        
        return train_idx, val_idx, test_idx

    def _enhanced_time_series_verification(self, original_targets: np.ndarray, normalized_targets: np.ndarray,
                                        train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
        """Enhanced verification with both scales"""
        
        # Original scale analysis
        orig_train = original_targets[train_idx]
        orig_val = original_targets[val_idx]
        orig_test = original_targets[test_idx]
        orig_overall = original_targets.mean()
        
        # Normalized scale analysis
        norm_train = normalized_targets[train_idx]
        norm_val = normalized_targets[val_idx]
        norm_test = normalized_targets[test_idx]
        
        print(f"📈 ENHANCED Time Series Verification:")
        print(f"   Sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        print(f"\n   📊 ORIGINAL SCALE (what we interpret):")
        print(f"      Overall: mean={orig_overall:.2f}")
        print(f"      Train:   mean={orig_train.mean():.2f}, std={orig_train.std():.2f}")
        print(f"      Val:     mean={orig_val.mean():.2f}, std={orig_val.std():.2f}")
        print(f"      Test:    mean={orig_test.mean():.2f}, std={orig_test.std():.2f}")
        
        # Calculate seasonal bias
        test_bias = abs(orig_test.mean() - orig_overall) / orig_overall * 100
        train_test_ratio = orig_test.mean() / orig_train.mean()
        
        print(f"      Test bias: {test_bias:.1f}%")
        print(f"      Test/Train ratio: {train_test_ratio:.2f}")
        
        print(f"\n   🔢 NORMALIZED SCALE (what model sees):")
        print(f"      Train: mean={norm_train.mean():.2f}, std={norm_train.std():.2f}")
        print(f"      Val:   mean={norm_val.mean():.2f}, std={norm_val.std():.2f}")
        print(f"      Test:  mean={norm_test.mean():.2f}, std={norm_test.std():.2f}")
        
        # Seasonal pattern analysis
        print(f"\n   🌡️ SEASONAL ANALYSIS:")
        if test_bias > 50:
            print(f"   🔥 STRONG seasonal pattern detected!")
            print(f"   📋 Test period has {train_test_ratio:.1f}x different cases than training")
            if train_test_ratio > 2:
                print(f"   📈 Test period = HIGH SEASON (epidemic period)")
                expected_performance = "challenging - predicting epidemic from non-epidemic data"
            elif train_test_ratio < 0.5:
                print(f"   📉 Test period = LOW SEASON (endemic period)")
                expected_performance = "moderate - predicting low activity"
            else:
                expected_performance = "reasonable - similar activity levels"
        elif test_bias > 20:
            print(f"   🌤️ Moderate seasonal pattern")
            expected_performance = "good - manageable seasonal variation"
        else:
            print(f"   ⚖️ Weak seasonal pattern")
            expected_performance = "excellent - minimal seasonal bias"
        
        print(f"   🎯 Expected performance: {expected_performance}")
        
        # Performance targets based on seasonal difficulty
        if train_test_ratio > 2 or train_test_ratio < 0.5:
            print(f"   📊 Realistic targets for high seasonal bias:")
            print(f"      MAE: < {orig_overall * 0.8:.1f} (acceptable)")
            print(f"      R²: > -0.5 (decent for seasonal forecasting)")
        else:
            print(f"   📊 Standard targets:")
            print(f"      MAE: < {orig_overall * 0.5:.1f} (good)")
            print(f"      R²: > 0.2 (good forecasting)")

    # ALSO: Enhanced evaluation with seasonal context

    def evaluate_with_seasonal_context(self, model: nn.Module, data_loader: DataLoader, 
                                    adj_matrix: torch.Tensor) -> Dict[str, float]:
        """Evaluate with seasonal forecasting context"""
        
        # Run normal evaluation first
        results = self.evaluate(model, data_loader, adj_matrix)
        
        # Add seasonal context interpretation
        mae = results['mae']
        r2 = results['r2']
        pred_stats = results['prediction_stats']
        
        print(f"\n🌡️ SEASONAL FORECASTING ASSESSMENT:")
        
        # Get seasonal difficulty from metadata
        if self.metadata and 'target_stats' in self.metadata:
            original_mean = self.metadata['target_stats'].get('original_mean', 0)
            
            # Assess performance relative to seasonal difficulty
            relative_mae = mae / original_mean if original_mean > 0 else float('inf')
            
            print(f"   📊 Relative Performance:")
            print(f"      MAE/Mean ratio: {relative_mae:.2f}")
            
            if relative_mae < 0.3:
                seasonal_assessment = "EXCELLENT for seasonal forecasting"
            elif relative_mae < 0.6:
                seasonal_assessment = "GOOD for seasonal forecasting"
            elif relative_mae < 1.0:
                seasonal_assessment = "ACCEPTABLE for seasonal forecasting"
            else:
                seasonal_assessment = "NEEDS IMPROVEMENT for seasonal forecasting"
            
            print(f"   🎯 Assessment: {seasonal_assessment}")
            
            # R² interpretation for time series
            if r2 > 0.3:
                r2_assessment = "Strong predictive power"
            elif r2 > 0.0:
                r2_assessment = "Moderate predictive power"  
            elif r2 > -0.3:
                r2_assessment = "Weak but acceptable for seasonal data"
            else:
                r2_assessment = "Poor - model struggles with patterns"
            
            print(f"   📈 R² interpretation: {r2_assessment}")
            
            # Practical recommendations
            print(f"   💡 Recommendations:")
            if relative_mae > 0.8:
                print(f"      - Add more seasonal features (month, week cyclical)")
                print(f"      - Consider ensemble methods")
                print(f"      - Increase model capacity")
            elif r2 < -0.2:
                print(f"      - Review model architecture")
                print(f"      - Check for overfitting")
            else:
                print(f"      - Current approach is working reasonably well")
                print(f"      - Fine-tune hyperparameters for improvement")
        
        return results
    
    def create_data_loaders(self, features: np.ndarray, targets: np.ndarray, 
                        metadata: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders with ENHANCED TIME SERIES splitting"""
        
        # Use adaptive batch size from metadata
        batch_size = self.config.BATCH_SIZE
        if 'adaptive_config' in metadata:
            batch_size = metadata['adaptive_config']['BATCH_SIZE']
            print(f"📦 Using adaptive batch size: {batch_size}")
        
        # 🎯 UPDATED: Use enhanced time series split instead of stratified
        train_idx, val_idx, test_idx = self._create_enhanced_time_series_split(features, targets, metadata)
        
        # Extract data using indices
        train_features = features[train_idx]
        train_targets = targets[train_idx]
        
        val_features = features[val_idx]
        val_targets = targets[val_idx]
        
        test_features = features[test_idx]
        test_targets = targets[test_idx]
        
        print(f"📊 Final data split: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")
        
        # Create datasets
        train_dataset = DengueDataset(train_features, train_targets, metadata,
                                    self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        val_dataset = DengueDataset(val_features, val_targets, metadata,
                                self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        test_dataset = DengueDataset(test_features, test_targets, metadata,
                                self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        
        # MPS-optimized DataLoader settings
        num_workers = 0 if self.device.type == 'mps' else 4
        pin_memory = self.device.type == 'cuda'
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss with adaptive weighting"""
        predictions = outputs['predictions']
        zero_probs = outputs['zero_probs']
        
        # Regression loss - use MSE for low scale, Huber for high scale
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            # For log-transformed targets, use Huber loss (more robust)
            regression_loss = F.huber_loss(predictions, targets, delta=1.0)
        else:
            # For original scale, use MSE
            regression_loss = F.mse_loss(predictions, targets)
        
        # Zero-inflation loss
        zero_targets = (targets == 0).float()
        bce_loss = F.binary_cross_entropy(zero_probs, zero_targets)
        
        # Combined loss with adaptive weights
        regression_weight = self.config.REGRESSION_WEIGHT
        zero_weight = 0.1 if self.metadata and self.metadata.get('target_transform') == 'log1p' else 0.2
        
        total_loss = regression_weight * regression_loss + zero_weight * bce_loss
        
        return total_loss
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, adj_matrix: torch.Tensor) -> float:
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_features, adj_matrix)
            
            # Compute loss
            loss = self.compute_loss(outputs, batch_targets)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # MPS synchronization
            if self.device.type == 'mps':
                torch.mps.synchronize()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader, 
                adj_matrix: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance with enhanced inverse transform verification"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = model(batch_features, adj_matrix)
                loss = self.compute_loss(outputs, batch_targets)
                total_loss += loss.item()
                
                predictions = outputs['predictions'].cpu().numpy()
                targets = batch_targets.cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
                
                if self.device.type == 'mps':
                    torch.mps.synchronize()
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 🎯 ENHANCED DEBUG: Print detailed before inverse transform
        print(f"🔍 EVALUATION DEBUG:")
        print(f"   Raw model outputs (normalized): pred_mean={all_predictions.mean():.3f}, target_mean={all_targets.mean():.3f}")
        
        # Store original normalized values for comparison
        normalized_predictions = all_predictions.copy()
        normalized_targets = all_targets.copy()
        
        # 🎯 ENHANCED INVERSE TRANSFORM WITH VALIDATION
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            print(f"🔄 Applying inverse log1p transform...")
            
            # Detailed validation before inverse transform
            negative_preds = np.sum(all_predictions < 0)
            if negative_preds > 0:
                print(f"   ⚠️ Found {negative_preds}/{len(all_predictions)} negative predictions, clipping to 0")
                all_predictions = np.maximum(all_predictions, 0)
            
            # Apply inverse transform
            all_predictions_original = np.expm1(all_predictions)
            all_targets_original = np.expm1(all_targets)
            
            print(f"   After inverse: pred_mean={all_predictions_original.mean():.2f}, target_mean={all_targets_original.mean():.2f}")
            
            # 🎯 SANITY CHECK
            if 'target_stats' in self.metadata:
                stats = self.metadata['target_stats']
                expected_mean = stats.get('original_mean', 0)
                
                print(f"   📊 SANITY CHECK:")
                print(f"      Expected overall mean: {expected_mean:.2f}")
                print(f"      Test set mean: {all_targets_original.mean():.2f}")
                
                # Check if test set is representative
                mean_ratio = all_targets_original.mean() / expected_mean if expected_mean > 0 else 0
                print(f"      Test/Overall ratio: {mean_ratio:.2f}")
                
                if 0.8 <= mean_ratio <= 1.2:
                    print(f"   ✅ Test set is representative of overall data")
                elif 0.5 <= mean_ratio <= 1.5:
                    print(f"   ⚠️ Test set somewhat different from overall data")
                else:
                    print(f"   🚨 Test set significantly different from overall data")
            
            all_predictions = all_predictions_original
            all_targets = all_targets_original
        
        else:
            print(f"📈 No transform applied - using original scale")
            
        # Ensure non-negative predictions
        negative_final = np.sum(all_predictions < 0)
        if negative_final > 0:
            print(f"⚠️ Clipping {negative_final} negative final predictions to 0")
            all_predictions = np.maximum(all_predictions, 0)
        
        # Calculate metrics
        try:
            mae = mean_absolute_error(all_targets, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            r2 = r2_score(all_targets, all_predictions)
        except Exception as e:
            print(f"🚨 Error calculating metrics: {e}")
            mae, rmse, r2 = float('inf'), float('inf'), -float('inf')
        
        # Zero-specific metrics
        zero_mask = all_targets == 0
        non_zero_mask = all_targets > 0
        
        zero_accuracy = np.mean((all_predictions[zero_mask] < 0.5)) if np.sum(zero_mask) > 0 else 0.0
        non_zero_mae = mean_absolute_error(all_targets[non_zero_mask], 
                                        all_predictions[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 0.0
        
        # Enhanced prediction statistics
        prediction_stats = {
            'pred_mean': float(all_predictions.mean()),
            'pred_std': float(all_predictions.std()),
            'pred_zeros': int(np.sum(all_predictions < 0.5)),
            'actual_zeros': int(np.sum(all_targets == 0)),
            'pred_max': float(all_predictions.max()),
            'actual_max': float(all_targets.max()),
            'pred_min': float(all_predictions.min()),
            'actual_min': float(all_targets.min()),
            'pred_median': float(np.median(all_predictions)),
            'actual_median': float(np.median(all_targets))
        }
        
        print(f"📊 Final metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return {
            'loss': total_loss / len(data_loader),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'zero_accuracy': zero_accuracy,
            'non_zero_mae': non_zero_mae,
            'prediction_stats': prediction_stats
        }
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             adj_matrix: torch.Tensor) -> Tuple[nn.Module, Dict]:
        """Complete training loop with adaptive configuration"""
        
        # Use adaptive parameters if available
        if self.metadata and 'adaptive_config' in self.metadata:
            adaptive_config = self.metadata['adaptive_config']
            lr = adaptive_config['LEARNING_RATE']
            weight_decay = adaptive_config['WEIGHT_DECAY']
            epochs = adaptive_config['EPOCHS']
            patience = adaptive_config['PATIENCE']
            
            print(f"🎯 Using adaptive training config:")
            print(f"   Learning Rate: {lr}")
            print(f"   Weight Decay: {weight_decay}")
            print(f"   Epochs: {epochs}")
            print(f"   Patience: {patience}")
        else:
            lr = self.config.LEARNING_RATE
            weight_decay = self.config.WEIGHT_DECAY
            epochs = self.config.EPOCHS
            patience = self.config.EARLY_STOPPING_PATIENCE
            print(f"🔧 Using default training config")
        
        # Initialize optimizer with adaptive parameters
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=max(5, patience//3), factor=0.5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': [],
            'learning_rates': []
        }
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, adj_matrix)
            
            # Validation
            val_metrics = self.evaluate(model, val_loader, adj_matrix)
            val_loss = val_metrics['loss']
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Manual verbose logging for scheduler
            if current_lr != old_lr:
                print(f"   📉 Learning rate reduced: {old_lr:.6f} → {current_lr:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_metrics['mae'])
            history['val_rmse'].append(val_metrics['rmse'])
            history['val_r2'].append(val_metrics['r2'])
            history['learning_rates'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"   Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"   Val MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
                print(f"   Zero Accuracy: {val_metrics['zero_accuracy']:.4f}, Non-zero MAE: {val_metrics['non_zero_mae']:.4f}")
                print(f"   Learning Rate: {current_lr:.6f}")
                
                # Print prediction stats for debugging
                stats = val_metrics['prediction_stats']
                print(f"   Pred: mean={stats['pred_mean']:.2f}, max={stats['pred_max']:.2f}, zeros={stats['pred_zeros']}")
                print(f"   Actual: max={stats['actual_max']:.2f}, zeros={stats['actual_zeros']}")
                print("-" * 70)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏰ Early stopping at epoch {epoch + 1} (patience={patience})")
                break
            
            # Stop if learning rate becomes too small
            if current_lr < 1e-7:
                print(f"⏰ Stopping due to very small learning rate: {current_lr:.2e}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model from epoch with val_loss={best_val_loss:.4f}")
        
        return model, history

    def walk_forward_validation(self, features: np.ndarray, targets: np.ndarray, 
                               metadata: Dict, param_grid: Dict) -> Dict:
        """
        Walk-forward validation for time series hyperparameter optimization
        Better than nested CV for temporal data
        """
        
        n_samples = len(features)
        min_train_size = int(0.6 * n_samples)  # Minimum 60% for training
        step_size = int(0.1 * n_samples)       # 10% step forward each time
        
        print(f"🚶 Starting Walk-Forward Validation...")
        print(f"   Min train size: {min_train_size}, Step size: {step_size}")
        print(f"   Total parameter combinations: {self._count_param_combinations(param_grid)}")
        
        best_params = None
        best_score = float('inf')
        all_results = []
        
        # Grid search over parameters
        param_combinations = self._generate_param_combinations(param_grid)
        
        for i, params in enumerate(param_combinations):
            print(f"\n🔧 Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            fold_scores = []
            
            # Walk forward through time
            for fold in range(3):  # 3 time splits instead of 5 folds
                train_end = min_train_size + fold * step_size
                val_start = train_end
                val_end = min(train_end + step_size, n_samples)
                
                if val_end >= n_samples:
                    break
                    
                print(f"   Fold {fold+1}: Train=0-{train_end}, Val={val_start}-{val_end}")
                
                # Create splits
                train_idx = np.arange(0, train_end)
                val_idx = np.arange(val_start, val_end)
                
                # Train model with current params
                try:
                    fold_score = self._train_and_evaluate_fold(
                        features, targets, metadata, train_idx, val_idx, params
                    )
                    fold_scores.append(fold_score)
                    print(f"   Fold {fold+1} MAE: {fold_score:.3f}")
                except Exception as e:
                    print(f"   Fold {fold+1} FAILED: {e}")
                    fold_scores.append(float('inf'))
            
            # Average score across folds
            if fold_scores:
                avg_score = np.mean([s for s in fold_scores if s != float('inf')])
                std_score = np.std([s for s in fold_scores if s != float('inf')])
            else:
                avg_score = float('inf')
                std_score = 0
            
            print(f"   Average MAE: {avg_score:.3f} ± {std_score:.3f}")
            
            all_results.append({
                'params': params,
                'mean_score': avg_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            })
            
            # Update best params
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"   🎯 New best score: {best_score:.3f}")
        
        # Final evaluation with best params
        print(f"\n✅ Walk-Forward Validation Complete!")
        print(f"🏆 Best parameters found:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        print(f"🎯 Best average MAE: {best_score:.3f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }

    def _generate_param_combinations(self, param_grid: dict) -> list[dict]:
        """Generate all parameter combinations for grid search"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _count_param_combinations(self, param_grid: Dict) -> int:
        """Count total parameter combinations"""
        total = 1
        for values in param_grid.values():
            total *= len(values)
        return total

    def _train_and_evaluate_fold(self, features: np.ndarray, targets: np.ndarray,
                               metadata: Dict, train_idx: np.ndarray, val_idx: np.ndarray,
                               params: Dict) -> float:
        """Train and evaluate model for one fold"""
        
        # Extract fold data
        fold_train_features = features[train_idx]
        fold_train_targets = targets[train_idx]
        fold_val_features = features[val_idx]
        fold_val_targets = targets[val_idx]
        
        # Update config with params
        fold_metadata = metadata.copy()
        fold_metadata['adaptive_config'] = {**metadata.get('adaptive_config', {}), **params}
        
        # Create datasets and loaders
        from data.dataset import DengueDataset, collate_fn
        
        train_dataset = DengueDataset(fold_train_features, fold_train_targets, fold_metadata,
                                    self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        val_dataset = DengueDataset(fold_val_features, fold_val_targets, fold_metadata,
                                  self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        
        batch_size = params.get('BATCH_SIZE', 16)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        # Initialize model
        from models.stgnn import STGNNDenguePredictor
        input_dim = len(metadata['feature_cols'])
        fold_model = STGNNDenguePredictor(self.config, input_dim, metadata['n_nodes'])
        fold_model = fold_model.to(self.device)
        
        # Create adjacency matrix
        location_coords = metadata['location_coords']
        from models.graph_constructor import GraphConstructor
        graph_constructor = GraphConstructor(self.config)
        spatial_adj = graph_constructor.build_spatial_adjacency(location_coords)
        adj_matrix = torch.FloatTensor(spatial_adj).to(self.device)
        
        # Store original metadata and update for fold
        original_metadata = self.metadata
        self.metadata = fold_metadata
        
        # Quick training with reduced epochs for efficiency
        original_config = self.config
        temp_config = self.config
        
        # Update temp config with fold params
        for key, value in params.items():
            if hasattr(temp_config, key):
                setattr(temp_config, key, value)
        
        # Reduce epochs for validation efficiency
        setattr(temp_config, 'EPOCHS', min(params.get('EPOCHS', 100), 50))
        setattr(temp_config, 'EARLY_STOPPING_PATIENCE', 8)
        
        self.config = temp_config
        
        try:
            # Train model
            fold_model, _ = self.train(fold_model, train_loader, val_loader, adj_matrix)
            
            # Evaluate
            val_metrics = self.evaluate(fold_model, val_loader, adj_matrix)
            
            return val_metrics['mae']
            
        except Exception as e:
            print(f"   Training failed: {e}")
            return float('inf')
            
        finally:
            # Restore original config and metadata
            self.config = original_config
            self.metadata = original_metadata