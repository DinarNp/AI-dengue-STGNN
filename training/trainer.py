import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
from data.dataset import DengueDataset, collate_fn
import os

class DengueTrainer:
    """Training and evaluation class"""
    
    def __init__(self, config):
        self.config = config
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
            # Set environment variable for better MPS performance
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
    def create_data_loaders(self, features: np.ndarray, targets: np.ndarray, 
                       metadata: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders"""
        
        # Time-based split
        n_samples = len(features)
        train_size = int(0.7 * n_samples)
        val_size = int(0.1 * n_samples)
        
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        
        val_features = features[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]
        
        test_features = features[train_size + val_size:]
        test_targets = targets[train_size + val_size:]
        
        # Create datasets
        train_dataset = DengueDataset(train_features, train_targets, metadata,
                                    self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        val_dataset = DengueDataset(val_features, val_targets, metadata,
                                self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        test_dataset = DengueDataset(test_features, test_targets, metadata,
                                    self.config.WINDOW_SIZE, self.config.FORECAST_HORIZON)
        
        # MPS-optimized DataLoader settings
        num_workers = 0 if self.device.type == 'mps' else 4
        pin_memory = self.device.type == 'cuda'  # Only for CUDA
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE,
                                shuffle=True, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=pin_memory)
        
        return train_loader, val_loader, test_loader
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        predictions = outputs['predictions']
        zero_probs = outputs['zero_probs']
        
        # Regression loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Zero-inflation loss (BCE for zero/non-zero classification)
        zero_targets = (targets == 0).float()
        bce_loss = F.binary_cross_entropy(zero_probs, zero_targets)
        
        # Combined loss
        total_loss = (self.config.REGRESSION_WEIGHT * mse_loss + 
                     0.1 * bce_loss)
        
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
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # MPS synchronization (important for accurate timing)
            if self.device.type == 'mps':
                torch.mps.synchronize()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, model: nn.Module, data_loader: DataLoader, 
                adj_matrix: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in data_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = model(batch_features, adj_matrix)
                
                # Compute loss
                loss = self.compute_loss(outputs, batch_targets)
                total_loss += loss.item()
                
                # Store predictions and targets
                predictions = outputs['predictions'].cpu().numpy()
                targets = batch_targets.cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
                
                # MPS synchronization
                if self.device.type == 'mps':
                    torch.mps.synchronize()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)
        
        # Zero-inflated specific metrics
        zero_mask = all_targets == 0
        non_zero_mask = all_targets > 0
        
        zero_accuracy = np.mean((all_predictions[zero_mask] < 0.5)) if np.sum(zero_mask) > 0 else 0.0
        if np.sum(non_zero_mask) > 0:
            non_zero_mae = mean_absolute_error(all_targets[non_zero_mask], 
                                             all_predictions[non_zero_mask])
        else:
            non_zero_mae = 0.0
        
        return {
            'loss': total_loss / len(data_loader),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'zero_accuracy': zero_accuracy,
            'non_zero_mae': non_zero_mae
        }
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             adj_matrix: torch.Tensor) -> Tuple[nn.Module, Dict]:
        """Complete training loop"""
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE,
                              weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        print("Starting training...")
        for epoch in range(self.config.EPOCHS):
            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, adj_matrix)
            
            # Validation
            val_metrics = self.evaluate(model, val_loader, adj_matrix)
            val_loss = val_metrics['loss']
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
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
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
                      f"RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}")
                print(f"Zero Accuracy: {val_metrics['zero_accuracy']:.4f}, "
                      f"Non-zero MAE: {val_metrics['non_zero_mae']:.4f}")
                print("-" * 60)
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history