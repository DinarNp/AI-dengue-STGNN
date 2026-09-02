# training/trainer.py - COMPLETE FIXED VERSION

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
import traceback


class CombinedDengueLoss(nn.Module):
    """Combined loss for zero-inflated dengue prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.regression_weight = getattr(config, 'REGRESSION_WEIGHT', 0.8)
        self.zero_weight = getattr(config, 'ZERO_INFLATION_WEIGHT', 0.2)
        self.temporal_reg_weight = getattr(config, 'TEMPORAL_REG_WEIGHT', 0.05)
        self.spatial_reg_weight = getattr(config, 'SPATIAL_REG_WEIGHT', 0.05)
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            outputs: dict with 'predictions' and 'zero_probs'
            targets: (batch_size, num_nodes) actual case counts
        """
        predictions = outputs['predictions']
        zero_probs = outputs['zero_probs']
        
        # Handle different shapes
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
        if len(zero_probs.shape) == 1:
            zero_probs = zero_probs.unsqueeze(0)
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(0)
        
        # Ensure shapes match
        if predictions.shape != targets.shape:
            print(f"⚠️ Shape mismatch: pred {predictions.shape}, target {targets.shape}")
            # Reshape if needed
            if predictions.numel() == targets.numel():
                predictions = predictions.reshape(targets.shape)
                zero_probs = zero_probs.reshape(targets.shape)

        # 1. Regression loss (Huber for robustness)
        regression_loss = F.huber_loss(predictions, targets, delta=1.0)
        
        # 2. Zero classification loss (Focal loss for class imbalance)
        zero_targets = (targets == 0).float()
        
        # Focal loss components
        alpha = 0.25
        gamma = 2.0
        pt = torch.where(zero_targets == 1, zero_probs, 1 - zero_probs)
        focal_loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8)
        zero_loss = focal_loss.mean()
        
        # 3. Temporal regularization (only if batch has time dimension)
        temporal_reg = torch.tensor(0.0, device=predictions.device)
        if len(predictions.shape) > 2 and predictions.shape[1] > 1:
            temporal_diff = predictions[:, 1:] - predictions[:, :-1]
            temporal_reg = torch.mean(torch.abs(temporal_diff))
        
        # 4. Spatial regularization (only if batch has multiple nodes)
        spatial_reg = torch.tensor(0.0, device=predictions.device)
        if predictions.shape[-1] > 1:  # Multiple nodes
            spatial_diff = predictions[:, 1:] - predictions[:, :-1]
            spatial_reg = torch.mean(torch.abs(spatial_diff))
        
        # Combined loss
        total_loss = (
            self.regression_weight * regression_loss +
            self.zero_weight * zero_loss +
            self.temporal_reg_weight * temporal_reg +
            self.spatial_reg_weight * spatial_reg
        )
        
        return total_loss


class DengueTrainer:
    """Training and evaluation class - COMPLETELY FIXED"""
    
    def __init__(self, config):
        self.config = config
        self.metadata = None
        
        # ✅ Initialize loss criterion
        self.criterion = CombinedDengueLoss(config)
        
        # Device selection with MPS support
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device('mps')
        #     print("Using Apple Silicon GPU (MPS)")
        #     os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        print(f"✅ Trainer initialized with CombinedDengueLoss")
    
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
        
        # Use 80% train, 10% val, 10% test
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
        
        orig_train = original_targets[train_idx]
        orig_val = original_targets[val_idx]
        orig_test = original_targets[test_idx]
        orig_overall = original_targets.mean()
        
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
        
        test_bias = abs(orig_test.mean() - orig_overall) / orig_overall * 100
        train_test_ratio = orig_test.mean() / orig_train.mean()
        
        print(f"      Test bias: {test_bias:.1f}%")
        print(f"      Test/Train ratio: {train_test_ratio:.2f}")
        
        print(f"\n   🔢 NORMALIZED SCALE (what model sees):")
        print(f"      Train: mean={norm_train.mean():.2f}, std={norm_train.std():.2f}")
        print(f"      Val:   mean={norm_val.mean():.2f}, std={norm_val.std():.2f}")
        print(f"      Test:  mean={norm_test.mean():.2f}, std={norm_test.std():.2f}")

    def _create_stratified_random_split(self, features: np.ndarray, targets: np.ndarray, 
                                    metadata: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        FIXED: Stratified random split to avoid seasonal bias
        This was recommended by the data analysis (bias only 0.4% vs 70.4% for time-based)
        """
        
        from sklearn.model_selection import train_test_split
        
        n_samples = len(features)
        print(f"📅 Creating STRATIFIED RANDOM split for {n_samples} samples...")
        
        # Get location information for stratification
        if 'Region' in metadata or 'node_ids' in metadata:
            n_nodes = metadata.get('n_nodes', 5)
            samples_per_node = n_samples // n_nodes
            
            # Create location labels
            location_labels = np.repeat(np.arange(n_nodes), samples_per_node)
            if len(location_labels) < n_samples:
                # Handle remainder
                location_labels = np.concatenate([
                    location_labels, 
                    np.array([n_nodes-1] * (n_samples - len(location_labels)))
                ])
        else:
            location_labels = None
        
        # First split: 70% train, 30% temp
        if location_labels is not None:
            train_idx, temp_idx = train_test_split(
                np.arange(n_samples),
                test_size=0.30,
                random_state=42,
                stratify=location_labels
            )
            
            # Second split: split temp into val (15%) and test (15%)
            temp_labels = location_labels[temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,  # 50% of 30% = 15%
                random_state=42,
                stratify=temp_labels
            )
        else:
            # Fallback without stratification
            train_idx, temp_idx = train_test_split(
                np.arange(n_samples),
                test_size=0.30,
                random_state=42
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                random_state=42
            )
        
        print(f"   Using 70/15/15 STRATIFIED RANDOM split")
        print(f"   Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        # Verify no seasonal bias
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]
        test_targets = targets[test_idx]
        
        # Apply inverse transform for interpretation
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            train_orig = np.expm1(train_targets)
            val_orig = np.expm1(val_targets)
            test_orig = np.expm1(test_targets)
            overall_orig = np.expm1(targets).mean()
        else:
            train_orig = train_targets
            val_orig = val_targets
            test_orig = test_targets
            overall_orig = targets.mean()
        
        print(f"\n   📊 ORIGINAL SCALE verification:")
        print(f"      Overall: mean={overall_orig:.2f}")
        print(f"      Train:   mean={train_orig.mean():.2f}, std={train_orig.std():.2f}")
        print(f"      Val:     mean={val_orig.mean():.2f}, std={val_orig.std():.2f}")
        print(f"      Test:    mean={test_orig.mean():.2f}, std={test_orig.std():.2f}")
        
        test_bias = abs(test_orig.mean() - overall_orig) / overall_orig * 100
        print(f"      Test bias: {test_bias:.1f}% ✅")
        
        if test_bias > 20:
            print(f"      ⚠️  Warning: Test bias still high despite random split")
        else:
            print(f"      ✅ Low bias achieved with stratified random split!")
        
        return train_idx, val_idx, test_idx

    # def create_data_loaders(self, features: np.ndarray, targets: np.ndarray, 
    #                     metadata: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    #     """Create data loaders with verification"""
        
    #     n_nodes = metadata.get('n_nodes', 5)
    #     n_total = len(features)
        
    #     print(f"\n🔍 Data Loader Verification:")
    #     print(f"   Total samples: {n_total}")
    #     print(f"   Expected nodes: {n_nodes}")
    #     print(f"   Samples per node: {n_total // n_nodes}")
        
    #     if n_total % n_nodes != 0:
    #         print(f"   ⚠️ WARNING: Total samples ({n_total}) not divisible by nodes ({n_nodes})")
    #         n_total_trimmed = (n_total // n_nodes) * n_nodes
    #         features = features[:n_total_trimmed]
    #         targets = targets[:n_total_trimmed]
    #         print(f"   ✂️ Trimmed to {n_total_trimmed} samples")
        
    #     batch_size = metadata.get('adaptive_config', {}).get('BATCH_SIZE', self.config.BATCH_SIZE)
    #     window_size = self.config.WINDOW_SIZE

    #     print(f"   Window size: {window_size}")
    #     print(f"   Batch size: {batch_size}")
        
    #     # Time series split
    #     train_idx, val_idx, test_idx = self._create_enhanced_time_series_split(features, targets, metadata)
        
    #     train_features = features[train_idx]
    #     train_targets = targets[train_idx]
    #     val_features = features[val_idx]
    #     val_targets = targets[val_idx]
    #     test_features = features[test_idx]
    #     test_targets = targets[test_idx]
        
    #     print(f"📊 Final data split: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")
        
    #     # Create datasets
    #     train_dataset = DengueDataset(train_features, train_targets, metadata,
    #                                 window_size, self.config.FORECAST_HORIZON)
    #     val_dataset = DengueDataset(val_features, val_targets, metadata,
    #                             window_size, self.config.FORECAST_HORIZON)
    #     test_dataset = DengueDataset(test_features, test_targets, metadata,
    #                             window_size, self.config.FORECAST_HORIZON)
        
    #     # Create data loaders
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                             shuffle=True, collate_fn=collate_fn,
    #                             num_workers=0, pin_memory=False, drop_last=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size,
    #                         shuffle=False, collate_fn=collate_fn,
    #                         num_workers=0, pin_memory=False, drop_last=False)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                             shuffle=False, collate_fn=collate_fn,
    #                             num_workers=0, pin_memory=False, drop_last=False)
        
    #     return train_loader, val_loader, test_loader

    def create_data_loaders(self, features: np.ndarray, targets: np.ndarray, 
                           metadata: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders with LOCATION-AWARE splitting
        
        Key fix: Keep location groups intact during splitting
        """
        
        n_nodes = metadata.get('n_nodes', 5)
        n_total = len(features)
        
        print(f"\n📊 Creating data loaders:")
        print(f"   Total samples: {n_total}")
        print(f"   Nodes: {n_nodes}")
        
        # ✅ CRITICAL FIX: Ensure data is divisible by n_nodes
        samples_per_node = n_total // n_nodes
        n_total_trimmed = samples_per_node * n_nodes
        
        if n_total != n_total_trimmed:
            print(f"   ⚠️  Trimming from {n_total} to {n_total_trimmed} samples")
            features = features[:n_total_trimmed]
            targets = targets[:n_total_trimmed]
        
        print(f"   Samples per node: {samples_per_node}")
        
        # ✅ LOCATION-AWARE STRATIFIED SPLIT
        train_idx, val_idx, test_idx = self._create_location_aware_split(
            n_total_trimmed, n_nodes, samples_per_node
        )
        
        # Extract splits
        train_features = features[train_idx]
        train_targets = targets[train_idx]
        val_features = features[val_idx]
        val_targets = targets[val_idx]
        test_features = features[test_idx]
        test_targets = targets[test_idx]
        
        print(f"   Train: {len(train_features)} samples")
        print(f"   Val:   {len(val_features)} samples")
        print(f"   Test:  {len(test_features)} samples")
        
        # Verify split quality
        self._verify_split_quality(targets, train_targets, val_targets, test_targets)
        
        # ✅ Create datasets - they will now work correctly
        # Detect if monthly data by checking samples per node (<= 60 suggests monthly)
        is_monthly_data = samples_per_node <= 60
        if is_monthly_data:
            window_size = getattr(self.config, 'WINDOW_SIZE_MONTHLY', 4)
            print(f"   📅 Detected MONTHLY data - using reduced window size: {window_size}")
        else:
            window_size = self.config.WINDOW_SIZE
            print(f"   📅 Detected WEEKLY data - using standard window size: {window_size}")
        forecast_horizon = self.config.FORECAST_HORIZON
        
        train_dataset = DengueDataset(train_features, train_targets, metadata,
                                     window_size, forecast_horizon)
        val_dataset = DengueDataset(val_features, val_targets, metadata,
                                   window_size, forecast_horizon)
        test_dataset = DengueDataset(test_features, test_targets, metadata,
                                    window_size, forecast_horizon)
        
        print(f"   Train sequences: {len(train_dataset)}")
        print(f"   Val sequences:   {len(val_dataset)}")
        print(f"   Test sequences:  {len(test_dataset)}")
        
        # Get batch size
        batch_size = metadata.get('adaptive_config', {}).get('BATCH_SIZE', self.config.BATCH_SIZE)
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
        
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches:   {len(val_loader)}")
        print(f"   Test batches:  {len(test_loader)}")
        
        return train_loader, val_loader, test_loader

    def _create_location_aware_split(self, n_total: int, n_nodes: int,
                                     samples_per_node: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Chronological (forward-chaining) split, per location.

        DengueDataPreprocessor.create_lag_features sorts each location's rows
        by [location, Year, Week/Month] before this point, so within each
        location's contiguous block of `samples_per_node` rows, index order
        IS time order. We therefore take the leading block of each location's
        rows as train, the next block as val, and the trailing block as test
        -- no shuffling. This guarantees:
          (a) every test-period week comes strictly after every training-period
              week for that location, so lag/rolling-mean features and the
              STGNN's sequence windows cannot see test-period information
              during training, and
          (b) sliding-window sequences built downstream (DengueDataset) never
              splice together weeks that are far apart in real time, since
              each split's slice is itself a contiguous run of calendar weeks.

        A prior version of this method shuffled each location's weeks
        (seed=42+node_idx) before taking the 70/15/15 split, then sorted the
        resulting index arrays -- which only cosmetically re-orders an
        already-randomized subset and does not restore contiguity. That
        produced in-sample interpolation dressed up as a forecast; this
        replacement is a genuine held-out-future evaluation.
        """

        print(f"\n📅 Creating CHRONOLOGICAL per-location split (no shuffling)...")

        train_indices = []
        val_indices = []
        test_indices = []

        n_train = int(self.config.TRAIN_FRACTION * samples_per_node) if hasattr(self.config, 'TRAIN_FRACTION') else int(0.70 * samples_per_node)
        n_val = int(self.config.VAL_FRACTION * samples_per_node) if hasattr(self.config, 'VAL_FRACTION') else int(0.15 * samples_per_node)

        # For each location (node)
        for node_idx in range(n_nodes):
            # Get the index range for this location (already time-ordered)
            node_start = node_idx * samples_per_node
            node_end = node_start + samples_per_node
            node_indices = np.arange(node_start, node_end)

            train_indices.extend(node_indices[:n_train])
            val_indices.extend(node_indices[n_train:n_train + n_val])
            test_indices.extend(node_indices[n_train + n_val:])

        # Convert to arrays (already in ascending time order per location, and
        # locations themselves appear in a fixed order, so no re-sort needed)
        train_idx = np.array(train_indices)
        val_idx = np.array(val_indices)
        test_idx = np.array(test_indices)

        print(f"   Split created (per-location weeks {n_train}/{n_val}/{samples_per_node - n_train - n_val}):")
        print(f"      Train: {len(train_idx)} indices (earliest weeks per location)")
        print(f"      Val:   {len(val_idx)} indices (middle weeks per location)")
        print(f"      Test:  {len(test_idx)} indices (latest weeks per location, held out)")

        return train_idx, val_idx, test_idx

    def _verify_split_quality(self, all_targets: np.ndarray,
                              train_targets: np.ndarray,
                              val_targets: np.ndarray,
                              test_targets: np.ndarray):
        """Verify split has no severe bias"""

        # Apply inverse transform if needed
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            all_orig = np.expm1(all_targets)
            train_orig = np.expm1(train_targets)
            val_orig = np.expm1(val_targets)
            test_orig = np.expm1(test_targets)
        else:
            all_orig = all_targets
            train_orig = train_targets
            val_orig = val_targets
            test_orig = test_targets

        overall_mean = all_orig.mean()

        print(f"\n   📊 Split quality check:")
        print(f"      Overall: mean={overall_mean:.2f}")
        print(f"      Train:   mean={train_orig.mean():.2f}")
        print(f"      Val:     mean={val_orig.mean():.2f}")
        print(f"      Test:    mean={test_orig.mean():.2f}")

        test_bias = abs(test_orig.mean() - overall_mean) / overall_mean * 100
        print(f"      Test bias: {test_bias:.1f}% {'✅' if test_bias < 20 else '⚠️'}")

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer, adj_matrix: torch.Tensor) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            try:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_features, adj_matrix)
                
                # Check for NaN in outputs
                if torch.isnan(outputs['predictions']).any():
                    print(f"⚠️ NaN in predictions, skipping batch")
                    continue
                
                # Calculate loss using criterion
                loss = self.criterion(outputs, batch_targets)
                
                if torch.isnan(loss):
                    print(f"⚠️ NaN loss, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                max_norm = self.metadata.get('adaptive_config', {}).get('GRADIENT_CLIP', 1.0) if self.metadata else 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                optimizer.step()
                
                # MPS synchronization
                if self.device.type == 'mps':
                    torch.mps.synchronize()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"❌ Error in training batch: {e}")
                traceback.print_exc()
                continue
        
        if num_batches == 0:
            return float('inf')
        
        return total_loss / num_batches

    def validate(self, model: nn.Module, val_loader: DataLoader, 
                 adj_matrix: torch.Tensor) -> float:
        """Validate model"""

        # Check if loader is empty
        if len(val_loader) == 0:
            print(f"   ⚠️ Validation loader is empty, returning inf")
            return float('inf')
    
        model.eval()
        total_loss = 0.0
        num_batches = 0
        num_skipped = 0
        
        with torch.no_grad():
            for batch_idx, (batch_features, batch_targets) in enumerate(val_loader):
                try:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # Check inputs
                    if torch.isnan(batch_features).any():
                        print(f"⚠️ Validation batch {batch_idx}: NaN in input features")
                        num_skipped += 1
                        continue
                    
                    if torch.isnan(batch_targets).any():
                        print(f"⚠️ Validation batch {batch_idx}: NaN in targets")
                        num_skipped += 1
                        continue
                    
                    outputs = model(batch_features, adj_matrix)
                    
                    # Check outputs
                    if torch.isnan(outputs['predictions']).any():
                        print(f"⚠️ Validation batch {batch_idx}: NaN in predictions")
                        num_skipped += 1
                        continue
                    
                    loss = self.criterion(outputs, batch_targets)
                    
                    if torch.isnan(loss):
                        print(f"⚠️ Validation batch {batch_idx}: NaN loss")
                        num_skipped += 1
                        continue
                    
                    if torch.isinf(loss):
                        print(f"⚠️ Validation batch {batch_idx}: Inf loss")
                        num_skipped += 1
                        continue

                    total_loss += loss.item()
                    num_batches += 1
                        
                    if self.device.type == 'mps':
                        torch.mps.synchronize()
                    
                except Exception as e:
                    print(f"❌ Error in validation batch {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            print(f"   ⚠️ WARNING: All validation batches were skipped!")
            return float('inf')
        
        return total_loss / num_batches

    def evaluate(self, model: nn.Module, data_loader: DataLoader, 
                adj_matrix: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_zero_probs = []
        num_batches_skipped = 0
        
        with torch.no_grad():
            for batch_idx, (batch_features, batch_targets) in enumerate(data_loader):
                try:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    # Check for NaN in inputs
                    if torch.isnan(batch_features).any() or torch.isnan(batch_targets).any():
                        print(f"⚠️ Eval batch {batch_idx}: NaN in inputs")
                        num_batches_skipped += 1
                        continue

                    outputs = model(batch_features, adj_matrix)
                    
                    predictions = outputs['predictions'].detach().cpu().numpy()
                    targets = batch_targets.cpu().numpy()
                    zero_probs = outputs['zero_probs'].cpu().numpy()
                    
                    # Skip if NaN
                    if np.isnan(predictions).any() or np.isnan(targets).any():
                        print(f"⚠️ Eval batch {batch_idx}: NaN in outputs")
                        num_batches_skipped += 1
                        continue
                    
                    all_predictions.extend(predictions.flatten())
                    all_targets.extend(targets.flatten())
                    all_zero_probs.extend(zero_probs.flatten())
                    
                    if self.device.type == 'mps':
                        torch.mps.synchronize()
                    
                except Exception as e:
                    print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                    num_batches_skipped += 1
                    continue
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_zero_probs = np.array(all_zero_probs)
        
        if len(all_predictions) == 0:
            print("🚨 No valid predictions collected")
            return {
                'loss': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'prediction_stats': {}
            }
    
        # Apply inverse transform if needed
        if self.metadata and self.metadata.get('target_transform') == 'log1p':
            all_predictions = np.expm1(np.maximum(all_predictions, 0))
            all_targets = np.expm1(all_targets)
            
        # Ensure non-negative
        all_predictions = np.maximum(all_predictions, 0)
        
        # Remove invalid values
        valid_mask = np.isfinite(all_predictions) & np.isfinite(all_targets)
        if not valid_mask.all():
            print(f"⚠️ Removing {(~valid_mask).sum()} invalid values")
            all_predictions = all_predictions[valid_mask]
            all_targets = all_targets[valid_mask]
            all_zero_probs = all_zero_probs[valid_mask]
        
        if len(all_predictions) == 0:
            print("🚨 No valid predictions after cleaning")
            return {
                'loss': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'prediction_stats': {}
            }
        
        # Calculate metrics
        try:
            mae = mean_absolute_error(all_targets, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            r2 = r2_score(all_targets, all_predictions)
            
            # Zero-specific metrics
            zero_mask = all_targets == 0
            non_zero_mask = all_targets > 0
            
            zero_accuracy = np.mean((all_predictions[zero_mask] < 0.5)) if zero_mask.any() else 0.0
            non_zero_mae = mean_absolute_error(all_targets[non_zero_mask], 
                                            all_predictions[non_zero_mask]) if non_zero_mask.any() else 0.0
            
            prediction_stats = {
                'pred_mean': float(all_predictions.mean()),
                'pred_std': float(all_predictions.std()),
                'pred_zeros': int(np.sum(all_predictions < 0.5)),
                'actual_zeros': int(np.sum(all_targets == 0)),
                'pred_max': float(all_predictions.max()),
                'actual_max': float(all_targets.max())
            }
            
            return {
                'loss': float(mae),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'zero_accuracy': zero_accuracy,
                'non_zero_mae': non_zero_mae,
                'prediction_stats': prediction_stats
            }
            
        except Exception as e:
            print(f"🚨 Error calculating metrics: {e}")
            traceback.print_exc()
            return {
                'loss': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'prediction_stats': {}
            }
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
             adj_matrix: torch.Tensor) -> Tuple[nn.Module, Dict]:
        """Complete training loop"""
        
        # Get adaptive parameters
        if self.metadata and 'adaptive_config' in self.metadata:
            adaptive_config = self.metadata['adaptive_config']
            # lr = adaptive_config.get('LEARNING_RATE', 0.001)
            # weight_decay = adaptive_config.get('WEIGHT_DECAY', 1e-6)
            # epochs = adaptive_config.get('EPOCHS', 100)
            # patience = adaptive_config.get('PATIENCE', 50)
            lr = adaptive_config.get('LEARNING_RATE')
            weight_decay = adaptive_config.get('WEIGHT_DECAY')
            epochs = adaptive_config.get('EPOCHS')
            patience = adaptive_config.get('PATIENCE')
            
            print(f"🎯 Using adaptive training config:")
            print(f"   Learning Rate: {lr}")
            print(f"   Weight Decay: {weight_decay}")
            print(f"   Epochs: {epochs}")
            print(f"   Patience: {patience}")
        else:
            lr = getattr(self.config, 'LEARNING_RATE', 0.001)
            weight_decay = getattr(self.config, 'WEIGHT_DECAY', 1e-6)
            epochs = getattr(self.config, 'EPOCHS', 100)
            patience = getattr(self.config, 'EARLY_STOPPING_PATIENCE', 50)
        
        # Move model to device
        model = model.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=max(5, patience//3), factor=0.5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],      # ← ADD THIS
            'val_rmse': [],     # ← ADD THIS
            'val_r2': [],       # ← ADD THIS
            'learning_rates': [] # ← ADD THIS (optional but useful)
        }
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, adj_matrix)
            
            # Validate
            # val_loss = self.validate(model, val_loader, adj_matrix)

            # ✅ FIX: Get full validation metrics instead of just loss
            val_metrics = self.evaluate(model, val_loader, adj_matrix)
            val_loss = val_metrics['loss']
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(float(val_metrics.get('mae', float('inf'))))
            history['val_rmse'].append(float(val_metrics.get('rmse', float('inf'))))
            history['val_r2'].append(float(val_metrics.get('r2', -float('inf'))))
            history['learning_rates'].append(float(optimizer.param_groups[0]['lr']))
        
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                  f"MAE: {val_metrics['mae']:.3f}, R²: {val_metrics['r2']:.3f} ⭐")
            else:
                patience_counter += 1
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                      f"MAE: {val_metrics['mae']:.3f}")
                    
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"✅ Loaded best model with val_loss={best_val_loss:.4f}")
        
        # ✅ Add summary metrics to history
        history['best_epoch'] = best_epoch
        history['best_val_loss'] = float(best_val_loss)
        # Get the train loss at best epoch (if available)
        if best_epoch > 0 and best_epoch <= len(history['train_loss']):
            history['final_train_mae'] = float(history['train_loss'][best_epoch - 1])
        else:
            history['final_train_mae'] = float(history['train_loss'][-1]) if history['train_loss'] else None
        
        return model, history