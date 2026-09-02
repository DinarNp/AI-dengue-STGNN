"""
Baseline models for dengue case-count forecasting, evaluated under the same
chronological (leak-free) split and forecast horizon as the STGNN.

Implements the models Reviewer 1 (Comment 8) and Reviewer 2 (Major Comment 2)
require for comparison:
  - Seasonal-naive: predict this week's target as the same epiweek's case
    count one year earlier (falls back to the region's historical mean for
    that epiweek if the prior year is unavailable).
  - Gradient Boosting: scikit-learn GradientBoostingRegressor on the same
    engineered tabular features the STGNN uses (lags, rolling stats,
    climate/NDVI, cyclical time encodings), trained/evaluated on the same
    log1p target scale.
  - LSTM-only: a single shared unidirectional LSTM over each region's own
    T=WINDOW_SIZE history (no graph, no cross-region attention), predicting
    that region's own target h=FORECAST_HORIZON weeks ahead. This isolates
    the added value of the STGNN's spatial-graph component.

All three share:
  - the identical chronological per-location split boundaries as
    training.trainer.DengueTrainer._create_location_aware_split
    (config.TRAIN_FRACTION / config.VAL_FRACTION, no shuffling), and
  - the identical evaluation convention as training.trainer.DengueTrainer
    .evaluate(): metrics computed in original case-count scale (log1p
    target is inverted with expm1 before scoring).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

from config.config import Config
from data.preprocessor import DengueDataPreprocessor


# ---------------------------------------------------------------------------
# Shared data prep: reuse the exact same preprocessing + split boundaries the
# STGNN pipeline uses, so every model is compared on identical train/val/test
# weeks and an identical feature set.
# ---------------------------------------------------------------------------

def load_prepared_data(data_path: str, config: Config):
    pre = DengueDataPreprocessor(config)
    df = pre.load_data(data_path)
    features, targets, metadata = pre.preprocess_data(df)

    n_nodes = metadata['n_nodes']
    n_total = len(features)
    samples_per_node = n_total // n_nodes
    n_total_trimmed = samples_per_node * n_nodes
    features = features[:n_total_trimmed]
    targets = targets[:n_total_trimmed]

    n_train = int(config.TRAIN_FRACTION * samples_per_node)
    n_val = int(config.VAL_FRACTION * samples_per_node)
    n_test = samples_per_node - n_train - n_val

    return {
        'features': features,
        'targets': targets,
        'metadata': metadata,
        'n_nodes': n_nodes,
        'samples_per_node': samples_per_node,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
    }


def _node_block(data: Dict, node_idx: int) -> Tuple[int, int]:
    start = node_idx * data['samples_per_node']
    return start, start + data['samples_per_node']


def _split_bounds(data: Dict, node_idx: int, fold: Dict[str, Tuple[int, int]] = None) -> Dict[str, Tuple[int, int]]:
    """
    Absolute row-index [start, end) bounds for train/val/test within one node's block.
    If `fold` is given (relative-to-node-start offsets, e.g. from
    temporal_cv.fold_boundaries), those are used instead of the single
    config.TRAIN_FRACTION/VAL_FRACTION split -- this is what enables
    rolling-origin cross-validation to reuse the same tabular/sequence
    builders below for each fold.
    """
    start, _ = _node_block(data, node_idx)
    if fold is not None:
        return {k: (start + a, start + b) for k, (a, b) in fold.items()}
    n_train, n_val = data['n_train'], data['n_val']
    return {
        'train': (start, start + n_train),
        'val': (start + n_train, start + n_train + n_val),
        'test': (start + n_train + n_val, start + data['samples_per_node']),
    }


def build_tabular_pairs(data: Dict, horizon: int, fold: Dict[str, Tuple[int, int]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build (X, y, node_idx) arrays per split. For a row at absolute index i
    within a split's own [start, end) range, the target is targets[i+horizon]
    -- but only kept if i+horizon is ALSO inside that same split's range, so
    no pair ever reaches across a train/val/test boundary.
    """
    features, targets = data['features'], data['targets']
    out = {'train': {'X': [], 'y': [], 'node': []},
           'val': {'X': [], 'y': [], 'node': []},
           'test': {'X': [], 'y': [], 'node': []}}

    for node_idx in range(data['n_nodes']):
        bounds = _split_bounds(data, node_idx, fold=fold)
        for split_name, (start, end) in bounds.items():
            for i in range(start, end - horizon):
                out[split_name]['X'].append(features[i])
                out[split_name]['y'].append(targets[i + horizon])
                out[split_name]['node'].append(node_idx)

    for split_name in out:
        out[split_name]['X'] = np.array(out[split_name]['X'])
        out[split_name]['y'] = np.array(out[split_name]['y'])
        out[split_name]['node'] = np.array(out[split_name]['node'])

    return out


def build_sequences(data: Dict, window_size: int, horizon: int, fold: Dict[str, Tuple[int, int]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-node (no-graph) sliding windows: X is (window_size, n_features), y is a scalar."""
    features, targets = data['features'], data['targets']
    out = {'train': {'X': [], 'y': [], 'node': []},
           'val': {'X': [], 'y': [], 'node': []},
           'test': {'X': [], 'y': [], 'node': []}}

    for node_idx in range(data['n_nodes']):
        bounds = _split_bounds(data, node_idx, fold=fold)
        for split_name, (start, end) in bounds.items():
            last_start = end - window_size - horizon
            for i in range(start, last_start + 1):
                out[split_name]['X'].append(features[i:i + window_size])
                out[split_name]['y'].append(targets[i + window_size + horizon - 1])
                out[split_name]['node'].append(node_idx)

    for split_name in out:
        out[split_name]['X'] = np.array(out[split_name]['X'])
        out[split_name]['y'] = np.array(out[split_name]['y'])
        out[split_name]['node'] = np.array(out[split_name]['node'])

    return out


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                          node_idx: np.ndarray, node_ids: List[str],
                          target_transform: str) -> Dict:
    if target_transform == 'log1p':
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(np.maximum(y_pred_log, 0))
    else:
        y_true = y_true_log
        y_pred = y_pred_log
    y_pred = np.maximum(y_pred, 0)

    overall = {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'n': int(len(y_true)),
        'y_true': y_true,  # original-scale arrays, for downstream bootstrap CIs
        'y_pred': y_pred,
    }

    per_region = {}
    for idx, name in enumerate(node_ids):
        mask = node_idx == idx
        if mask.sum() == 0:
            continue
        per_region[name] = {
            'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
            'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            'r2': float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else float('nan'),
            'n': int(mask.sum()),
        }

    return {'overall': overall, 'per_region': per_region}


# ---------------------------------------------------------------------------
# Baseline 1: Seasonal-naive
# ---------------------------------------------------------------------------

def run_seasonal_naive(data_path: str, config: Config, fold: Dict[str, Tuple[int, int]] = None) -> Dict:
    """
    Predict Cases(region, year, week) as Cases(region, year-1, week). Falls
    back to that region's mean Cases at that epiweek across all other
    available years if the exact prior-year week is missing.
    Operates directly on raw (unscaled) Cases, independent of the STGNN
    feature pipeline, but uses the identical chronological test window.
    """
    pre = DengueDataPreprocessor(config)
    df = pre.load_data(data_path)
    df = pre.create_date_features(df)
    location_col = 'Region' if 'Region' in df.columns else 'Puskesmas'
    df = df.sort_values([location_col, 'Year', 'Week']).reset_index(drop=True)

    node_ids = df[location_col].drop_duplicates().tolist()
    n_nodes = len(node_ids)
    samples_per_node = len(df) // n_nodes
    horizon = config.FORECAST_HORIZON

    lookup = {(row[location_col], row['Year'], row['Week']): row['Cases'] for _, row in df.iterrows()}
    region_week_mean = df.groupby([location_col, 'Week'])['Cases'].mean().to_dict()

    y_true, y_pred, node_idx_out = [], [], []
    for node_idx, name in enumerate(node_ids):
        start = node_idx * samples_per_node
        end = start + samples_per_node
        if fold is not None:
            test_start = start + fold['test'][0]
            test_end = start + fold['test'][1]
        else:
            n_train = int(config.TRAIN_FRACTION * samples_per_node)
            n_val = int(config.VAL_FRACTION * samples_per_node)
            test_start = start + n_train + n_val
            test_end = end
        for i in range(test_start, min(test_end, end) - horizon):
            target_row = df.iloc[i + horizon]
            key = (name, target_row['Year'] - 1, target_row['Week'])
            pred = lookup.get(key)
            if pred is None:
                pred = region_week_mean.get((name, target_row['Week']), df['Cases'].mean())
            y_true.append(target_row['Cases'])
            y_pred.append(pred)
            node_idx_out.append(node_idx)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    node_idx_out = np.array(node_idx_out)

    return evaluate_predictions(y_true, y_pred, node_idx_out, node_ids, target_transform='none')


# ---------------------------------------------------------------------------
# Baseline 2: Gradient Boosting (scikit-learn), on STGNN's own engineered features
# ---------------------------------------------------------------------------

def run_gradient_boosting(data_path: str, config: Config, seed: int = 42,
                           fold: Dict[str, Tuple[int, int]] = None, data: Dict = None) -> Dict:
    if data is None:
        data = load_prepared_data(data_path, config)
    pairs = build_tabular_pairs(data, horizon=config.FORECAST_HORIZON, fold=fold)

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    model.fit(pairs['train']['X'], pairs['train']['y'])

    y_pred_test = model.predict(pairs['test']['X'])
    node_ids = data['metadata']['node_ids']
    result = evaluate_predictions(
        pairs['test']['y'], y_pred_test, pairs['test']['node'], node_ids,
        target_transform=data['metadata']['target_transform'],
    )
    result['feature_importance'] = dict(zip(
        data['metadata']['feature_cols'],
        model.feature_importances_.round(4).tolist(),
    ))
    return result


# ---------------------------------------------------------------------------
# Baseline 3: LSTM-only (no graph, no spatial attention)
# ---------------------------------------------------------------------------

class LSTMOnlyForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def run_lstm_only(data_path: str, config: Config, epochs: int = 100, patience: int = 15,
                   seed: int = 42, device: str = 'cpu',
                   fold: Dict[str, Tuple[int, int]] = None, data: Dict = None) -> Dict:
    torch.manual_seed(seed)
    if data is None:
        data = load_prepared_data(data_path, config)
    seqs = build_sequences(data, window_size=config.WINDOW_SIZE, horizon=config.FORECAST_HORIZON, fold=fold)

    input_dim = data['features'].shape[1]
    model = LSTMOnlyForecaster(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    X_train = torch.FloatTensor(seqs['train']['X']).to(device)
    y_train = torch.FloatTensor(seqs['train']['y']).to(device)
    X_val = torch.FloatTensor(seqs['val']['X']).to(device)
    y_val = torch.FloatTensor(seqs['val']['y']).to(device)
    X_test = torch.FloatTensor(seqs['test']['X']).to(device)

    batch_size = 16
    n_train = len(X_train)
    best_val = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for b in range(0, n_train, batch_size):
            idx = perm[b:b + batch_size]
            optimizer.zero_grad()
            pred = model(X_train[idx])
            loss = criterion(pred, y_train[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test).cpu().numpy()

    node_ids = data['metadata']['node_ids']
    return evaluate_predictions(
        seqs['test']['y'], y_pred_test, seqs['test']['node'], node_ids,
        target_transform=data['metadata']['target_transform'],
    )


if __name__ == '__main__':
    import json
    config = Config()
    data_path = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'

    print("=" * 70)
    print("Running seasonal-naive baseline...")
    seasonal_result = run_seasonal_naive(data_path, config)
    print(json.dumps(seasonal_result['overall'], indent=2))

    print("=" * 70)
    print("Running gradient boosting baseline...")
    gb_result = run_gradient_boosting(data_path, config)
    print(json.dumps(gb_result['overall'], indent=2))

    print("=" * 70)
    print("Running LSTM-only baseline...")
    lstm_result = run_lstm_only(data_path, config)
    print(json.dumps(lstm_result['overall'], indent=2))
