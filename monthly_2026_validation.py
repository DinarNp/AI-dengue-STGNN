#!/usr/bin/env python3
"""
Genuine prospective validation: train on monthly 2021-2024 data, validate on
2025, test on true 2026 (Jan-Mar) case reports that did not exist when the
training data was assembled -- as real a forecast test as this project can
currently produce, addressing Reviewer 2's core "is this forecasting the
future?" challenge as directly as possible.

Data source: webapp_restructured/database/dengue_app.db (dengue_cases +
climate_data + ndvi_data + regencies, joined and exported to
data/fix/data_monthly_5kab_2021_2026.csv). 2026 rows carry
data_source='csv_import'/'manual' with distinct reported_by_id/timestamps in
April 2026, and 2026 climate/NDVI come from live 'openweather'/'modis'
sources -- this looks like real incremental data entry, not a seeded demo
dataset, but that has NOT been independently confirmed with the user beyond
this technical check.

Sequence construction differs from the main weekly experiments: with only
63 months total (48 train / 12 val / 3 test), an isolated-block-per-split
design (as used for the main weekly split and temporal_cv.py) leaves the
3-month test block too short to form even one window (needs
WINDOW_SIZE_MONTHLY + horizon = 5 consecutive months). Instead, windows are
built once over the FULL chronological series per region, and a sequence is
assigned to train/val/test purely by which period its TARGET month falls in
-- so a test-period prediction may legitimately use trailing history from
the validation period (already-known past data relative to that target),
while training never sees any val/test-period target value. This is the
textbook-correct rolling-forecast construction the reviewer described.

Usage:
    python3 monthly_2026_validation.py --epochs 300 --patience 100 --seed 44
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor
from models.stgnn import STGNNDenguePredictor
from training.trainer import DengueTrainer

DATA_PATH = 'data/fix/data_monthly_5kab_2021_2026.csv'
TRAIN_END = 48  # months 0..47 = Jan2021-Dec2024
VAL_END = 60    # months 48..59 = 2025
# months 60..62 = Jan-Mar 2026 (test)


def load_monthly_data(config: Config):
    pre = DengueDataPreprocessor(config)
    df = pre.load_data(DATA_PATH)
    features, targets, metadata = pre.preprocess_data(df)
    n_nodes = metadata['n_nodes']
    samples_per_node = len(features) // n_nodes
    return features, targets, metadata, n_nodes, samples_per_node


def build_full_series_sequences(features, targets, n_nodes, samples_per_node, window, horizon):
    """One sequence per (seq_idx), containing all n_nodes -- split by target month, not by
    pre-sliced blocks, so short val/test periods can still form sequences using trailing
    context from the immediately preceding (already-known) period."""
    max_seq = samples_per_node - window - horizon + 1
    seqs = {'train': [], 'val': [], 'test': []}
    for seq_idx in range(max_seq):
        target_rel = seq_idx + window + horizon - 1
        split = 'train' if target_rel < TRAIN_END else ('val' if target_rel < VAL_END else 'test')
        seq_features = np.zeros((window, n_nodes, features.shape[1]))
        seq_targets = np.zeros(n_nodes)
        for node_idx in range(n_nodes):
            node_start = node_idx * samples_per_node
            start = node_start + seq_idx
            seq_features[:, node_idx, :] = features[start:start + window]
            seq_targets[node_idx] = targets[node_start + seq_idx + window + horizon - 1]
        seqs[split].append((seq_features, seq_targets))
    return seqs


def build_full_series_tabular(features, targets, n_nodes, samples_per_node, horizon):
    """Per-node (X_t, y_{t+horizon}) pairs, split by target month, for GB/seasonal-naive-style use."""
    out = {'train': {'X': [], 'y': [], 'node': []}, 'val': {'X': [], 'y': [], 'node': []},
           'test': {'X': [], 'y': [], 'node': []}}
    for node_idx in range(n_nodes):
        node_start = node_idx * samples_per_node
        for i in range(0, samples_per_node - horizon):
            target_rel = i + horizon
            split = 'train' if target_rel < TRAIN_END else ('val' if target_rel < VAL_END else 'test')
            out[split]['X'].append(features[node_start + i])
            out[split]['y'].append(targets[node_start + target_rel])
            out[split]['node'].append(node_idx)
    for s in out:
        out[s]['X'] = np.array(out[s]['X'])
        out[s]['y'] = np.array(out[s]['y'])
        out[s]['node'] = np.array(out[s]['node'])
    return out


def eval_metrics(y_true_log, y_pred_log, target_transform):
    if target_transform == 'log1p':
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(np.maximum(y_pred_log, 0))
    else:
        y_true, y_pred = y_true_log, y_pred_log
    y_pred = np.maximum(y_pred, 0)
    return {'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
            'n': int(len(y_true)), 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()}


def run_seasonal_naive_monthly(config: Config):
    pre = DengueDataPreprocessor(config)
    df = pre.load_data(DATA_PATH)
    df = pre.create_date_features(df)
    df = df.sort_values(['Region', 'Year', 'Month']).reset_index(drop=True)
    lookup = {(r['Region'], r['Year'], r['Month']): r['Cases'] for _, r in df.iterrows()}
    region_month_mean = df.groupby(['Region', 'Month'])['Cases'].mean().to_dict()

    test_df = df[df['Year'] == 2026]
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        key = (row['Region'], row['Year'] - 1, row['Month'])
        pred = lookup.get(key, region_month_mean.get((row['Region'], row['Month']), df['Cases'].mean()))
        y_true.append(row['Cases'])
        y_pred.append(pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'n': int(len(y_true)), 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()}


def run_gradient_boosting_monthly(features, targets, metadata, n_nodes, samples_per_node, seed=42):
    pairs = build_full_series_tabular(features, targets, n_nodes, samples_per_node, horizon=1)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
    model.fit(pairs['train']['X'], pairs['train']['y'])
    y_pred = model.predict(pairs['test']['X'])
    return eval_metrics(pairs['test']['y'], y_pred, metadata['target_transform'])


class LSTMOnlyForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_full_series_per_node_sequences(features, targets, n_nodes, samples_per_node, window, horizon):
    out = {'train': {'X': [], 'y': []}, 'val': {'X': [], 'y': []}, 'test': {'X': [], 'y': []}}
    for node_idx in range(n_nodes):
        node_start = node_idx * samples_per_node
        max_seq = samples_per_node - window - horizon + 1
        for i in range(max_seq):
            target_rel = i + window + horizon - 1
            split = 'train' if target_rel < TRAIN_END else ('val' if target_rel < VAL_END else 'test')
            out[split]['X'].append(features[node_start + i:node_start + i + window])
            out[split]['y'].append(targets[node_start + target_rel])
    for s in out:
        out[s]['X'] = np.array(out[s]['X'])
        out[s]['y'] = np.array(out[s]['y'])
    return out


def run_lstm_only_monthly(features, targets, metadata, n_nodes, samples_per_node,
                           window, horizon, epochs, patience, seed=42):
    torch.manual_seed(seed)
    seqs = build_full_series_per_node_sequences(features, targets, n_nodes, samples_per_node, window, horizon)
    model = LSTMOnlyForecaster(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    X_train = torch.FloatTensor(seqs['train']['X'])
    y_train = torch.FloatTensor(seqs['train']['y'])
    X_val = torch.FloatTensor(seqs['val']['X'])
    y_val = torch.FloatTensor(seqs['val']['y'])
    X_test = torch.FloatTensor(seqs['test']['X'])
    best_val, best_state, patience_counter = float('inf'), None, 0
    batch_size = 16
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for b in range(0, len(X_train), batch_size):
            idx = perm[b:b + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val, best_state, patience_counter = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    return eval_metrics(seqs['test']['y'], y_pred, metadata['target_transform'])


def run_stgnn_monthly(features, targets, metadata, n_nodes, samples_per_node,
                       window, horizon, config, epochs, patience, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    seqs = build_full_series_sequences(features, targets, n_nodes, samples_per_node, window, horizon)

    def to_tensors(split_seqs):
        X = torch.FloatTensor(np.stack([s[0] for s in split_seqs]))
        y = torch.FloatTensor(np.stack([s[1] for s in split_seqs]))
        return X, y

    X_train, y_train = to_tensors(seqs['train'])
    X_val, y_val = to_tensors(seqs['val'])
    X_test, y_test = to_tensors(seqs['test'])

    batch_size = metadata.get('adaptive_config', {}).get('BATCH_SIZE', config.BATCH_SIZE)
    loaders = {
        'train': DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True),
        'val': DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        'test': DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
    }

    graph_constructor = GraphConstructor(config)
    spatial_adj = graph_constructor.build_spatial_adjacency(metadata['location_coords'], k_neighbors=3)
    adj_matrix = torch.FloatTensor(spatial_adj)

    trainer = DengueTrainer(config)
    trainer.set_metadata(metadata)

    adaptive_config = metadata.get('adaptive_config', {})
    input_dim = len(metadata['feature_cols'])
    hidden_dim = adaptive_config.get('HIDDEN_DIM', config.HIDDEN_DIM)
    num_layers = adaptive_config.get('NUM_LAYERS', config.NUM_LAYERS)
    model = STGNNDenguePredictor(config=config, input_dim=input_dim, hidden_dim=hidden_dim,
                                  output_dim=1, num_layers=num_layers)

    trained_model, history = trainer.train(model, loaders['train'], loaders['val'], adj_matrix)
    test_metrics = trainer.evaluate(trained_model, loaders['test'], adj_matrix)
    return {'mae': float(test_metrics['mae']), 'rmse': float(test_metrics['rmse']),
            'r2': float(test_metrics['r2']), 'best_epoch': history.get('best_epoch'),
            'n_train_sequences': len(seqs['train']), 'n_val_sequences': len(seqs['val']),
            'n_test_sequences': len(seqs['test'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS_OVERRIDE = args.epochs
    config.PATIENCE_OVERRIDE = args.patience

    features, targets, metadata, n_nodes, samples_per_node = load_monthly_data(config)
    print(f"n_nodes={n_nodes} samples_per_node={samples_per_node} (expect 63: Jan2021-Mar2026)")
    print(f"Split by target month: train<{TRAIN_END} (2021-2024), val<{VAL_END} (2025), test>={VAL_END} (2026 Jan-Mar)")

    window, horizon = config.WINDOW_SIZE_MONTHLY, config.FORECAST_HORIZON_MONTHLY

    results = {}
    print("\nSeasonal-naive (same calendar month, prior year)...")
    results['seasonal_naive'] = run_seasonal_naive_monthly(config)
    print(json.dumps({k: v for k, v in results['seasonal_naive'].items() if k not in ('y_true', 'y_pred')}, indent=2))

    print("\nGradient boosting...")
    results['gradient_boosting'] = run_gradient_boosting_monthly(features, targets, metadata, n_nodes, samples_per_node, seed=args.seed)
    print(json.dumps({k: v for k, v in results['gradient_boosting'].items() if k not in ('y_true', 'y_pred')}, indent=2))

    print("\nLSTM-only...")
    results['lstm_only'] = run_lstm_only_monthly(features, targets, metadata, n_nodes, samples_per_node,
                                                  window, horizon, args.epochs, args.patience, seed=args.seed)
    print(json.dumps({k: v for k, v in results['lstm_only'].items() if k not in ('y_true', 'y_pred')}, indent=2))

    print("\nSTGNN...")
    results['stgnn'] = run_stgnn_monthly(features, targets, metadata, n_nodes, samples_per_node,
                                          window, horizon, config, args.epochs, args.patience, seed=args.seed)
    print(json.dumps(results['stgnn'], indent=2))

    print(f"\n{'='*70}\n2026 PROSPECTIVE VALIDATION SUMMARY (train 2021-2024, val 2025, test 2026 Jan-Mar, n=15)\n{'='*70}")
    for name, r in results.items():
        print(f"{name:<20} MAE={r['mae']:.2f}  RMSE={r.get('rmse', float('nan')):.2f}  R2={r.get('r2')}")

    import os
    os.makedirs('revision_results', exist_ok=True)
    with open('revision_results/monthly_2026_validation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to revision_results/monthly_2026_validation.json")


if __name__ == '__main__':
    main()
