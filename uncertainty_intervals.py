#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals on MAE/RMSE/R2 for the STGNN (loaded from
a saved checkpoint, no retraining needed) and the three baselines, all on
the identical held-out chronological test set. Addresses the remaining half
of Reviewer 2 Major Comment 2 ("report uncertainty intervals on all
metrics") that the seed sweep and temporal CV don't directly cover: this is
the residual-resampling uncertainty of a SINGLE fitted model's test-set
error, not variability across training seeds or time cutoffs.

Usage:
    python3 uncertainty_intervals.py --checkpoint dengue_stgnn_model.pth --n-boot 1000
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.config import Config
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor
from models.stgnn import STGNNDenguePredictor
from training.trainer import DengueTrainer
from baselines import (
    load_prepared_data, run_seasonal_naive, run_gradient_boosting, run_lstm_only,
    build_tabular_pairs, build_sequences, LSTMOnlyForecaster,
)

DATA_PATH = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    maes, rmses, r2s = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        try:
            r2s.append(r2_score(yt, yp))
        except ValueError:
            pass

    def ci(vals):
        vals = np.array(vals)
        return {
            'point': None,  # filled by caller with the non-bootstrapped estimate
            'mean': float(vals.mean()),
            'ci_lower': float(np.percentile(vals, 100 * alpha / 2)),
            'ci_upper': float(np.percentile(vals, 100 * (1 - alpha / 2))),
        }

    return {'mae': ci(maes), 'rmse': ci(rmses), 'r2': ci(r2s)}


def get_stgnn_test_predictions(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    metadata = checkpoint['metadata']

    pre = DengueDataPreprocessor(config)
    df = pre.load_data(DATA_PATH)
    features, targets, metadata2 = pre.preprocess_data(df)

    trainer = DengueTrainer(config)
    trainer.set_metadata(metadata2)
    _, _, test_loader = trainer.create_data_loaders(features, targets, metadata2)

    adaptive_config = metadata2.get('adaptive_config', {})
    input_dim = len(metadata2['feature_cols'])
    hidden_dim = adaptive_config.get('HIDDEN_DIM', config.HIDDEN_DIM)
    num_layers = adaptive_config.get('NUM_LAYERS', config.NUM_LAYERS)
    model = STGNNDenguePredictor(config=config, input_dim=input_dim, hidden_dim=hidden_dim,
                                  output_dim=1, num_layers=num_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    graph_constructor = GraphConstructor(config)
    spatial_adj = graph_constructor.build_spatial_adjacency(metadata2['location_coords'], k_neighbors=3)
    adj_matrix = torch.FloatTensor(spatial_adj)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            outputs = model(batch_features, adj_matrix)
            preds = outputs['predictions'].numpy()
            targs = batch_targets.numpy()
            if metadata2['target_transform'] == 'log1p':
                preds = np.expm1(np.maximum(preds, 0))
                targs = np.expm1(targs)
            preds = np.maximum(preds, 0)
            all_preds.extend(preds.flatten())
            all_targets.extend(targs.flatten())

    return np.array(all_targets), np.array(all_preds), checkpoint.get('test_metrics', {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='dengue_stgnn_model.pth')
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = Config()
    results = {}

    print("STGNN (from checkpoint)...")
    y_true, y_pred, saved_metrics = get_stgnn_test_predictions(args.checkpoint)
    point = {'mae': float(mean_absolute_error(y_true, y_pred)),
             'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
             'r2': float(r2_score(y_true, y_pred))}
    ci = bootstrap_ci(y_true, y_pred, n_boot=args.n_boot, seed=args.seed)
    for k in ci:
        ci[k]['point'] = point[k]
    results['stgnn'] = ci
    print(json.dumps(ci, indent=2))

    for name, fn_result in [
        ('seasonal_naive', run_seasonal_naive(DATA_PATH, config)),
        ('gradient_boosting', run_gradient_boosting(DATA_PATH, config)),
    ]:
        print(f"\n{name}...")
        o = fn_result['overall']
        point = {'mae': o['mae'], 'rmse': o['rmse'], 'r2': o['r2']}
        ci = bootstrap_ci(o['y_true'], o['y_pred'], n_boot=args.n_boot, seed=args.seed)
        for k in ci:
            ci[k]['point'] = point[k]
        results[name] = ci
        print(json.dumps(ci, indent=2))

    print("\nlstm_only...")
    lstm_data = load_prepared_data(DATA_PATH, config)
    seqs = build_sequences(lstm_data, window_size=config.WINDOW_SIZE, horizon=config.FORECAST_HORIZON)
    torch.manual_seed(42)
    model = LSTMOnlyForecaster(input_dim=lstm_data['features'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.HuberLoss(delta=1.0)
    X_train = torch.FloatTensor(seqs['train']['X'])
    y_train = torch.FloatTensor(seqs['train']['y'])
    X_val = torch.FloatTensor(seqs['val']['X'])
    y_val = torch.FloatTensor(seqs['val']['y'])
    X_test = torch.FloatTensor(seqs['test']['X'])
    best_val, best_state, patience_counter = float('inf'), None, 0
    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_train))
        for b in range(0, len(X_train), 16):
            idx = perm[b:b + 16]
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
            if patience_counter >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_log = model(X_test).numpy()
    if lstm_data['metadata']['target_transform'] == 'log1p':
        y_pred_lstm = np.expm1(np.maximum(pred_log, 0))
        y_true_lstm = np.expm1(seqs['test']['y'])
    else:
        y_pred_lstm = np.maximum(pred_log, 0)
        y_true_lstm = seqs['test']['y']
    point_lstm = {'mae': float(mean_absolute_error(y_true_lstm, y_pred_lstm)),
                  'rmse': float(np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))),
                  'r2': float(r2_score(y_true_lstm, y_pred_lstm))}
    ci_lstm = bootstrap_ci(y_true_lstm, y_pred_lstm, n_boot=args.n_boot, seed=args.seed)
    for k in ci_lstm:
        ci_lstm[k]['point'] = point_lstm[k]
    results['lstm_only'] = ci_lstm
    print(json.dumps(ci_lstm, indent=2))

    print(f"\n{'='*70}\n95% BOOTSTRAP CONFIDENCE INTERVALS (n_boot={args.n_boot})\n{'='*70}")
    print(f"{'Model':<20}{'MAE [95% CI]':>30}{'R2 [95% CI]':>30}")
    for name, r in results.items():
        if 'ci_lower' in r.get('mae', {}):
            mae, r2 = r['mae'], r['r2']
            print(f"{name:<20}{mae['point']:.2f} [{mae['ci_lower']:.2f}, {mae['ci_upper']:.2f}]"
                  f"{'':>5}{r2['point']:.3f} [{r2['ci_lower']:.3f}, {r2['ci_upper']:.3f}]")
        else:
            print(f"{name:<20} (point estimate only, no bootstrap -- see comparison_table.csv)")

    import os
    os.makedirs('revision_results', exist_ok=True)
    with open('revision_results/uncertainty_intervals.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to revision_results/uncertainty_intervals.json")


if __name__ == '__main__':
    main()
