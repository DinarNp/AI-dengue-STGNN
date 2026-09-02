#!/usr/bin/env python3
"""
Rolling-origin (expanding-window) temporal cross-validation, addressing
Reviewer 2 Major Comment 2's explicit request: "use temporal cross-
validation ... report uncertainty intervals on all metrics."

Distinct from run_seed_sweep.py (same train/test cutoff, different random
initializations): here the train/test time cutoff itself moves forward
across >=3 non-overlapping future test blocks, each time expanding the
training window backward to the start of the series. Every fold's test
block is evaluated by every model (seasonal-naive, gradient boosting,
LSTM-only, STGNN), so the reported mean +/- SD reflects genuine forecast
performance across different points in the case-count trajectory (not
just across STGNN random seeds).

Usage:
    python3 temporal_cv.py --n-folds 4 --epochs 300 --patience 100
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import Config
from data.dataset import DengueDataset, collate_fn
from data.preprocessor import DengueDataPreprocessor
from models.graph_constructor import GraphConstructor
from models.stgnn import STGNNDenguePredictor
from training.trainer import DengueTrainer
from baselines import (
    load_prepared_data, run_seasonal_naive, run_gradient_boosting, run_lstm_only,
)

DATA_PATH = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'


def fold_boundaries(samples_per_node: int, n_folds: int, min_train_frac: float = 0.5,
                     val_frac_of_train: float = 0.15):
    """
    Node-relative [start, end) offsets for train/val/test in each fold.
    Fold i: train = [0, train_end_i - val_len), val = [train_end_i - val_len,
    train_end_i), test = [train_end_i, train_end_i + test_block) (last fold
    absorbs any remainder so all weeks after min_train are eventually tested).
    """
    min_train = int(min_train_frac * samples_per_node)
    remaining = samples_per_node - min_train
    test_block = remaining // n_folds

    folds = []
    for i in range(n_folds):
        train_end = min_train + i * test_block
        val_len = max(4, int(val_frac_of_train * train_end))
        val_start = train_end - val_len
        test_start = train_end
        test_end = test_start + test_block if i < n_folds - 1 else samples_per_node
        folds.append({'train': (0, val_start), 'val': (val_start, train_end), 'test': (test_start, test_end)})
    return folds


def run_stgnn_fold(data, fold, config: Config, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    features, targets, metadata = data['features'], data['targets'], data['metadata']
    n_nodes, samples_per_node = data['n_nodes'], data['samples_per_node']

    split_arrays = {}
    for split_name in ('train', 'val', 'test'):
        a, b = fold[split_name]
        feats, targs = [], []
        for node_idx in range(n_nodes):
            node_start = node_idx * samples_per_node
            feats.append(features[node_start + a:node_start + b])
            targs.append(targets[node_start + a:node_start + b])
        split_arrays[split_name] = (np.concatenate(feats), np.concatenate(targs))

    window_size = config.WINDOW_SIZE
    horizon = config.FORECAST_HORIZON
    # DengueDataset expects one contiguous block per node of equal length;
    # concatenating equal-length per-node slices above preserves that layout,
    # and DengueDataset infers samples-per-node as len(feats)//n_nodes itself.
    datasets = {}
    for split_name, (feats, targs) in split_arrays.items():
        datasets[split_name] = DengueDataset(feats, targs, metadata, window_size, horizon)

    batch_size = metadata.get('adaptive_config', {}).get('BATCH_SIZE', config.BATCH_SIZE)
    loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=(name == 'train'),
                                 collate_fn=collate_fn, drop_last=(name == 'train'))
               for name, ds in datasets.items()}

    if len(datasets['train']) == 0 or len(datasets['test']) == 0:
        return None

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
            'n_train_sequences': len(datasets['train']), 'n_test_sequences': len(datasets['test'])}


def summarize(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {'mean': None, 'std': None, 'values': values}
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 'values': values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=44)
    args = parser.parse_args()

    config = Config()
    config.EPOCHS_OVERRIDE = args.epochs
    config.PATIENCE_OVERRIDE = args.patience

    pre = DengueDataPreprocessor(config)
    df_raw = pre.load_data(DATA_PATH)
    data = load_prepared_data(DATA_PATH, config)
    folds = fold_boundaries(data['samples_per_node'], args.n_folds)

    print(f"Fold boundaries (node-relative week offsets): {folds}")

    results = {'seasonal_naive': [], 'gradient_boosting': [], 'lstm_only': [], 'stgnn': []}

    for i, fold in enumerate(folds):
        print(f"\n{'='*70}\nFOLD {i+1}/{len(folds)}: train=[0,{fold['train'][1]}) val={fold['val']} test={fold['test']}\n{'='*70}")

        sn = run_seasonal_naive(DATA_PATH, config, fold=fold)
        results['seasonal_naive'].append(sn['overall'])
        print(f"  seasonal-naive: MAE={sn['overall']['mae']:.2f} R2={sn['overall']['r2']:.3f}")

        gb = run_gradient_boosting(DATA_PATH, config, fold=fold, data=data)
        results['gradient_boosting'].append(gb['overall'])
        print(f"  gradient-boosting: MAE={gb['overall']['mae']:.2f} R2={gb['overall']['r2']:.3f}")

        lstm = run_lstm_only(DATA_PATH, config, epochs=args.epochs, patience=args.patience,
                              seed=args.seed, fold=fold, data=data)
        results['lstm_only'].append(lstm['overall'])
        print(f"  lstm-only: MAE={lstm['overall']['mae']:.2f} R2={lstm['overall']['r2']:.3f}")

        stgnn = run_stgnn_fold(data, fold, config, seed=args.seed)
        results['stgnn'].append(stgnn)
        if stgnn:
            print(f"  stgnn: MAE={stgnn['mae']:.2f} R2={stgnn['r2']:.3f} (best_epoch={stgnn['best_epoch']}, "
                  f"n_train_seq={stgnn['n_train_sequences']}, n_test_seq={stgnn['n_test_sequences']})")

    summary = {}
    for model_name, fold_results in results.items():
        maes = [r['mae'] for r in fold_results if r is not None]
        rmses = [r['rmse'] for r in fold_results if r is not None]
        r2s = [r['r2'] for r in fold_results if r is not None]
        summary[model_name] = {'mae': summarize(maes), 'rmse': summarize(rmses), 'r2': summarize(r2s),
                                'raw_folds': fold_results}

    print(f"\n{'='*70}\nROLLING-ORIGIN CV SUMMARY ({args.n_folds} folds)\n{'='*70}")
    for name, s in summary.items():
        if s['mae']['mean'] is not None:
            print(f"{name:<20} MAE={s['mae']['mean']:.2f}+/-{s['mae']['std']:.2f}  "
                  f"RMSE={s['rmse']['mean']:.2f}+/-{s['rmse']['std']:.2f}  "
                  f"R2={s['r2']['mean']:.3f}+/-{s['r2']['std']:.3f}")

    import os
    os.makedirs('revision_results', exist_ok=True)
    with open('revision_results/temporal_cv.json', 'w') as f:
        json.dump({'folds': folds, 'results': summary}, f, indent=2, default=str)
    print("\nSaved to revision_results/temporal_cv.json")


if __name__ == '__main__':
    main()
