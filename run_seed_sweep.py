#!/usr/bin/env python3
"""
Multi-seed robustness sweep for the STGNN and the LSTM-only baseline.

The codebase never called torch.manual_seed/np.random.seed anywhere in the
training path, so every run got a different random initialization -- one
run can stall in a bad basin (flat validation loss for 60+ epochs) while
another converges normally within a handful of epochs. This script fixes
that by seeding each run explicitly and repeating training across multiple
seeds, reporting mean +/- SD (and min/max) on the held-out chronological
test set -- this is also the seed-level uncertainty reporting Reviewer 2
asked for (Major Comment 2 uncertainty intervals; Major Comment 4 stability
across seeds for the attribution/robustness discussion).

Usage:
    python3 run_seed_sweep.py --seeds 42 43 44 45 46 --patience 150
"""
import argparse
import json
import random
import time

import numpy as np
import torch

from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem
from baselines import run_lstm_only

DATA_PATH = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_stgnn_seeded(seed: int, epochs: int, patience: int):
    set_all_seeds(seed)
    config = Config()
    config.EPOCHS_OVERRIDE = epochs
    config.PATIENCE_OVERRIDE = patience
    system = DenguePredictionSystem(config)
    model, metrics, metadata, history = system.run_complete_pipeline(
        data_path=DATA_PATH, generate_paper_analysis=False
    )
    return {
        'mae': float(metrics['mae']), 'rmse': float(metrics['rmse']),
        'r2': float(metrics['r2']), 'best_epoch': history.get('best_epoch'),
    }


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(arr.min()), 'max': float(arr.max()), 'values': values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=150)
    args = parser.parse_args()

    config = Config()
    stgnn_runs, lstm_runs = [], []

    for seed in args.seeds:
        print(f"\n{'='*70}\nSTGNN seed={seed}\n{'='*70}")
        t0 = time.time()
        res = run_stgnn_seeded(seed, args.epochs, args.patience)
        res['seed'] = seed
        res['seconds'] = time.time() - t0
        print(f"seed={seed} -> MAE={res['mae']:.3f} RMSE={res['rmse']:.3f} R2={res['r2']:.3f} best_epoch={res['best_epoch']} ({res['seconds']:.1f}s)")
        stgnn_runs.append(res)

        set_all_seeds(seed)
        lstm_res = run_lstm_only(DATA_PATH, config, epochs=args.epochs, patience=args.patience, seed=seed)
        lstm_o = lstm_res['overall']
        lstm_o['seed'] = seed
        print(f"LSTM-only seed={seed} -> MAE={lstm_o['mae']:.3f} RMSE={lstm_o['rmse']:.3f} R2={lstm_o['r2']:.3f}")
        lstm_runs.append(lstm_o)

    summary = {
        'stgnn': {
            'mae': summarize([r['mae'] for r in stgnn_runs]),
            'rmse': summarize([r['rmse'] for r in stgnn_runs]),
            'r2': summarize([r['r2'] for r in stgnn_runs]),
            'best_epoch': [r['best_epoch'] for r in stgnn_runs],
            'raw_runs': stgnn_runs,
        },
        'lstm_only': {
            'mae': summarize([r['mae'] for r in lstm_runs]),
            'rmse': summarize([r['rmse'] for r in lstm_runs]),
            'r2': summarize([r['r2'] for r in lstm_runs]),
            'raw_runs': lstm_runs,
        },
    }

    print(f"\n{'='*70}\nSEED-SWEEP SUMMARY (n={len(args.seeds)} seeds)\n{'='*70}")
    for name in ('stgnn', 'lstm_only'):
        m = summary[name]
        print(f"{name}: MAE={m['mae']['mean']:.3f}+/-{m['mae']['std']:.3f}  "
              f"RMSE={m['rmse']['mean']:.3f}+/-{m['rmse']['std']:.3f}  "
              f"R2={m['r2']['mean']:.3f}+/-{m['r2']['std']:.3f}")

    with open('revision_results/seed_sweep.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved to revision_results/seed_sweep.json")


if __name__ == '__main__':
    main()
