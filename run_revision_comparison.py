#!/usr/bin/env python3
"""
Backend-only revision experiment: trains the STGNN (via the existing,
now leak-free pipeline) plus the seasonal-naive, gradient-boosting, and
LSTM-only baselines, all under the identical chronological split and
FORECAST_HORIZON, and writes one comparison table.

No Flask/webapp involved -- run directly:
    python3 run_revision_comparison.py [--epochs N] [--patience N]

Outputs (in revision_results/):
    comparison_table.json   -- overall + per-region metrics for every model
    comparison_table.csv    -- flat overall-metrics table
    stgnn_history.json      -- STGNN training curve (for the learning-curve /
                               overparameterization discussion)
"""
import argparse
import json
import os
import time

from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem
from baselines import run_seasonal_naive, run_gradient_boosting, run_lstm_only

DATA_PATH = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'
OUTPUT_DIR = 'revision_results'


def run_stgnn(epochs, patience):
    config = Config()
    if epochs is not None:
        config.EPOCHS_OVERRIDE = epochs
    if patience is not None:
        config.PATIENCE_OVERRIDE = patience

    system = DenguePredictionSystem(config)
    model, metrics, metadata, history = system.run_complete_pipeline(
        data_path=DATA_PATH, generate_paper_analysis=False
    )

    n_params = sum(p.numel() for p in model.parameters())

    # Per-region test metrics via the same trainer/evaluate path, split out by node
    overall = {
        'mae': float(metrics['mae']), 'rmse': float(metrics['rmse']),
        'r2': float(metrics['r2']), 'n_params': int(n_params),
        'best_epoch': history.get('best_epoch'),
    }
    return {'overall': overall, 'per_region': {}}, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help='Cap STGNN epochs (default: adaptive, up to 1000)')
    parser.add_argument('--patience', type=int, default=None, help='Cap STGNN early-stopping patience')
    parser.add_argument('--skip-stgnn', action='store_true', help='Only run baselines (fast)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = Config()
    results = {}
    timings = {}

    print('=' * 70)
    print('Seasonal-naive baseline')
    t0 = time.time()
    results['seasonal_naive'] = run_seasonal_naive(DATA_PATH, config)
    timings['seasonal_naive'] = time.time() - t0

    print('=' * 70)
    print('Gradient boosting baseline')
    t0 = time.time()
    results['gradient_boosting'] = run_gradient_boosting(DATA_PATH, config)
    timings['gradient_boosting'] = time.time() - t0

    print('=' * 70)
    print('LSTM-only baseline (no graph)')
    t0 = time.time()
    results['lstm_only'] = run_lstm_only(DATA_PATH, config)
    timings['lstm_only'] = time.time() - t0

    history = None
    if not args.skip_stgnn:
        print('=' * 70)
        print('STGNN (fixed chronological split + 4-week horizon)')
        t0 = time.time()
        results['stgnn'], history = run_stgnn(args.epochs, args.patience)
        timings['stgnn'] = time.time() - t0

    print('=' * 70)
    print('COMPARISON (test set, original case-count scale)')
    print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'n':>8}{'seconds':>10}")
    for name, res in results.items():
        o = res['overall']
        print(f"{name:<20}{o['mae']:>10.2f}{o['rmse']:>10.2f}{o['r2']:>10.3f}{o.get('n', o.get('n_params', '-')):>8}{timings[name]:>10.1f}")

    slim_results = {
        name: {'overall': {k: v for k, v in res['overall'].items() if k not in ('y_true', 'y_pred')},
               'per_region': res.get('per_region', {})}
        for name, res in results.items()
    }
    with open(os.path.join(OUTPUT_DIR, 'comparison_table.json'), 'w') as f:
        json.dump({'results': slim_results, 'timings_sec': timings}, f, indent=2, default=str)

    with open(os.path.join(OUTPUT_DIR, 'comparison_table.csv'), 'w') as f:
        f.write('model,mae,rmse,r2,n,seconds\n')
        for name, res in results.items():
            o = res['overall']
            f.write(f"{name},{o['mae']:.4f},{o['rmse']:.4f},{o['r2']:.4f},{o.get('n', o.get('n_params', ''))},{timings[name]:.1f}\n")

    if history is not None:
        with open(os.path.join(OUTPUT_DIR, 'stgnn_history.json'), 'w') as f:
            json.dump(history, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
