#!/usr/bin/env python3
"""
Model-agnostic permutation importance for the trained STGNN, addressing
Reviewer 2 Major Comment 4: gradient-based saliency (analysis/paper_results.py
compute_feature_importance) is scale-dependent and high-variance; this adds
a stability-tested, model-agnostic alternative reported as mean +/- SD over
repeated shufflings, on the held-out chronological test set only.

For each feature column, its values are shuffled across the sample axis
(independently at every window position and node) while all other features
are kept intact; the resulting increase in test MAE (original case-count
scale) is the feature's importance. Repeated `--repeats` times with
different shuffles to report a standard deviation, and the whole thing is
re-run across `--seeds` model initializations so the ranking's stability
to training-seed variance can be reported too.

Usage:
    python3 run_permutation_importance.py --seed 44 --repeats 20
"""
import argparse
import json

import numpy as np
import torch

from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem

DATA_PATH = 'data/fix/data_weekly_5kab_2021_2025_ndvi.csv'


def get_test_tensors(system, features, targets, metadata):
    _, _, test_loader = system.trainer.create_data_loaders(features, targets, metadata)
    X = np.stack([s['features'] for s in test_loader.dataset.sequences])
    y = np.stack([s['targets'] for s in test_loader.dataset.sequences])
    return torch.FloatTensor(X), torch.FloatTensor(y)


def compute_mae(model, adj_matrix, X, y, target_transform, device):
    model.eval()
    with torch.no_grad():
        preds = model(X.to(device), adj_matrix.to(device))['predictions'].cpu().numpy()
    if target_transform == 'log1p':
        preds = np.expm1(np.maximum(preds, 0))
        y_true = np.expm1(y.numpy())
    else:
        preds = np.maximum(preds, 0)
        y_true = y.numpy()
    preds = np.maximum(preds, 0)
    return float(np.mean(np.abs(preds.flatten() - y_true.flatten())))


def permutation_importance(model, adj_matrix, X, y, feature_cols, target_transform,
                            device, n_repeats: int, seed: int):
    rng = np.random.RandomState(seed)
    baseline_mae = compute_mae(model, adj_matrix, X, y, target_transform, device)

    n_features = X.shape[-1]
    importances = {}
    for feat_idx in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(X.shape[0])
            X_perm = X.clone()
            X_perm[:, :, :, feat_idx] = X_perm[perm][:, :, :, feat_idx]
            mae = compute_mae(model, adj_matrix, X_perm, y, target_transform, device)
            deltas.append(mae - baseline_mae)
        importances[feature_cols[feat_idx]] = {
            'mean_increase': float(np.mean(deltas)),
            'std_increase': float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0,
        }

    ranked = dict(sorted(importances.items(), key=lambda kv: -kv[1]['mean_increase']))
    return {'baseline_mae': baseline_mae, 'importances': ranked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = Config()
    config.EPOCHS_OVERRIDE = args.epochs
    config.PATIENCE_OVERRIDE = args.patience

    system = DenguePredictionSystem(config)
    model, metrics, metadata, history = system.run_complete_pipeline(
        data_path=DATA_PATH, generate_paper_analysis=False
    )
    print(f"\nTrained model test MAE={metrics['mae']:.3f} R2={metrics['r2']:.3f} (seed={args.seed})")

    df = system.preprocessor.load_data(DATA_PATH)
    features, targets, metadata2 = system.preprocessor.preprocess_data(df)
    X_test, y_test = get_test_tensors(system, features, targets, metadata2)

    device = system.trainer.device
    result = permutation_importance(
        model, system.adj_matrix, X_test, y_test,
        metadata['feature_cols'], metadata['target_transform'],
        device, n_repeats=args.repeats, seed=args.seed,
    )

    print(f"\nBaseline test MAE: {result['baseline_mae']:.3f}")
    print(f"{'Feature':<28}{'Mean MAE increase':>20}{'SD':>10}")
    for feat, stats in result['importances'].items():
        print(f"{feat:<28}{stats['mean_increase']:>20.4f}{stats['std_increase']:>10.4f}")

    import os
    os.makedirs('revision_results', exist_ok=True)
    with open('revision_results/permutation_importance.json', 'w') as f:
        json.dump({'seed': args.seed, 'test_mae': metrics['mae'], 'test_r2': metrics['r2'],
                    **result}, f, indent=2, default=str)
    print("\nSaved to revision_results/permutation_importance.json")


if __name__ == '__main__':
    main()
