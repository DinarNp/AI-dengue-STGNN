"""
Train the Canonical Winning STGNN Config and Export a Live-Servable Checkpoint

Trains the exact architecture validated in revision_experiments (real learned
GAT spatial attention, hidden_dim=64, num_layers=2, weekly window_size=8,
forecast_horizon=4, chronological 70/10/20 split) on the canonical
NDVI-corrected weekly dataset, and saves a checkpoint in the
{model_state_dict, config, metadata, ...} dict shape that
`models/predictor.py::DenguePredictor` expects.

This fills a real gap: no existing script in revision_experiments both trains
this exact winning config AND calls torch.save with this dict shape --
generate_final_paper_results.py trains it but only emits JSON/plots (no
checkpoint), and experiments/dengue_pipeline.py saves a checkpoint in the
right shape but with the OLD default config (hidden_dim=256/num_layers=4)
unless the overrides below are set first.

Usage:
    python3 scripts/train_and_export_checkpoint.py [--seed 46] [--epochs 300] [--patience 100] [--warmup 10]

Output:
    webapp_weekly/models/dengue_stgnn_weekly_model.pth
"""
import argparse
import os
import random
import sys

REVISION_EXPERIMENTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'revision_experiments',
)
sys.path.insert(0, REVISION_EXPERIMENTS)

import numpy as np
import torch

from config.config import Config
from experiments.dengue_pipeline import DenguePredictionSystem

DATA_PATH = os.path.join(REVISION_EXPERIMENTS, 'data', 'fix', 'data_weekly_5kab_2021_2025_ndvi_neocorrected.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models',
                            'dengue_stgnn_weekly_model.pth')
HIDDEN_DIM = 64
NUM_LAYERS = 2


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=46, help='matches the seed used for the manuscript\'s illustrative model')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: canonical dataset not found at {DATA_PATH}")
        sys.exit(1)

    set_all_seeds(args.seed)

    config = Config()
    config.EPOCHS_OVERRIDE = args.epochs
    config.PATIENCE_OVERRIDE = args.patience
    config.HIDDEN_DIM_OVERRIDE = HIDDEN_DIM
    config.NUM_LAYERS_OVERRIDE = NUM_LAYERS
    config.WARMUP_EPOCHS = args.warmup

    system = DenguePredictionSystem(config)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cwd = os.getcwd()
    try:
        # run_complete_pipeline saves to a hardcoded relative filename
        # ('dengue_stgnn_model.pth') in the current working directory, so we
        # chdir into the target dir for the duration of the call.
        os.chdir(os.path.dirname(OUTPUT_PATH))
        system.run_complete_pipeline(data_path=DATA_PATH, generate_paper_analysis=False)
        saved_as = os.path.join(os.path.dirname(OUTPUT_PATH), 'dengue_stgnn_model.pth')
    finally:
        os.chdir(cwd)

    os.replace(saved_as, OUTPUT_PATH)
    print(f"\nCheckpoint exported to: {OUTPUT_PATH}")

    # Sanity check: confirm the saved dict shape and the winning config actually landed.
    checkpoint = torch.load(OUTPUT_PATH, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in checkpoint and 'config' in checkpoint and 'metadata' in checkpoint, \
        "checkpoint missing expected keys"
    saved_config = checkpoint['config']
    adaptive = checkpoint['metadata'].get('adaptive_config', {})
    actual_hidden = adaptive.get('HIDDEN_DIM', saved_config.HIDDEN_DIM)
    actual_layers = adaptive.get('NUM_LAYERS', saved_config.NUM_LAYERS)
    print(f"Checkpoint config check: HIDDEN_DIM={actual_hidden}, NUM_LAYERS={actual_layers} "
          f"(expected {HIDDEN_DIM}/{NUM_LAYERS})")
    assert actual_hidden == HIDDEN_DIM and actual_layers == NUM_LAYERS, \
        "checkpoint does not reflect the winning capacity config"
    print(f"Test metrics: {checkpoint.get('test_metrics')}")


if __name__ == '__main__':
    main()
