"""
scripts/03_train_model.py

Component 3: Train the Program Delta Network.

Prerequisites
-------------
Component 1: outputs/programs/H_matrix.npy
             outputs/programs/checkpoint_gene_names.txt
             outputs/programs/W_matrix.npy          (control cell activities)

Component 2: outputs/diffusion/P_matrix.npy
             outputs/diffusion/perturbation_genes.txt

Dataset: A processed Replogle K562 h5ad with clean gene-level perturbation
         labels. Download from:
           https://gwps.wi.mit.edu  (select K562 essential)
         or use any h5ad where the perturbation column contains HGNC gene
         symbols (not guide-coordinate identifiers).

Usage
-----
    # Basic run (specify your processed h5ad path)
    python scripts/03_train_model.py --data_path /path/to/k562_processed.h5ad

    # Custom hyperparameters
    python scripts/03_train_model.py \\
        --data_path /path/to/data.h5ad \\
        --perturbation_col perturbation \\
        --k 90 --hidden_mult 4 --n_residual 2 \\
        --lr 1e-3 --max_epochs 400 --batch_size 64

    # Skip PDS computation (much faster evaluation)
    python scripts/03_train_model.py --data_path ... --no_pds

Outputs
-------
    outputs/model/best_model.pt          — model weights
    outputs/model/training_history.csv   — per-epoch train/val losses
    outputs/model/test_metrics.json      — final MAE, DES, PDS
    outputs/model/test_per_gene.csv      — per-gene MAE and DES
    outputs/model/splits.json            — train/val/test gene lists
    outputs/model/figures/
        training_curves.png
        metric_distributions.png
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import log_ram_usage
from src.model.dataset import (
    load_and_preprocess,
    build_training_pairs,
    make_gene_splits,
    PerturbationDataset,
    compute_program_activities_batch,
)
from src.model.network import ProgramDeltaNetwork
from src.model.train import Trainer
from src.model.metrics import evaluate_test_set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Component 3: Train Program Delta Network'
    )
    # Data
    p.add_argument('--data_path', type=str, required=True,
        help='Path to processed Replogle K562 .h5ad with gene-level labels.')
    p.add_argument('--perturbation_col', type=str, default='perturbation',
        help='obs column containing perturbation gene names (default: perturbation).')
    p.add_argument('--programs_dir', type=str, default='outputs/programs',
        help='Component 1 output directory.')
    p.add_argument('--diffusion_dir', type=str, default='outputs/diffusion',
        help='Component 2 output directory.')
    p.add_argument('--output_dir', type=str, default='outputs/model',
        help='Directory for model outputs.')
    p.add_argument('--min_cells', type=int, default=10,
        help='Minimum cells per perturbation to include in training (default: 10).')

    # Architecture
    p.add_argument('--hidden_mult', type=int, default=4,
        help='Hidden width = K × hidden_mult (default: 4).')
    p.add_argument('--n_residual', type=int, default=2,
        help='Number of residual blocks (default: 2).')
    p.add_argument('--dropout', type=float, default=0.2,
        help='Dropout rate (default: 0.2).')

    # Training
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--max_epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lambda_cos', type=float, default=0.1,
        help='Weight for cosine alignment loss term (default: 0.1).')
    p.add_argument('--seed', type=int, default=42)

    # Splits
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--val_frac',   type=float, default=0.15)

    # Evaluation
    p.add_argument('--no_pds', action='store_true',
        help='Skip PDS computation (faster evaluation).')
    p.add_argument('--n_pds_pairs', type=int, default=100)

    # Device
    p.add_argument('--no_gpu', action='store_true')

    return p.parse_args()


# ── Device selection ──────────────────────────────────────────────────────────

def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        try:
            dev = torch.device('cuda')
            _ = torch.zeros(2, 2, device=dev) @ torch.ones(2, 2, device=dev)
            torch.cuda.synchronize()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return dev
        except RuntimeError as e:
            logger.warning(f"GPU probe failed ({e}) — using CPU.")
    logger.info("Using CPU.")
    return torch.device('cpu')


# ── Load Component 1 & 2 outputs ─────────────────────────────────────────────

def load_prerequisites(programs_dir: str, diffusion_dir: str) -> dict:
    prog = Path(programs_dir)
    diff = Path(diffusion_dir)

    H_nmf      = np.load(str(prog / 'H_matrix.npy'))          # (K, G)
    gene_names = (prog / 'checkpoint_gene_names.txt').read_text().strip().splitlines()
    W_ctrl     = np.load(str(prog / 'W_matrix.npy'))           # (n_ctrl, K)

    P_matrix   = np.load(str(diff / 'P_matrix.npy'))           # (n_hvg, K)
    pert_genes = (diff / 'perturbation_genes.txt').read_text().strip().splitlines()

    K = H_nmf.shape[0]
    G = H_nmf.shape[1]

    logger.info(
        f"Loaded prerequisites:\n"
        f"  H_nmf:     {H_nmf.shape}  (K={K} programs × G={G} HVGs)\n"
        f"  W_ctrl:    {W_ctrl.shape} (control cell activities)\n"
        f"  P_matrix:  {P_matrix.shape}\n"
        f"  HVG genes: {len(gene_names)}\n"
        f"  P genes:   {len(pert_genes)}"
    )

    return {
        'H_nmf':      H_nmf,
        'gene_names': gene_names,
        'W_ctrl':     W_ctrl,
        'P_matrix':   P_matrix,
        'pert_genes': pert_genes,
        'K':          K,
    }


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(
    history:     list,
    splits:      dict,
    test_metrics: dict,
    output_dir:  str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)

    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)

    with open(out / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)

    # Serialisable summary of test metrics
    summary = {
        'mean_mae': test_metrics.get('mean_mae'),
        'mean_des': test_metrics.get('mean_des'),
        'pds':      test_metrics.get('pds'),
    }
    with open(out / 'test_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if 'per_gene' in test_metrics:
        test_metrics['per_gene'].to_csv(out / 'test_per_gene.csv', index=False)

    logger.info(f"Outputs saved to '{out}/'")


def plot_training_curves(history: list, output_path: str):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss curves
    axes[0].plot(df['epoch'], df['train_loss'], label='Train', color='steelblue')
    axes[0].plot(df['epoch'], df['val_loss'],   label='Val',   color='crimson')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MSE curves
    axes[1].plot(df['epoch'], df['train_mse'], label='Train MSE', color='steelblue')
    axes[1].plot(df['epoch'], df['val_mse'],   label='Val MSE',   color='crimson')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE (program-space delta)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Training curves saved: {output_path}")


def plot_metric_distributions(test_metrics: dict, output_path: str):
    import matplotlib.pyplot as plt

    df = test_metrics.get('per_gene')
    if df is None or df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df['mae'].dropna(), bins=30, color='steelblue',
                 edgecolor='none', alpha=0.85)
    axes[0].axvline(df['mae'].mean(), color='crimson', lw=2, linestyle='--',
                    label=f"Mean={df['mae'].mean():.4f}")
    axes[0].set_xlabel('MAE per gene')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MAE Distribution (test set)')
    axes[0].legend()

    axes[1].hist(df['des'].dropna(), bins=30, color='darkorange',
                 edgecolor='none', alpha=0.85)
    axes[1].axvline(df['des'].mean(), color='crimson', lw=2, linestyle='--',
                    label=f"Mean={df['des'].mean():.4f}")
    axes[1].set_xlabel('DES (Pearson r)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('DES Distribution (test set)')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Metric distributions saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device(not args.no_gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info('=' * 60)
    logger.info('  COMPONENT 3: Program Delta Network Training')
    logger.info('=' * 60)

    # ── Step 1: Load prerequisites ────────────────────────────────────────────
    logger.info('STEP 1 — Loading Component 1 & 2 outputs')
    pre = load_prerequisites(args.programs_dir, args.diffusion_dir)
    log_ram_usage('after loading prerequisites')

    # ── Step 2: Load & preprocess dataset ────────────────────────────────────
    logger.info('STEP 2 — Loading and preprocessing dataset')
    adata = load_and_preprocess(
        data_path        = args.data_path,
        hvg_gene_names   = pre['gene_names'],
        perturbation_col = args.perturbation_col,
    )
    log_ram_usage('after dataset load')

    # ── Step 3: Build training pairs ─────────────────────────────────────────
    logger.info('STEP 3 — Building training pairs')
    pairs = build_training_pairs(
        adata                   = adata,
        H_nmf                   = pre['H_nmf'],
        P_matrix                = pre['P_matrix'],
        perturbation_genes_in_P = pre['pert_genes'],
        min_cells_per_pert      = args.min_cells,
    )
    log_ram_usage(f"after building {len(pairs['gene_names'])} training pairs")

    # ── Step 4: Gene-level splits ─────────────────────────────────────────────
    logger.info('STEP 4 — Creating train / val / test splits')
    splits = make_gene_splits(
        pairs['gene_names'],
        train_frac = args.train_frac,
        val_frac   = args.val_frac,
        seed       = args.seed,
    )

    train_ds = PerturbationDataset(pairs, splits['train'])
    val_ds   = PerturbationDataset(pairs, splits['val'])
    test_ds  = PerturbationDataset(pairs, splits['test'])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False)

    logger.info(
        f"  Train: {len(train_ds)} genes  "
        f"Val: {len(val_ds)} genes  "
        f"Test: {len(test_ds)} genes"
    )

    # ── Step 5: Build model ───────────────────────────────────────────────────
    logger.info('STEP 5 — Building model')
    model = ProgramDeltaNetwork(
        K           = pre['K'],
        hidden_mult = args.hidden_mult,
        n_residual  = args.n_residual,
        dropout     = args.dropout,
    )
    logger.info(model.summary())

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    logger.info('STEP 6 — Training')
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        max_epochs     = args.max_epochs,
        patience       = args.patience,
        lambda_cos     = args.lambda_cos,
        checkpoint_dir = args.output_dir,
    )
    history = trainer.fit()
    model   = trainer.load_best()

    # ── Step 7: Compute control cell activities for population generation ─────
    logger.info('STEP 7 — Projecting control cells to program space')
    # Use W_ctrl from Component 1 (already computed for the same control cells)
    ctrl_W = pre['W_ctrl']
    logger.info(f"  Using pre-computed W_ctrl: {ctrl_W.shape}")

    # ── Step 8: Evaluate on test set ──────────────────────────────────────────
    logger.info('STEP 8 — Evaluating on test set')
    test_metrics = evaluate_test_set(
        model          = model,
        test_dataset   = test_ds,
        pairs_data     = pairs,
        H_nmf          = pre['H_nmf'],
        ctrl_W         = ctrl_W,
        adata          = adata,
        device         = device,
        n_pds_pairs    = args.n_pds_pairs,
        compute_pds_flag = not args.no_pds,
    )

    # ── Step 9: Save & report ─────────────────────────────────────────────────
    logger.info('STEP 9 — Saving outputs')
    save_outputs(history, splits, test_metrics, args.output_dir)

    plot_training_curves(
        history,
        str(out_dir / 'figures' / 'training_curves.png'),
    )
    plot_metric_distributions(
        test_metrics,
        str(out_dir / 'figures' / 'metric_distributions.png'),
    )

    logger.info('')
    logger.info('=' * 60)
    logger.info('  COMPONENT 3 RESULTS')
    logger.info('=' * 60)
    logger.info(f"  MAE  : {test_metrics['mean_mae']:.4f}")
    logger.info(f"  DES  : {test_metrics['mean_des']:.4f}")
    logger.info(f"  PDS  : {test_metrics['pds']:.4f}")
    logger.info(f"  Model: {out_dir}/best_model.pt")
    logger.info('')

    return model, test_metrics


if __name__ == '__main__':
    main()