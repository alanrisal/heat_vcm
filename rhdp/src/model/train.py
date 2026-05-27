"""
src/model/train.py

Training loop for the Program Delta Network.

Loss
----
Primary: MSE on program-space delta.
    loss = MSE(predicted_delta, observed_delta)

This directly optimises reconstruction of the program-level perturbation
effect. Decoding back to gene space (for MAE/DES evaluation) happens at
inference time, not during training.

Optional cosine alignment term:
    cos_loss = 1 - cosine_similarity(predicted_delta, observed_delta)

This penalises directional errors (wrong sign of delta) independently of
magnitude. Directional accuracy drives DES (differential expression score).
Combined: loss = MSE + λ_cos × cos_loss, λ_cos=0.1 default.

Training strategy
-----------------
- Adam optimiser, lr=1e-3, weight_decay=1e-4
- OneCycleLR scheduler (warmup + cosine anneal)
- Early stopping on validation MSE with patience 30
- Gradient clipping at norm 1.0
- Best checkpoint saved by validation loss

Dataset is small (~700 training genes) so each epoch is sub-second on GPU.
We run 300 epochs maximum; early stopping usually triggers around 80–150.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Loss ──────────────────────────────────────────────────────────────────────

def perturbation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_cos: float = 0.1,
) -> tuple:
    """
    Combined MSE + cosine alignment loss.

    Args:
        pred:       (batch, K) predicted deltas.
        target:     (batch, K) observed deltas.
        lambda_cos: Weight for cosine alignment term.

    Returns:
        (total_loss, mse_loss, cos_loss) — all scalar tensors.
    """
    mse  = F.mse_loss(pred, target)
    cos  = (1.0 - F.cosine_similarity(pred, target, dim=1).mean())
    total = mse + lambda_cos * cos
    return total, mse, cos


# ── Training / validation steps ───────────────────────────────────────────────

def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimiser:  torch.optim.Optimizer,
    scheduler,
    device:     torch.device,
    lambda_cos: float = 0.1,
    clip_grad:  float = 1.0,
) -> dict:
    model.train()
    total_loss = mse_loss = cos_loss = 0.0
    n_batches  = 0

    for p_g, delta in loader:
        p_g   = p_g.to(device)
        delta = delta.to(device)

        optimiser.zero_grad()
        pred    = model(p_g)
        loss, mse, cos = perturbation_loss(pred, delta, lambda_cos)
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimiser.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        mse_loss   += mse.item()
        cos_loss   += cos.item()
        n_batches  += 1

    return {
        'loss': total_loss / n_batches,
        'mse':  mse_loss   / n_batches,
        'cos':  cos_loss   / n_batches,
        'lr':   optimiser.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    lambda_cos: float = 0.1,
) -> dict:
    model.eval()
    total_loss = mse_loss = cos_loss = 0.0
    n_batches  = 0

    for p_g, delta in loader:
        p_g   = p_g.to(device)
        delta = delta.to(device)
        pred  = model(p_g)
        loss, mse, cos = perturbation_loss(pred, delta, lambda_cos)

        total_loss += loss.item()
        mse_loss   += mse.item()
        cos_loss   += cos.item()
        n_batches  += 1

    return {
        'loss': total_loss / n_batches,
        'mse':  mse_loss   / n_batches,
        'cos':  cos_loss   / n_batches,
    }


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training loop with validation, early stopping,
    checkpointing, and logging.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        max_epochs:     int   = 300,
        patience:       int   = 30,
        lambda_cos:     float = 0.1,
        checkpoint_dir: str   = 'outputs/model',
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.max_epochs     = max_epochs
        self.patience       = patience
        self.lambda_cos     = lambda_cos
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimiser = Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # OneCycleLR: warmup for 10% then cosine anneal
        steps_per_epoch = len(train_loader)
        self.scheduler  = OneCycleLR(
            self.optimiser,
            max_lr=lr,
            total_steps=max_epochs * steps_per_epoch,
            pct_start=0.10,
            anneal_strategy='cos',
        )

        self.history     = []
        self.best_val    = float('inf')
        self.best_epoch  = 0
        self.no_improve  = 0

    def fit(self) -> list:
        """
        Run the training loop.

        Returns:
            history — list of dicts, one per epoch, with train/val metrics.
        """
        logger.info(f"Training on {self.device}  |  max_epochs={self.max_epochs}")
        logger.info(model_param_summary(self.model))

        t_start = time.time()

        for epoch in range(1, self.max_epochs + 1):
            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimiser,
                self.scheduler, self.device, self.lambda_cos,
            )
            val_metrics = validate(
                self.model, self.val_loader, self.device, self.lambda_cos
            )

            record = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}':   v for k, v in val_metrics.items()},
            }
            self.history.append(record)

            # ── Logging ───────────────────────────────────────────────────────
            if epoch % 10 == 0 or epoch <= 5:
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch:4d}/{self.max_epochs}  "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_mse={val_metrics['mse']:.4f}  "
                    f"lr={train_metrics['lr']:.2e}  "
                    f"[{elapsed:.0f}s]"
                )

            # ── Checkpoint + early stopping ───────────────────────────────────
            if val_metrics['loss'] < self.best_val:
                self.best_val   = val_metrics['loss']
                self.best_epoch = epoch
                self.no_improve = 0
                self._save_checkpoint('best_model.pt')
            else:
                self.no_improve += 1

            if self.no_improve >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss={self.best_val:.4f} at epoch {self.best_epoch}."
                )
                break

        total_time = time.time() - t_start
        logger.info(
            f"Training complete in {total_time:.1f}s. "
            f"Best val_loss={self.best_val:.4f} at epoch {self.best_epoch}."
        )
        return self.history

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss':    self.best_val,
            'best_epoch':       self.best_epoch,
            'model_config': {
                'K':           self.model.K,
                'hidden_mult': self.model.hidden // self.model.K,
                'n_residual':  len(self.model.residual_blocks),
            },
        }, path)

    def load_best(self):
        """Restore model weights from the best checkpoint."""
        ckpt = torch.load(
            self.checkpoint_dir / 'best_model.pt',
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        logger.info(
            f"Loaded best checkpoint: "
            f"val_loss={ckpt['best_val_loss']:.4f} at epoch {ckpt['best_epoch']}."
        )
        return self.model


def model_param_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Model parameters: {total:,}  ({total/1e6:.3f} M)"