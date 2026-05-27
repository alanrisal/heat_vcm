"""
src/model/network.py

Program Delta Network — the sole learned component in the RHDP architecture.

Architecture
------------
Input  → LayerNorm → Project-Up → [Residual Blocks] → Project-Down → Output

    Input:   K-dim p_g vector (perturbation influence in program space)
    Output:  K-dim Δ vector   (predicted shift in program activities)

Why this design:
  - LayerNorm at input: p_g values come from heat diffusion × NMF projection,
    so their scale varies widely. Normalising at input makes training stable.
  - Project-Up then Residual: expand to a wider hidden space so the network
    has capacity to learn non-linear gain functions, then compress back.
  - Residual connections: allow gradients to flow cleanly in the K=90 / width
    ~360 space without vanishing.
  - No activation on output: Δ can be positive (up-regulation) or negative
    (down-regulation), so we don't clip the final layer.
  - Dropout: K562 essential screen has ~700–800 training perturbations — tiny
    by deep-learning standards. Moderate dropout prevents overfitting.

Parameter count (K=90, hidden_mult=4):
    LayerNorm:       180 params
    Proj-up:         90×360 + 360 = 32,760
    2× Residual:     2 × (360×360 + 360 + 360×360 + 360) = ~518k
    Proj-down:       360×90 + 90 = 32,490
    Total:           ~583k  ← tiny, trains in seconds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: LN → Linear → GELU → Dropout → Linear → + skip.

    Pre-activation (norm before linear) is more stable for small datasets
    than post-activation because the norm prevents the skip connection from
    dominating as weights grow.
    """

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # Small initialisation on the second linear so the residual block
        # starts as near-identity. Helps early training.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h


class ProgramDeltaNetwork(nn.Module):
    """
    MLP that maps a K-dim perturbation influence vector p_g to a K-dim
    program activity delta Δ_g.

    Args:
        K:           Number of NMF programs (input and output dimension).
        hidden_mult: Hidden width = K × hidden_mult. Default 4 → width 360
                     for K=90.
        n_residual:  Number of residual blocks in the hidden space. Default 2.
        dropout:     Dropout probability in residual blocks. Default 0.2.
    """

    def __init__(
        self,
        K:           int   = 90,
        hidden_mult: int   = 4,
        n_residual:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.K      = K
        self.hidden = K * hidden_mult

        # ── Input normalisation ──────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(K)

        # ── Project up: K → hidden ───────────────────────────────────────────
        self.proj_up = nn.Sequential(
            nn.Linear(K, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Residual blocks in hidden space ──────────────────────────────────
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden, dropout=dropout)
            for _ in range(n_residual)
        ])

        # ── Project down: hidden → K (no activation) ─────────────────────────
        self.proj_down = nn.Linear(self.hidden, K)

        self._init_proj()

    def _init_proj(self):
        nn.init.xavier_uniform_(self.proj_up[0].weight)
        nn.init.zeros_(self.proj_up[0].bias)
        # Small init on proj_down so predictions start near zero
        nn.init.normal_(self.proj_down.weight, std=0.01)
        nn.init.zeros_(self.proj_down.bias)

    def forward(self, p_g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_g: (batch, K) perturbation influence vectors.
        Returns:
            delta: (batch, K) predicted program activity deltas.
        """
        x = self.input_norm(p_g)
        x = self.proj_up(x)
        for block in self.residual_blocks:
            x = block(x)
        delta = self.proj_down(x)
        return delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            f"ProgramDeltaNetwork",
            f"  K={self.K}  hidden={self.hidden}  "
            f"n_residual={len(self.residual_blocks)}",
            f"  Parameters: {self.count_parameters():,}",
        ]
        return '\n'.join(lines)