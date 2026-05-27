"""
src/programs/nmf.py

NMF gene program extraction with PyTorch GPU acceleration.

Factorization:  X  ≈  W · H
    X  (n_cells, n_genes)   — log-normalized control expression
    W  (n_cells, K)         — cell-level program activity scores
    H  (K, n_genes)         — program-gene loading vectors  ← the programs

Algorithm
---------
Uses Lee & Seung multiplicative update rules, which are pure matrix operations
and fully parallelise on GPU:

    H ← H * (W^T X)  / (W^T W H  +  alpha_H·H  +  ε)
    W ← W * (X  H^T) / (W  H H^T              +  ε)

Initialization: sklearn nndsvda on CPU (fast, deterministic, gives a good
starting point that requires far fewer GPU iterations than random init).

GPU / CPU fallback
------------------
If a CUDA device is available it is used automatically.
If not (or if use_gpu=False), everything runs in PyTorch on CPU, which is
still slightly faster than sklearn for large matrices due to more efficient
BLAS calls.

Why not sklearn NMF on GPU?
sklearn's coordinate-descent NMF is inherently sequential (each coordinate
update conditions on the previous). Multiplicative updates are fully parallel
across all elements of W and H simultaneously, making them far better suited
to GPU execution despite needing more iterations to converge.
"""

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.decomposition import NMF as _SklearnNMF

logger = logging.getLogger(__name__)


# ── Device Selection ──────────────────────────────────────────────────────────

def _get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        dev = torch.device('cuda')
        # Probe the device with a small operation before committing.
        # "no kernel image available" means PyTorch was compiled for a different
        # CUDA compute capability than this GPU — we catch it here and fall back
        # rather than crashing mid-computation.
        try:
            _ = torch.zeros(2, 2, device=dev) @ torch.ones(2, 2, device=dev)
            torch.cuda.synchronize()
            name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected and verified: {name} — using CUDA.")
            return dev
        except RuntimeError as e:
            logger.warning(
                f"GPU probe failed ({e}). "
                f"This usually means the installed PyTorch was compiled for a "
                f"different CUDA version than the one on this machine. "
                f"Falling back to CPU. "
                f"To fix: reinstall PyTorch matching your CUDA version from "
                f"https://pytorch.org/get-started/locally/"
            )

    dev = torch.device('cpu')
    if use_gpu and not torch.cuda.is_available():
        logger.info("No CUDA GPU found — running on CPU.")
    else:
        logger.info("Running NMF on CPU.")
    return dev


# ── nndsvda Initialisation ────────────────────────────────────────────────────

def _nndsvda_init(X: np.ndarray, n_components: int, random_state: int):
    """
    Use sklearn's nndsvda initialisation to seed W and H.

    nndsvda is deterministic and places W/H near a good local minimum,
    which cuts the number of multiplicative-update iterations needed by ~3-5×
    compared with random initialisation.

    We only use sklearn here for the init — it never calls fit().
    """
    model = _SklearnNMF(
        n_components=n_components,
        init='nndsvda',
        random_state=random_state,
    )
    # _initialize_nmf is a private sklearn function but stable across versions
    from sklearn.decomposition._nmf import _initialize_nmf
    W0, H0 = _initialize_nmf(X, n_components, init='nndsvda', random_state=random_state)
    # Clip to ensure strict positivity (nndsvda can produce exact zeros)
    eps = np.finfo(np.float32).eps
    W0 = np.clip(W0, eps, None).astype(np.float32)
    H0 = np.clip(H0, eps, None).astype(np.float32)
    return W0, H0


# ── Core NMF Fitting ──────────────────────────────────────────────────────────

def fit_nmf(
    adata: ad.AnnData,
    n_programs: int = 50,
    random_state: int = 42,
    max_iter: int = 1000,
    alpha_H: float = 0.1,
    l1_ratio: float = 0.0,
    tol: float = 1e-3,
    use_gpu: bool = True,
    check_every: int = 20,
) -> tuple:
    """
    Fit NMF via GPU-accelerated multiplicative updates.

    Args:
        adata:        Preprocessed control AnnData (HVG subset, log-normalized).
        n_programs:   K — number of gene programs to extract.
        random_state: Seed for nndsvda initialisation.
        max_iter:     Maximum multiplicative update iterations.
                      Each iteration is a full pass over all elements of W and H.
                      Multiplicative updates typically need 300-800 iterations
                      to converge at tol=1e-3 on K562-scale data.
        alpha_H:      L2 regularisation strength on H (gene loadings).
                      Higher = programs more focused on fewer genes.
                      Good range: 0.05 – 0.5. Default 0.1.
        l1_ratio:     Reserved for future L1/L2 mixing (currently unused in
                      multiplicative updates; L2 is applied via alpha_H).
        tol:          Convergence threshold on relative change in reconstruction
                      error between checks. 1e-3 is appropriate for scRNA-seq.
        use_gpu:      Use CUDA if available. Falls back to CPU silently.
        check_every:  How often (in iterations) to evaluate convergence.
                      Lower = more accurate stopping, higher = faster per-iter.

    Returns:
        (result, W, H) where:
            result — SimpleNamespace with .reconstruction_err_, .n_iter_,
                     .converged (mirrors the sklearn NMF result interface)
            W      — cell activity matrix, shape (n_cells, K), np.ndarray float32
            H      — program-gene loadings, shape (K, n_genes), np.ndarray float32
    """
    # ── Extract dense float32 matrix ─────────────────────────────────────────
    if hasattr(adata.X, 'toarray'):
        X_np = adata.X.toarray().astype(np.float32)
    else:
        X_np = np.array(adata.X, dtype=np.float32)

    min_val = X_np.min()
    if min_val < 0:
        logger.warning(
            f"Expression matrix has min value {min_val:.6f} < 0. "
            f"Clipping to 0 before NMF."
        )
        X_np = np.clip(X_np, 0, None)

    n_cells, n_genes = X_np.shape
    logger.info(
        f"Fitting NMF: K={n_programs}, shape=({n_cells}, {n_genes}), "
        f"alpha_H={alpha_H}, tol={tol}, max_iter={max_iter}"
    )

    # ── Initialise W and H via nndsvda (CPU) ──────────────────────────────────
    logger.info("Initialising W, H via nndsvda...")
    t_init = time.time()
    W_np, H_np = _nndsvda_init(X_np, n_programs, random_state)
    logger.info(f"nndsvda init done in {time.time() - t_init:.1f}s.")

    # ── Move to device ────────────────────────────────────────────────────────
    device = _get_device(use_gpu)
    eps    = torch.tensor(1e-10, dtype=torch.float32, device=device)

    X = torch.from_numpy(X_np).to(device)
    W = torch.from_numpy(W_np).to(device)
    H = torch.from_numpy(H_np).to(device)
    alpha = torch.tensor(alpha_H, dtype=torch.float32, device=device)

    # ── Multiplicative update loop ────────────────────────────────────────────
    # Lee & Seung update rules with L2 regularisation on H:
    #
    #   H ← H * (W^T X)        / (W^T W H  +  alpha·H  +  ε)
    #   W ← W * (X  H^T)       / (W  H H^T             +  ε)
    #
    # All operations are batched matrix multiplications — fully parallel on GPU.

    t0 = time.time()
    prev_err = float('inf')
    converged = False
    n_iter = 0

    for i in range(1, max_iter + 1):
        n_iter = i

        # ── Update H ─────────────────────────────────────────────────────────
        WtX   = W.t() @ X              # (K, n_genes)
        WtWH  = (W.t() @ W) @ H        # (K, n_genes)
        H     = H * WtX / (WtWH + alpha * H + eps)

        # ── Update W ─────────────────────────────────────────────────────────
        XHt   = X @ H.t()              # (n_cells, K)
        WHHt  = W @ (H @ H.t())        # (n_cells, K)
        W     = W * XHt / (WHHt + eps)

        # ── Convergence check ────────────────────────────────────────────────
        if i % check_every == 0 or i == max_iter:
            residual = X - W @ H
            err      = float(torch.norm(residual, p='fro').item())
            rel_change = abs(prev_err - err) / (prev_err + 1e-12)

            if i % (check_every * 10) == 0 or i <= check_every:
                elapsed = time.time() - t0
                logger.info(
                    f"  iter {i:4d}/{max_iter}  "
                    f"recon_err={err:.4f}  "
                    f"rel_Δ={rel_change:.2e}  "
                    f"elapsed={elapsed:.1f}s"
                )

            if rel_change < tol:
                converged = True
                logger.info(
                    f"NMF converged at iter {i}  "
                    f"(rel_change={rel_change:.2e} < tol={tol})"
                )
                break

            prev_err = err

    elapsed = time.time() - t0
    final_err = float(torch.norm(X - W @ H, p='fro').item())

    if not converged:
        logger.warning(
            f"NMF did not converge in {max_iter} iterations "
            f"(final rel_change={rel_change:.2e}, tol={tol}). "
            f"Consider increasing max_iter."
        )

    logger.info(
        f"NMF done: {n_iter} iters, {elapsed:.1f}s, "
        f"recon_err={final_err:.4f}, converged={converged}."
    )

    # ── Return as numpy (move off GPU) ────────────────────────────────────────
    W_out = W.cpu().numpy()
    H_out = H.cpu().numpy()

    # Free GPU memory explicitly
    del X, W, H, residual
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    result = SimpleNamespace(
        reconstruction_err_=final_err,
        n_iter_=n_iter,
        converged=converged,
    )

    return result, W_out, H_out


# ── Program Summary ───────────────────────────────────────────────────────────

def build_program_dataframe(
    H: np.ndarray,
    gene_names: list,
    n_top: int = 100,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of the top-loading genes for each program.

    For biological verification: inspect the top genes of each program and ask
    whether they share a known biological function.

    Args:
        H:          Program-gene loading matrix, shape (K, n_genes).
        gene_names: Ordered list of gene names (columns of H).
        n_top:      Number of top genes to record per program.

    Returns:
        DataFrame with columns: [program_id, rank, gene, loading_score]
    """
    records = []
    for k in range(H.shape[0]):
        scores  = H[k]
        top_idx = np.argsort(scores)[::-1][:n_top]
        for rank, idx in enumerate(top_idx):
            records.append({
                'program_id':    k,
                'rank':          rank + 1,
                'gene':          gene_names[idx],
                'loading_score': float(scores[idx]),
            })
    return pd.DataFrame(records)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_nmf_results(
    W: np.ndarray,
    H: np.ndarray,
    gene_names: list,
    cell_barcodes: list,
    output_dir: str,
) -> None:
    """
    Persist NMF outputs to disk in binary (.npy) and human-readable (.csv) formats.

    Saved files:
        H_matrix.npy    — H matrix, fast to reload  (K × n_genes)
        W_matrix.npy    — W matrix, fast to reload  (n_cells × K)
        H_matrix.csv    — H with gene names as columns
        W_matrix.csv    — W with cell barcodes as index
        top_genes.csv   — top 100 genes per program (tidy format)
        gene_names.txt  — ordered gene list (one per line)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K           = H.shape[0]
    program_ids = [f'Program_{k}' for k in range(K)]

    np.save(out / 'H_matrix.npy', H)
    np.save(out / 'W_matrix.npy', W)

    pd.DataFrame(H, index=program_ids, columns=gene_names).to_csv(out / 'H_matrix.csv')
    pd.DataFrame(W, index=cell_barcodes, columns=program_ids).to_csv(out / 'W_matrix.csv')
    build_program_dataframe(H, gene_names, n_top=100).to_csv(out / 'top_genes.csv', index=False)

    with open(out / 'gene_names.txt', 'w') as f:
        f.write('\n'.join(gene_names))

    logger.info(f"NMF results saved to '{out}/'")


def load_nmf_results(output_dir: str) -> dict:
    """Load previously saved NMF results. Returns dict with H, W, gene_names, top_genes."""
    out = Path(output_dir)
    H   = np.load(out / 'H_matrix.npy')
    W   = np.load(out / 'W_matrix.npy')

    with open(out / 'gene_names.txt') as f:
        gene_names = [line.strip() for line in f.readlines()]

    top_genes = pd.read_csv(out / 'top_genes.csv')
    logger.info(f"Loaded NMF results: H={H.shape}, W={W.shape}, {len(gene_names)} genes.")
    return {'H': H, 'W': W, 'gene_names': gene_names, 'top_genes': top_genes}