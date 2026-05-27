"""
scripts/inspect_adata.py

Diagnostic utility — run this BEFORE 02_compute_diffusion.py when the
perturbation column is not obvious.

Prints:
  1. All obs columns with dtype, unique count, and 5 example values.
  2. All uns keys and their types/contents.
  3. Any obs column whose values overlap with known human gene symbols.
  4. Attempted parse of guide-coordinate identifiers to extract gene names.

Usage:
    python scripts/inspect_adata.py
    python scripts/inspect_adata.py --data_path /path/to/data.h5ad
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--no_pertpy',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading dataset...")
    if not args.no_pertpy:
        try:
            import pertpy as pt
            adata = pt.data.replogle_2022_k562_essential()
            print(f"Loaded via pertpy: {adata.n_obs:,} cells × {adata.n_vars:,} genes\n")
        except Exception as e:
            print(f"pertpy load failed: {e}")
            sys.exit(1)
    else:
        import scanpy as sc
        adata = sc.read_h5ad(args.data_path)
        print(f"Loaded from disk: {adata.n_obs:,} cells × {adata.n_vars:,} genes\n")

    sep = '─' * 70

    # ── 1. obs columns ────────────────────────────────────────────────────────
    print(sep)
    print("OBS COLUMNS")
    print(sep)
    print(f"  {'Column':<35} {'Dtype':<12} {'N unique':>9}  Examples")
    print(f"  {'─'*35} {'─'*12} {'─'*9}  {'─'*40}")

    low_card_cols = []
    for col in adata.obs.columns:
        try:
            series  = adata.obs[col].astype(str)
            n_uniq  = series.nunique()
            examples = '  |  '.join(series.drop_duplicates().iloc[:3].tolist())[:60]
            flag = ''
            if 2 <= n_uniq <= 5000:
                flag = '  ← possible gene column'
                low_card_cols.append(col)
            print(f"  {col:<35} {str(adata.obs[col].dtype):<12} {n_uniq:>9}  {examples}{flag}")
        except Exception as ex:
            print(f"  {col:<35} (error: {ex})")

    print()

    # ── 2. uns keys ───────────────────────────────────────────────────────────
    print(sep)
    print("UNS KEYS  (unstructured metadata)")
    print(sep)
    if not adata.uns:
        print("  (empty)")
    for key, val in adata.uns.items():
        if hasattr(val, '__len__'):
            print(f"  {key:<35} type={type(val).__name__}  len={len(val)}")
            # Print first few entries for dicts / lists
            if isinstance(val, dict):
                for k2, v2 in list(val.items())[:3]:
                    print(f"      {str(k2)[:30]}: {str(v2)[:50]}")
            elif hasattr(val, '__iter__') and not isinstance(val, str):
                try:
                    samples = list(val)[:3]
                    print(f"      first 3: {samples}")
                except Exception:
                    pass
        else:
            print(f"  {key:<35} {str(val)[:60]}")
    print()

    # ── 3. obsm / obsp keys ───────────────────────────────────────────────────
    if adata.obsm:
        print(sep)
        print("OBSM KEYS  (cell embeddings)")
        print(sep)
        for key, val in adata.obsm.items():
            shape = getattr(val, 'shape', 'unknown')
            print(f"  {key:<35} shape={shape}")
        print()

    # ── 4. var columns ────────────────────────────────────────────────────────
    print(sep)
    print("VAR COLUMNS  (gene metadata)")
    print(sep)
    if adata.var.empty:
        print("  (no var columns)")
    else:
        for col in adata.var.columns:
            series  = adata.var[col].astype(str)
            n_uniq  = series.nunique()
            example = series.iloc[0] if len(series) > 0 else ''
            print(f"  {col:<35} {str(adata.var[col].dtype):<12} {n_uniq:>9}  {example[:50]}")
    print()

    # ── 5. Overlap check against gene-name-like values ────────────────────────
    print(sep)
    print("OVERLAP CHECK  (which obs columns contain gene-symbol-like values)")
    print(sep)

    # Use var_names as the reference set of gene symbols in this dataset
    var_genes = set(adata.var_names.astype(str))

    for col in adata.obs.columns:
        try:
            obs_vals = set(adata.obs[col].astype(str).unique())
            overlap  = obs_vals & var_genes
            if len(overlap) > 5:
                print(f"  {col:<35} {len(overlap):>5} values overlap with var_names  "
                      f"(e.g. {', '.join(list(overlap)[:4])})")
        except Exception:
            pass
    print()

    # ── 6. Guide-coordinate parse attempt ─────────────────────────────────────
    print(sep)
    print("GUIDE COORDINATE PARSE  (attempt to extract gene info from identifiers)")
    print(sep)

    # Find first high-cardinality string obs column (likely the guide column)
    guide_col = None
    for col in adata.obs.columns:
        n_uniq = adata.obs[col].nunique()
        if n_uniq > 5000:
            sample = str(adata.obs[col].iloc[0])
            if 'chr' in sample.lower() or '_' in sample:
                guide_col = col
                break

    if guide_col:
        print(f"  High-cardinality column: '{guide_col}'")
        print(f"  Sample values:")
        for v in adata.obs[guide_col].drop_duplicates().iloc[:8]:
            print(f"    {v}")

        # Check if any part of split values matches known gene symbols
        print(f"\n  Attempting to find gene-name fragments in values...")
        n_tested = 0
        gene_hits = {}
        for val in adata.obs[guide_col].astype(str).drop_duplicates().iloc[:500]:
            parts = val.replace('_top_two', '').replace('chr', ' chr').split()
            for part in parts:
                part_clean = part.strip('_').upper()
                if part_clean in var_genes:
                    gene_hits[part_clean] = gene_hits.get(part_clean, 0) + 1
            n_tested += 1
        if gene_hits:
            print(f"  Gene name fragments found in {n_tested} tested values:")
            for g, cnt in sorted(gene_hits.items(), key=lambda x: -x[1])[:10]:
                print(f"    {g}: found in {cnt} identifiers")
        else:
            print(f"  No gene name fragments matched var_names in {n_tested} tested values.")
            print(f"  The identifiers likely use genomic coordinates, not HGNC symbols.")
            print(f"  You may need a guide-library annotation file to map coordinates → genes.")
    else:
        print("  No high-cardinality string column with guide-like format found.")

    print()
    print(sep)
    print("RECOMMENDATION")
    print(sep)
    if low_card_cols:
        print(f"  Columns with 2–5000 unique values (likely candidates for gene targets):")
        for col in low_card_cols:
            n = adata.obs[col].nunique()
            ex = str(adata.obs[col].iloc[0])[:40]
            print(f"    --perturbation_col {col:<25} ({n} unique, e.g. '{ex}')")
    else:
        print("  No column with 2–5000 unique values found.")
        print("  The dataset may require guide-to-gene mapping.")
        print("  Check adata.uns for a guide annotation table.")

    del adata
    gc.collect()


if __name__ == '__main__':
    main()