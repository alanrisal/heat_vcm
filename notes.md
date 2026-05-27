# Notes on RHDP Model

A compilation of errors, processes and fixes made in the development of the model

## NMF Matrix Creation 

Switched matrix compuations to GPU, converged at 260 iterations for NMF. 

50 programs was not unique enough for genetic markers, switched to k = 90, programs are representative of human enriched genetic clusters

Pathway enrichment works by 

03:26:23 | INFO     | ============================================================
03:26:23 | INFO     |   COMPONENT 1: Gene Program Extraction  (K=90)
03:26:23 | INFO     | ============================================================
03:26:23 | INFO     | STEPS 1-3 — Loading from preprocessing checkpoint (skipping data load)...
03:26:34 | INFO     | RAM usage  [checkpoint loaded — X=(42878, 5000)]: 1.70 GB
03:26:34 | INFO     | Preprocessed matrix loaded: (42878, 5000)  (cells × HVGs)
03:26:34 | INFO     | Expression matrix ready: (42878, 5000)  (cells × HVGs)
03:26:34 | INFO     | STEP 4 — Fitting NMF  (K=90, alpha_H=0.1, tol=0.001, max_iter=1000)
03:26:39 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:26:39 | INFO     | Initialising W, H via nndsvda...
03:26:42 | INFO     | nndsvda init done in 3.6s.
03:26:43 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:26:43 | INFO     |   iter   20/1000  recon_err=6029.2319  rel_Δ=nan  elapsed=0.2s
03:26:44 | INFO     |   iter  200/1000  recon_err=5633.2798  rel_Δ=1.44e-03  elapsed=1.2s
03:26:44 | INFO     | NMF converged at iter 260  (rel_change=8.80e-04 < tol=0.001)
03:26:44 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5615.7778, converged=True.
03:26:44 | INFO     | RAM usage  [after NMF fit]: 2.25 GB
03:26:49 | INFO     | NMF results saved to 'outputs/programs/'
03:26:49 | INFO     | RAM usage  [after save]: 2.27 GB
03:26:49 | INFO     | STEP 5 — Running verification checks
03:26:49 | INFO     |   [1/4] Reconstruction quality...
03:26:57 | INFO     |   [2/4] Program sparsity...
03:26:57 | INFO     |   [3/4] Program uniqueness...
03:26:57 | INFO     |   [4/4] NMF stability (5 runs, max_iter=1000)...
03:26:57 | INFO     | Running NMF 5× with K=90 to assess stability...
03:26:57 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:26:57 | INFO     | Initialising W, H via nndsvda...
03:27:01 | INFO     | nndsvda init done in 3.9s.
03:27:01 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:27:02 | INFO     |   iter   20/1000  recon_err=6028.6982  rel_Δ=nan  elapsed=0.1s
03:27:03 | INFO     |   iter  200/1000  recon_err=5632.4146  rel_Δ=1.45e-03  elapsed=1.1s
03:27:03 | INFO     | NMF converged at iter 260  (rel_change=8.82e-04 < tol=0.001)
03:27:03 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5614.8496, converged=True.
03:27:03 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:27:03 | INFO     | Initialising W, H via nndsvda...
03:27:07 | INFO     | nndsvda init done in 4.3s.
03:27:07 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:27:08 | INFO     |   iter   20/1000  recon_err=6030.5933  rel_Δ=nan  elapsed=0.1s
03:27:09 | INFO     |   iter  200/1000  recon_err=5635.2241  rel_Δ=1.46e-03  elapsed=1.1s
03:27:09 | INFO     | NMF converged at iter 260  (rel_change=9.02e-04 < tol=0.001)
03:27:09 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5617.3452, converged=True.
03:27:09 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:27:09 | INFO     | Initialising W, H via nndsvda...
03:27:13 | INFO     | nndsvda init done in 4.3s.
03:27:13 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:27:14 | INFO     |   iter   20/1000  recon_err=6029.0269  rel_Δ=nan  elapsed=0.1s
03:27:15 | INFO     |   iter  200/1000  recon_err=5633.5986  rel_Δ=1.46e-03  elapsed=1.1s
03:27:15 | INFO     | NMF converged at iter 260  (rel_change=8.88e-04 < tol=0.001)
03:27:15 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5615.9331, converged=True.
03:27:15 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:27:15 | INFO     | Initialising W, H via nndsvda...
03:27:19 | INFO     | nndsvda init done in 3.9s.
03:27:19 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:27:19 | INFO     |   iter   20/1000  recon_err=6029.9395  rel_Δ=nan  elapsed=0.1s
03:27:20 | INFO     |   iter  200/1000  recon_err=5634.9844  rel_Δ=1.44e-03  elapsed=1.1s
03:27:21 | INFO     | NMF converged at iter 260  (rel_change=8.86e-04 < tol=0.001)
03:27:21 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5617.3789, converged=True.
03:27:21 | INFO     | Fitting NMF: K=90, shape=(42878, 5000), alpha_H=0.1, tol=0.001, max_iter=1000
03:27:21 | INFO     | Initialising W, H via nndsvda...
03:27:25 | INFO     | nndsvda init done in 4.1s.
03:27:25 | INFO     | GPU detected and verified: NVIDIA RTX PRO 4500 Blackwell — using CUDA.
03:27:25 | INFO     |   iter   20/1000  recon_err=6030.4453  rel_Δ=nan  elapsed=0.1s
03:27:26 | INFO     |   iter  200/1000  recon_err=5633.5933  rel_Δ=1.48e-03  elapsed=1.1s
03:27:26 | INFO     | NMF converged at iter 260  (rel_change=9.01e-04 < tol=0.001)
03:27:26 | INFO     | NMF done: 260 iters, 1.5s, recon_err=5615.6372, converged=True.
03:27:26 | INFO     | Stability score: 0.847 (acceptable)
03:27:26 | INFO     | STEP 6 — Pathway enrichment (requires internet)...
03:30:05 | INFO     | Enrichment results saved to 'outputs/programs/pathway_enrichment.csv'
03:30:05 | INFO     | STEP 7 — Generating report and verification figure

==============================================================
  GENE PROGRAM VERIFICATION REPORT
  K = 90 programs
==============================================================

[1] RECONSTRUCTION QUALITY
  Relative Frobenius error           : 0.4466  (< 0.5 acceptable)
  Mean per-gene R²                   : 0.0906
  Median per-gene R²                 : 0.0288  (> 0.4 good)
  Genes with R² > 0.5                : 2.2%

[2] PROGRAM SPARSITY
  Mean Gini coefficient              : 0.8861  (> 0.7 good)
  Mean effective genes/prog          : 731.8  (< 200 good)
  Top-10 genes share                 : 8.2%
  Top-50 genes share                 : 24.8%

[3] PROGRAM UNIQUENESS
  Mean inter-program cosine sim      : 0.3112  (< 0.3 good)
  Max inter-program cosine sim       : 0.7346
  Near-duplicate pairs (>0.8)        : 0  (0 is ideal)

[4] NMF STABILITY
  Mean stability score               : 0.8470
  Assessment                    : ACCEPTABLE

[5] PATHWAY ENRICHMENT
  Top enriched term per program (first 15 programs):
    Prog  0: Parkinson disease                                 adj.p=2.5e-28
    Prog  1: Hydrogen Peroxide Catabolic Process (GO:0042744)  adj.p=3.3e-09
    Prog  2: Hypoxia                                           adj.p=8.3e-10
    Prog  3: G2-M Checkpoint                                   adj.p=3.3e-67
    Prog  4: G2-M Checkpoint                                   adj.p=3.3e-59
    Prog  5: mTORC1 Signaling                                  adj.p=6.2e-18
    Prog  6: Myc Targets V1                                    adj.p=4.3e-11
    Prog  7: Myc Targets V1                                    adj.p=2.2e-05
    Prog  8: Prion disease                                     adj.p=1.1e-09
    Prog  9: heme Metabolism                                   adj.p=4.0e-19
    Prog 10: Diabetic cardiomyopathy                           adj.p=2.3e-09
    Prog 11: Myc Targets V2                                    adj.p=1.1e-05
    Prog 12: Parkinson disease                                 adj.p=1.1e-12
    Prog 13: Hydrogen Peroxide Catabolic Process (GO:0042744)  adj.p=4.0e-09
    Prog 14: Myc Targets V1                                    adj.p=1.8e-06

==============================================================

03:30:06 | INFO     | Verification figure saved to 'outputs/programs/figures/verification_K90.png'
03:30:06 | INFO     | STEP 8 — Pass / fail evaluation
03:30:06 | INFO     |
03:30:06 | INFO     | ✓  ALL CHECKS PASSED.
03:30:06 | INFO     |    K=90 gene programs are ready for Component 2.
03:30:06 | INFO     |    Results saved to: outputs/programs/




Component 2 

/workspace/heat_vcm/rhdp# python scripts/02_compute_diffusion.py
04:37:43 | INFO     | ============================================================
04:37:43 | INFO     |   COMPONENT 2: Graph Heat Diffusion Encoding
04:37:43 | INFO     |   β=0.1  min_score=700
04:37:43 | INFO     | ============================================================
04:37:43 | INFO     | STEP 1 — Loading Component 1 outputs
04:37:43 | INFO     | Loaded Component 1 outputs: H_nmf=(90, 5000), 5000 HVG genes.
04:37:43 | INFO     | RAM usage  [after loading H and gene names]: 0.32 GB
04:37:43 | INFO     | STEP 2 — Using HVG genes as perturbation targets
04:37:43 | INFO     |   5,000 genes → one p_g vector each. Guide-to-gene mapping for the actual expression data is handled separately in Component 3 (see scripts/build_guide_map.py).
04:37:43 | INFO     | RAM usage  [perturbation target list set]: 0.32 GB
04:37:43 | INFO     | STEP 3 — Building regulatory graph  (min_score=700)
04:37:43 | INFO     |   Cache hit: 9606.protein.links.v12.0.txt.gz
04:37:43 | INFO     |   Cache hit: 9606.protein.info.v12.0.txt.gz
04:37:43 | INFO     | Loading STRING-DB gene name map...
04:37:43 | INFO     |   Loaded 19,699 ENSP → gene name mappings.
04:37:43 | INFO     | Filtering STRING-DB edges  (min_score=700, HVG genes=5,000)...
04:37:52 | INFO     |   Scanned 13,715,404 edges. Kept 18,680 within HVG set at score >= 700.
04:37:52 | INFO     | Adjacency matrix: 5000×5000 sparse.  3,034 / 5,000 HVG genes are connected in the graph.
04:37:52 | INFO     | Laplacian computed: 5000×5000, nnz=42,360, density=0.1694%.
04:37:52 | INFO     | RAM usage  [after graph construction]: 0.33 GB
04:37:52 | INFO     | STEP 5 — Computing heat diffusion  (β=0.1)
04:37:53 | INFO     | Seed matrix: 5000 perturbations — direct=5000, neighbour=0, missing=0 (100.0% represented)
04:37:53 | INFO     | Running heat diffusion  (β=0.1)...
04:37:55 | INFO     | Heat diffusion (GPU eigen path) on cuda...
04:37:56 | INFO     |   GPU diffusion complete.
04:37:56 | INFO     | Projecting diffusion profiles into program space...
04:37:56 | INFO     | P_matrix shape: (5000, 90)  (n_perturbations × K_programs)
04:37:56 | INFO     | RAM usage  [after heat diffusion]: 1.43 GB
04:37:56 | INFO     | STEP 6 — Saving outputs
04:37:56 | INFO     | Component 2 outputs saved to 'outputs/diffusion/'
04:37:56 | INFO     |   P_matrix shape: (5000, 90)
04:37:56 | INFO     | STEP 7 — Verification checks
04:37:56 | INFO     | Concentration check: 5000 direct genes. Median self-rank=0, frac in top-50=100.00%
04:37:56 | INFO     | Program signal: median CV=1.984, frac zero-vector=0.00%
04:37:56 | INFO     | STEP 8 — Report and figure

==============================================================
  COMPONENT 2 VERIFICATION REPORT
  β = 0.1
==============================================================

[1] COVERAGE
  Total perturbations                   : 5000
  Direct (gene in HVG)                  : 5000
  Neighbour seed                        : 0
  Missing (zero p_g)                    : 0
  Fraction represented                  : 100.0%  (>= 60% to pass)
  Status                                : ✓  PASS

[2] DIFFUSION CONCENTRATION
  Genes checked                         : 5000
  Median self-rank                      : 0  (lower is better)
  frac_self_in_top_50                   : 100.0%  (>= 50% to pass)
  Status                                : ✓  PASS

[3] GRAPH STATISTICS
  HVG genes in graph                    : 3034
  Total HVG genes                       : 5000
  Graph edges retained                  : 18680

[4] PROGRAM SIGNAL
  Perturbations checked                 : 5000
  Median CV of p_g                      : 1.9840  (> 0.3 to pass)
  Fraction zero-vector                  : 0.0%  (< 50% to pass)
  Status                                : ✓  PASS

==============================================================

04:37:56 | INFO     | Component 2 verification figure saved: outputs/diffusion/figures/component2_beta0.1.png
04:37:56 | INFO     |
04:37:56 | INFO     | ✓  ALL CHECKS PASSED.
04:37:56 | INFO     |    P_matrix is ready for Component 3.
04:37:56 | INFO     |    Shape: (5000, 90)
04:37:56 | INFO     |    Saved to: outputs/diffusion/
04:37:56 | INFO     |