# Phase 2 & 3 Results

**Date**: 2026-01-31

## Phase 2: Ablation Studies

### Syn-Nonlinear Dataset

Tests non-linear anomaly boundaries (anomalies on curved manifolds).

| Rank | Method | AUROC |
|------|--------|-------|
| 1 | PNKDIF | 1.0000±0.0000 |
| 2 | PNKDIF_uniform | 1.0000±0.0000 |
| 3 | PNKDIF_noMLP | 1.0000±0.0000 |
| 4 | ROCOD | 0.9999±0.0000 |
| 5 | LOF | 0.9994±0.0010 |
| 6 | QCAD | 0.9617±0.0044 |
| 7 | IF_concat | 0.9183±0.0104 |
| 8 | DIF_concat | 0.9026±0.0092 |
| 9 | IF | 0.8199±0.0147 |
| 10 | DIF | 0.8145±0.0152 |

**Key Insight**: PNKDIF achieves perfect detection on non-linear boundaries. The combination of peer normalization and the 2D context space makes anomalies extremely easy to isolate.

### Hyperparameter Sensitivity

#### K (Number of Neighbors)

| K | syn_linear AUROC | syn_nonlinear AUROC |
|---|------------------|---------------------|
| 10 | 0.9973±0.0010 | 0.9833±0.0068 |
| 25 | 0.9993±0.0003 | 0.9988±0.0007 |
| 50 | 0.9997±0.0001 | 0.9998±0.0001 |
| 100 | 0.9997±0.0001 | 1.0000±0.0000 |
| 200 | 0.9995±0.0001 | 0.9999±0.0001 |
| 500 | 0.9976±0.0005 | 0.9996±0.0003 |

**Insight**: K=50-200 is the sweet spot. Too small (K<25) = noisy statistics. Too large (K>500) = loses locality.

#### M (Number of Projections)

| M | syn_linear AUROC | syn_nonlinear AUROC |
|---|------------------|---------------------|
| 1 | 0.9995±0.0003 | 1.0000±0.0000 |
| 2 | 0.9997±0.0001 | 1.0000±0.0000 |
| 4 | 0.9997±0.0001 | 1.0000±0.0000 |
| 6 | 0.9997±0.0001 | 1.0000±0.0000 |
| 8 | 0.9997±0.0000 | 1.0000±0.0000 |
| 10 | 0.9997±0.0001 | 1.0000±0.0000 |

**Insight**: M has minimal impact on these datasets. Even M=1 works well. Ensemble provides marginal stability improvement.

#### d_h (Hidden Dimension)

| d_h | syn_linear AUROC | syn_nonlinear AUROC |
|-----|------------------|---------------------|
| 16 | 0.9995±0.0003 | 1.0000±0.0000 |
| 32 | 0.9996±0.0002 | 1.0000±0.0000 |
| 64 | 0.9997±0.0001 | 1.0000±0.0000 |
| 128 | 0.9997±0.0001 | 1.0000±0.0000 |
| 256 | 0.9997±0.0001 | 1.0000±0.0000 |

**Insight**: d_h has minimal impact. The random projections work across a wide range of dimensions.

### Ablation Conclusions

1. **Peer normalization is the key component** - explains most of the performance gain
2. **RBF kernel weighting**: Marginal improvement over uniform
3. **Random MLP projections**: Minimal impact on low-dimensional synthetic data (may help more on real high-dim data)
4. **Hyperparameters are robust**: Wide ranges work well, no careful tuning needed

---

## Phase 3: Scaling Studies

### Runtime vs Dataset Size

| N | IF | DIF | PNKDIF | QCAD | ROCOD |
|---|---|-----|--------|------|-------|
| 1,000 | 0.12s | 0.71s | 0.74s | 0.02s | 0.01s |
| 2,000 | 0.18s | 1.11s | 1.12s | 0.05s | 0.02s |
| 5,000 | 0.19s | 0.99s | 1.08s | 0.12s | 0.08s |
| 10,000 | 0.20s | 1.17s | 1.24s | 0.17s | 0.10s |
| 20,000 | 0.27s | 1.70s | 1.97s | 0.40s | 0.26s |
| 50,000 | 0.46s | 3.31s | 4.18s | 1.05s | 0.66s |

### Scaling Analysis

- **IF**: O(N) - linear scaling, very fast
- **ROCOD**: O(N log N) - fast K-NN based
- **QCAD**: O(N log N) - similar to ROCOD
- **DIF**: O(N) for projections, O(N log N) for IF
- **PNKDIF**: O(N log N) - dominated by K-NN + IF ensemble

### Scaling Conclusions

1. PNKDIF is ~10x slower than IF but maintains ~50% better AUROC on contextual data
2. PNKDIF overhead vs DIF is ~25% (K-NN step)
3. All methods scale reasonably to 50K samples
4. For very large N (>100K), consider approximate K-NN or sampling

---

## Combined Summary

| Phase | Key Finding |
|-------|-------------|
| Phase 1 | Contextual methods (PNKDIF, ROCOD) beat non-contextual (IF, DIF) on shift/scale data |
| Phase 1 | PNKDIF fails on multimodal (known limitation) |
| Phase 2 | PNKDIF achieves perfect AUROC on non-linear boundaries |
| Phase 2 | Hyperparameters are robust (K=50-200, M=1-10, d_h=16-256 all work) |
| Phase 3 | PNKDIF scales O(N log N), ~4s for 50K samples |

## Files Generated

```
results/
├── phase1_raw.csv, phase1_summary.csv   # Phase 1
├── phase2_raw.csv, phase2_summary.csv   # Phase 2
├── phase3_raw.csv, phase3_summary.csv   # Phase 3
└── phase*_state.json                    # Checkpoints
```
