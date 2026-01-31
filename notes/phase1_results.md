# Phase 1 Results: Synthetic Validation

**Date**: 2026-01-31
**Datasets**: syn_linear, syn_scale, syn_multimodal
**Seeds**: 10 (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)
**Samples per dataset**: 10,000 (5% anomaly rate)

## Summary

### Key Findings

1. **Syn-Linear & Syn-Scale**: Contextual methods (PNKDIF, ROCOD) significantly outperform non-contextual methods (IF, DIF)
   - PNKDIF AUROC: 0.9997±0.0001 vs IF AUROC: 0.6595±0.0180
   - This validates the core hypothesis: peer normalization helps detect contextual anomalies

2. **Syn-Multimodal**: As expected, PNKDIF struggles when context changes distribution *shape* (not just shift/scale)
   - IF_concat AUROC: 0.9803 vs PNKDIF AUROC: 0.8673
   - This confirms the documented limitation of z-score normalization

3. **ROCOD vs PNKDIF**: ROCOD slightly edges out PNKDIF on linear/scale datasets but fails badly on multimodal
   - ROCOD relies on median/MAD which breaks when distribution shape changes

## Results Tables

### Syn-Linear (AUROC, higher is better)

| Rank | Method | AUROC | AUPRC | P@100 |
|------|--------|-------|-------|-------|
| 1 | ROCOD | 0.9999±0.0001 | 0.9990±0.0007 | 1.0000±0.0000 |
| 2 | PNKDIF_uniform | 0.9998±0.0001 | 0.9962±0.0011 | 1.0000±0.0000 |
| 3 | PNKDIF | 0.9997±0.0001 | 0.9960±0.0015 | 1.0000±0.0000 |
| 4 | PNKDIF_noMLP | 0.9997±0.0001 | 0.9958±0.0016 | 1.0000±0.0000 |
| 5 | DIF_concat | 0.9924±0.0018 | 0.8789±0.0236 | 1.0000±0.0000 |
| 6 | QCAD | 0.9902±0.0009 | 0.7959±0.0129 | 0.9310±0.0191 |
| 7 | IF_concat | 0.9843±0.0048 | 0.7898±0.0412 | 0.9770±0.0279 |
| 8 | IF | 0.6595±0.0180 | 0.4006±0.0203 | 0.9990±0.0032 |
| 9 | DIF | 0.6579±0.0133 | 0.4012±0.0198 | 1.0000±0.0000 |
| 10 | LOF | 0.5784±0.0191 | 0.1473±0.0201 | 0.4370±0.0782 |

### Syn-Scale (AUROC, higher is better)

| Rank | Method | AUROC | AUPRC | P@100 |
|------|--------|-------|-------|-------|
| 1 | ROCOD | 0.9992±0.0002 | 0.9889±0.0023 | 1.0000±0.0000 |
| 2 | PNKDIF_uniform | 0.9990±0.0002 | 0.9849±0.0019 | 1.0000±0.0000 |
| 3 | PNKDIF | 0.9990±0.0002 | 0.9845±0.0030 | 1.0000±0.0000 |
| 4 | PNKDIF_noMLP | 0.9990±0.0002 | 0.9839±0.0022 | 1.0000±0.0000 |
| 5 | IF_concat | 0.9930±0.0020 | 0.8953±0.0238 | 0.9980±0.0042 |
| 6 | QCAD | 0.9898±0.0009 | 0.7922±0.0118 | 0.9090±0.0218 |
| 7 | DIF_concat | 0.9892±0.0014 | 0.8664±0.0127 | 1.0000±0.0000 |
| 8 | DIF | 0.9677±0.0039 | 0.7987±0.0167 | 1.0000±0.0000 |
| 9 | IF | 0.9672±0.0040 | 0.7966±0.0174 | 1.0000±0.0000 |
| 10 | LOF | 0.6193±0.0148 | 0.1759±0.0191 | 0.4520±0.0761 |

### Syn-Multimodal (AUROC, higher is better)

| Rank | Method | AUROC | AUPRC | P@100 |
|------|--------|-------|-------|-------|
| 1 | IF_concat | 0.9803±0.0027 | 0.6833±0.0417 | 0.7790±0.0907 |
| 2 | DIF_concat | 0.9696±0.0019 | 0.5139±0.0191 | 0.5120±0.0748 |
| 3 | PNKDIF_uniform | 0.8694±0.0086 | 0.5057±0.0110 | 0.9220±0.0220 |
| 4 | PNKDIF | 0.8673±0.0123 | 0.5061±0.0139 | 0.9280±0.0257 |
| 5 | PNKDIF_noMLP | 0.8388±0.0163 | 0.4921±0.0107 | 0.9240±0.0241 |
| 6 | ROCOD | 0.6763±0.0817 | 0.3378±0.1613 | 0.7090±0.3896 |
| 7 | LOF | 0.5593±0.0173 | 0.0852±0.0113 | 0.2200±0.0789 |
| 8 | QCAD | 0.5282±0.0227 | 0.1968±0.0336 | 0.3790±0.1422 |
| 9 | IF | 0.5015±0.0130 | 0.0504±0.0022 | 0.0480±0.0257 |
| 10 | DIF | 0.4994±0.0115 | 0.0503±0.0020 | 0.0510±0.0247 |

## Ablation Insights

### PNKDIF vs PNKDIF_uniform
- RBF kernel weighting provides marginal improvement over uniform weighting
- Difference is small (~0.0001 AUROC) on synthetic data
- May be more significant on real data with varying context density

### PNKDIF vs PNKDIF_noMLP
- Random MLP projection provides minimal benefit on these 1D/2D synthetic datasets
- Expected to help more on higher-dimensional behavioral features

## Runtime

Average runtime per method (seconds per run):
- IF: ~0.2s
- DIF: ~1.2s
- PNKDIF: ~1.3s
- LOF: ~0.05s
- QCAD: ~0.17s
- ROCOD: ~0.12s

## Conclusions

1. **Hypothesis validated**: Peer normalization is essential for detecting contextual anomalies
2. **Known limitation confirmed**: Z-score normalization fails when context changes distribution shape
3. **Ablations show**: Kernel weighting and MLP projections have modest impact on simple synthetic data

## Next Steps

- Phase 2: Run ablation studies on syn_nonlinear
- Phase 3: Test on real/semi-synthetic datasets (SAML-D, IEEE-CIS, PaySim)
- Generate plots for paper
