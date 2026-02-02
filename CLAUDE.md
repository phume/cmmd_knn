# Project Knowledge & Preferences

This file captures insights, preferences, and notes learned from working on this project. Use this as context for future sessions.

## Workflow Preferences

### Git & Version Control
- **Commit frequently**: Commit and push after completing any meaningful work (file creation, updates, task completion)
- **Commit style**: Use descriptive messages like "Add method formalization with pseudocode and complexity analysis"
- **No co-author line**: Do NOT include "Co-Authored-By: Claude..." in commit messages

### Task Order
- **LaTeX before code**: Work on paper/documentation first, then implement code
- **Don't jump ahead**: Wait for explicit instruction before moving to next phase (e.g., don't start coding when asked to work on LaTeX)

### Communication Style
- **Ask before assuming**: If unclear about task scope, clarify rather than doing extra work
- **Be concise**: Don't over-explain, get to the point

## Current Status (2026-02-01)

### Critical Issue: PNKIF vs ROCOD Novelty

**Problem**: PNKIF is essentially ROCOD with two swaps:
- Peer weighting: uniform → RBF kernel
- Scoring: max(z-score) → Isolation Forest

**Latest benchmark (all datasets):**

| Dataset | IF | ROCOD | PNKIF | Winner |
|---------|-----|-------|-------|--------|
| Syn-Linear | 0.616 | **1.000** | **1.000** | Tie |
| Syn-Cluster | 0.492 | **0.929** | 0.920 | ROCOD +1% |
| Syn-HighDim | 0.835 | 0.848 | **0.906** | PNKIF +6% |
| Syn-Nonlinear | 0.815 | **1.000** | **1.000** | Tie |
| Adult-shift | 0.994 | 0.942 | **0.997** | PNKIF +0% |
| Adult-swap | 0.503 | 0.526 | **0.549** | PNKIF +2% |
| Bank-shift | 0.999 | 0.936 | **0.999** | PNKIF +0% |
| Bank-swap | 0.486 | 0.517 | **0.528** | PNKIF +1% |
| Cardio | **0.947** | 0.761 | 0.765 | IF +18% |
| Thyroid | **0.965** | 0.572 | 0.655 | IF +31% |

**Verdict**: PNKIF wins 6/10 but margins are small (0-6%). Only meaningful win is Syn-HighDim (+6%).

### ConQuest Paper Review

Read "Context discovery for anomaly detection" (Calikus et al., 2025). Key takeaways:

1. **SNN (Shared Nearest Neighbors) distance** - More robust to high-dim context
2. **CoNN distance** = Euclidean(behavior) / SNN(context) - Jointly considers both spaces
3. **Multi-context ensemble** - Run on different context subsets, average scores
4. **Three objectives for context quality**:
   - Context-behavior dependency (distance correlation)
   - Minimum redundancy between contexts
   - Maximum discrimination (kurtosis of scores)

### New Methods Implemented (from ConQuest)

Added to `src/models/pnkdif.py`:
- **PNKIF_SNN**: Uses SNN-based peer weighting instead of RBF kernel
- **MCAF**: Multi-Context Anomaly Factor using CoNN distance
- **MultiContextPNKIF**: Ensemble over multiple context subsets
- **compute_context_quality()**: Diagnostic for context relevance

**Speed comparison (N=5000):**
| Method | AUROC | Time | vs ROCOD |
|--------|-------|------|----------|
| ROCOD | 0.929 | 0.2s | 1.0x |
| PNKIF | 0.924 | 0.4s | 2x slower |
| PNKIF_SNN | 0.928 | 5.1s | **25x slower** |
| MCAF | 0.922 | 14.1s | **70x slower** |

**Problem**: SNN/MCAF are too slow without significant accuracy gain.

### Paper Review Feedback

Did mock peer review. Key concerns:
1. **Novelty over ROCOD is incremental** - need stronger differentiation
2. **Context feature selection unaddressed** - the hard problem
3. **Swap injection results missing** - should include negative results
4. **Real dataset results favor IF** - PNKIF only wins with injection
5. **Missing statistical tests** - need Wilcoxon, CD diagrams
6. **Missing citations** - ConQuest, CAD (Song et al., 2007)

## Options to Improve Novelty

### Option A: Hybrid ROCOD + PNKIF
Combine best of both:
- Robust statistics (median/MAD) from ROCOD
- RBF kernel weighting from PNKIF
- Isolation Forest scoring from PNKIF

### Option B: Focus on Diagnostic Framework
De-emphasize PNKIF as "new method", emphasize:
- Comparing PNKIF vs IF reveals if context matters
- This diagnostic is the contribution, not the algorithm

### Option C: Add Context Discovery
Implement ConQuest-style automatic context selection as preprocessing.

### Option D: Multi-Context Ensemble for AML
For AML specifically, ensemble over:
- Geographic context (country, domestic/cross-border)
- Customer context (type, segment, account age)
- Temporal context (time patterns)

## File Organization

```
CDIF/
├── paper/           # LaTeX source files
│   ├── main.tex     # Main document
│   ├── sections/    # Individual sections
│   └── references.bib
├── design_notes/    # Method formalization, literature comparison
├── src/
│   ├── models/
│   │   ├── pnkdif.py      # PNKIF, PNKDIF, PNKIF_SNN, MCAF, etc.
│   │   └── baselines.py   # IF, ROCOD, QCAD, LOF, etc.
│   ├── data/
│   │   ├── synthetic.py   # Synthetic dataset generators
│   │   ├── real_datasets.py  # Adult, Bank, Cardio loaders
│   │   └── fraud_datasets.py # SAML-D, Thyroid, etc.
│   └── evaluation/
├── related_work/
│   └── conquest Context discovery for anomaly detection.pdf
└── results/
```

## Key References

- **ROCOD**: Liang & Parthasarathy, "Robust Contextual Outlier Detection" (CIKM 2016)
- **QCAD**: Liang et al., conditional anomaly detection
- **ConQuest**: Calikus et al., "Context discovery for anomaly detection" (2025)
- **Deep Isolation Forest**: Xu et al. (2023)
- **CAD**: Song et al. (2007) - original contextual anomaly paper

## Next Steps

1. [ ] **Decide on differentiation strategy** (Option A/B/C/D above)
2. [ ] **If Option A**: Implement hybrid ROCOD+PNKIF method
3. [ ] **If Option B**: Reframe paper around diagnostic framework
4. [ ] **Run statistical tests** (Wilcoxon signed-rank)
5. [ ] **Update paper** with honest positioning
6. [ ] **Add missing citations** (ConQuest, CAD)

## Notes

- MLP projections (PNKDIF) provide marginal benefit (<2%), PNKIF (no MLP) is recommended
- Swap injection is fundamentally hard - all methods score ~0.52 (near random)
- Global anomalies (Cardio, Thyroid): IF wins by large margin
- Context selection is the real problem - all methods assume it's given
