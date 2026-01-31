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

## Project-Specific Notes

### PNKDIF Method
- Core idea: Peer-normalized features + frozen random MLP projections + Isolation Forest
- Key novelty claims:
  1. First training-free contextual anomaly detector with non-linear decision boundaries
  2. Kernel-weighted peer normalization (soft K-NN instead of hard cutoff)
- Main limitations: Curse of dimensionality in context space, assumes simple shift/scale relationship

### Paper Structure
- IEEE conference format (IEEEtran document class)
- Sections: Introduction, Related Work, Methodology, Experiments, Discussion, Conclusion
- Experiments section has placeholder tables (marked with TODO) for actual results

### Key References
- Deep Isolation Forest (Xu et al., 2023) - inspiration for random MLP projections
- QCAD/ROCOD (Liang et al.) - existing contextual anomaly detection baselines
- Random Fourier Features (Rahimi & Recht, 2007) - theoretical basis for frozen projections

## File Organization

```
CDIF/
├── paper/           # LaTeX source files
│   ├── main.tex     # Main document
│   ├── sections/    # Individual sections
│   └── references.bib
├── design_notes/    # Method formalization, literature comparison
├── src/             # Source code (to be implemented)
│   ├── models/      # PNKDIF implementation
│   ├── baselines/   # Comparison methods
│   ├── data/        # Data loading utilities
│   ├── evaluation/  # Metrics and evaluation
│   └── experiments/ # Experiment scripts
├── configs/         # Configuration files
├── scripts/         # Utility scripts
└── related_work/    # Reference papers
```

## Completed Tasks

- [x] Implement PNKDIF in Python (src/models/pnkdif.py)
- [x] Set up baseline implementations (src/models/baselines.py)
- [x] Phase 1: Synthetic validation (syn_linear, syn_scale, syn_multimodal)
- [x] Phase 2: Ablation studies (syn_nonlinear, hyperparameter sensitivity K/M/d_h)
- [x] Phase 3: Scaling studies (N=1K to 50K)
- [x] Phase 4: Real/semi-synthetic datasets (Adult, Bank, Cardio)
- [x] Results in results/phase*.csv, analysis in notes/

## Key Results

### Synthetic Data (Phases 1-3)

| Dataset | PNKDIF AUROC | IF AUROC | Winner |
|---------|--------------|----------|--------|
| syn_linear | 0.9997 | 0.6595 | PNKDIF (+51%) |
| syn_scale | 0.9990 | 0.9672 | PNKDIF (+3%) |
| syn_nonlinear | 1.0000 | 0.8199 | PNKDIF (+22%) |
| syn_multimodal | 0.8673 | 0.5015 | PNKDIF (+73%)* |

*Note: IF_concat (0.9803) beats PNKDIF on multimodal - known limitation

### Real Data (Phase 4)

| Dataset | Best Method | AUROC | Notes |
|---------|-------------|-------|-------|
| adult_shift | PNKDIF_noMLP | 0.996 | All methods ~0.99 |
| adult_swap | PNKDIF | 0.558 | Near-random for all |
| bank_shift | IF/PNKDIF_noMLP | 0.999 | All methods ~0.99 |
| bank_swap | PNKDIF | 0.538 | Near-random for all |
| cardio | IF | 0.949 | PNKDIF at 0.780 |

**Key insight**: PNKDIF_noMLP often best - random MLP may hurt on low-dim real data. Swap injection is fundamentally hard (behavior IS normal, just from different context).

## Phase 4 No-Download Results (COMPLETED)

Ran on Syn-HighDimContext, Syn-Cluster, Thyroid with 10 seeds x 12 methods = 360 runs.

### Syn-HighDimContext (20D context, only 2 informative)
| Method | AUROC |
|--------|-------|
| PNKDIF_uniform | 0.925 |
| PNKDIF | 0.925 |
| ROCOD | 0.865 |
| IF | 0.841 |
| IF_concat | 0.593 |

**Key insight**: PNKDIF handles high-dim context; concat methods fail.

### Syn-Cluster (5 clusters with cluster-specific behavior)
| Method | AUROC |
|--------|-------|
| PNKDIF_uniform | 0.954 |
| PNKDIF | 0.953 |
| ROCOD | 0.947 |
| IF/PNKDIF_global | 0.50 |

**Key insight**: Context-ignoring methods fail completely.

### Thyroid (ODDS - non-contextual anomalies)
| Method | AUROC |
|--------|-------|
| IF_concat | 0.977 |
| PNKDIF_global | 0.955 |
| IF | 0.955 |
| PNKDIF | 0.703 |

**Key insight**: Thyroid has global anomalies, not contextual. PNKDIF_global outperforms peer-normalized variants.

## Items Implemented

### Datasets (loaders in src/data/):
- **SAML-D** - `fraud_datasets.py`
- **IEEE-CIS Fraud** - `fraud_datasets.py`
- **PaySim** - `fraud_datasets.py`
- **Credit Card Fraud** - `fraud_datasets.py`
- **Thyroid (ODDS)** - `fraud_datasets.py` (auto-downloads)
- **Syn-HighDimContext** - `synthetic.py`
- **Syn-Cluster** - `synthetic.py`

### Method variants (in src/models/pnkdif.py):
- **PNKDIF_single** (M=1)
- **PNKDIF_global** (global normalization)

### Other:
- 10 seeds: [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
- Feature caching: fraud_datasets.py saves to data/cache/

## Future Tasks

- [ ] **Download fraud datasets** (user needs to download from Kaggle):
  - SAML-D: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
  - IEEE-CIS: https://www.kaggle.com/competitions/ieee-fraud-detection
  - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1
  - Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- [ ] Run Phase 4 Extended: `python scripts/run_phase4_extended.py`
- [ ] Update paper experiments section with Phase 4 no-download results
- [ ] Add statistical testing (Wilcoxon, critical difference diagrams)
- [ ] γ (kernel bandwidth) sensitivity analysis
- [ ] Finalize paper for submission

## Experiment Runner Notes

- `scripts/run_phase1.py`: Non-interactive runner with checkpoint/resume
- Results saved incrementally to `results/phase1_raw.csv`
- Checkpoint in `results/phase1_state.json` for crash recovery
- Logs to `logs/phase1.log`
- Errors to `results/phase1_errors.csv`
