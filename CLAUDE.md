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
- [x] Results in results/phase*.csv, analysis in notes/

## Key Results

| Dataset | PNKDIF AUROC | IF AUROC | Winner |
|---------|--------------|----------|--------|
| syn_linear | 0.9997 | 0.6595 | PNKDIF (+51%) |
| syn_scale | 0.9990 | 0.9672 | PNKDIF (+3%) |
| syn_nonlinear | 1.0000 | 0.8199 | PNKDIF (+22%) |
| syn_multimodal | 0.8673 | 0.5015 | PNKDIF (+73%)* |

*Note: IF_concat (0.9803) beats PNKDIF on multimodal - known limitation

## Future Tasks

- [ ] Phase 4: Real/semi-synthetic datasets (SAML-D, IEEE-CIS, PaySim)
- [ ] Generate plots for paper (PR curves, bar plots, scaling figure)
- [ ] Update paper with experimental results
- [ ] Finalize paper for submission

## Experiment Runner Notes

- `scripts/run_phase1.py`: Non-interactive runner with checkpoint/resume
- Results saved incrementally to `results/phase1_raw.csv`
- Checkpoint in `results/phase1_state.json` for crash recovery
- Logs to `logs/phase1.log`
- Errors to `results/phase1_errors.csv`
