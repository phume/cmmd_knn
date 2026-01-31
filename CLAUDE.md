# Project Knowledge & Preferences

This file captures insights, preferences, and notes learned from working on this project. Use this as context for future sessions.

## Workflow Preferences

### Git & Version Control
- **Commit frequently**: Commit and push after completing any meaningful work (file creation, updates, task completion)
- **Commit style**: Use descriptive messages like "Add method formalization with pseudocode and complexity analysis"

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

## Future Tasks

- [ ] Implement PNKDIF in Python (src/models/)
- [ ] Set up baseline implementations
- [ ] Run experiments and fill in result tables
- [ ] Add scalability figure
- [ ] Finalize paper for submission
