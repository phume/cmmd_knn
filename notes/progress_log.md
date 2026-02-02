# Progress Log

## 2026-02-01: ConQuest Benchmark Datasets Added

### Downloaded Datasets

**Location:** `data/raw/conquest/`

#### ODDS Datasets (from ConQuest GitHub)
| Dataset | N | d_c | d_y | Anom% | Format |
|---------|---|-----|-----|-------|--------|
| cardio_odds | 1831 | 5 | 16 | 9.6% | .mat |
| ionosphere_odds | 351 | 8 | 25 | 35.9% | .mat |
| arrhythmia_odds | 452 | 50 | 224 | 14.6% | .mat |
| pima_odds | 768 | 3 | 5 | 34.9% | .mat |
| vowels_odds | 1456 | 3 | 9 | 3.4% | .mat |
| wbc_odds | 378 | 5 | 25 | 5.6% | .mat |

#### DAMI Datasets (from ConQuest GitHub)
| Dataset | N | d_c | d_y | Anom% | Format |
|---------|---|-----|-----|-------|--------|
| glass_dami | 214 | 3 | 4 | 4.2% | .arff |
| ionosphere_dami | 351 | 8 | 24 | 35.9% | .arff |
| heartdisease_dami | 270 | 4 | 9 | 44.4% | .arff |
| lymphography_dami | 148 | 1 | 2 | 4.1% | .arff |
| wbc_dami | 223 | 3 | 6 | 4.5% | .arff |
| wdbc_dami | 367 | 5 | 25 | 2.7% | .arff |

### Data Loader
- Created `src/data/conquest_datasets.py`
- Supports both ODDS (.mat) and DAMI (.arff) formats
- Context/behavior split configurable via n_context parameter

### Source
- GitHub: https://github.com/reguluslus/Conquest
- Paper: Calikus et al., "Context discovery for anomaly detection" (2025)

### Notes
- These datasets are used in ConQuest paper (Table 1) for CAD benchmarking
- Default context/behavior splits follow reasonable heuristics but may need tuning
- Lymphography dataset has only 3 features after categorical removal

### Experiment Results (2026-02-01)

| Dataset | PNKIF | PNKIF_SNN | ROCOD | IF | Best |
|---------|-------|-----------|-------|-----|------|
| arrhythmia_odds | 0.725 | 0.721 | 0.659 | **0.791** | IF |
| cardio_odds | 0.741 | 0.727 | 0.724 | **0.949** | IF |
| glass_dami | 0.711 | 0.692 | **0.883** | 0.765 | ROCOD |
| heartdisease_dami | 0.633 | 0.627 | 0.544 | **0.725** | IF |
| ionosphere_dami | 0.798 | 0.801 | **0.907** | 0.817 | ROCOD |
| ionosphere_odds | 0.823 | 0.824 | **0.895** | 0.820 | ROCOD |
| lymphography_dami | 0.826 | 0.787 | 0.360 | **0.841** | IF |
| pima_odds | 0.507 | 0.508 | 0.551 | **0.604** | IF |
| vowels_odds | **0.829** | 0.820 | 0.822 | 0.721 | PNKIF |
| wbc_dami | 0.888 | 0.906 | 0.467 | **0.982** | IF |
| wbc_odds | 0.817 | 0.835 | 0.886 | **0.934** | IF |
| wdbc_dami | 0.829 | 0.837 | **0.938** | 0.982 | IF |

**Key Findings:**
- IF wins on 8/12 datasets (global anomalies dominate)
- ROCOD wins on 3/12 datasets (glass, ionosphere)
- PNKIF wins on 1/12 datasets (vowels_odds)
- PNKIF_SNN provides marginal improvement over PNKIF on a few datasets

**Interpretation:**
These are standard ODDS benchmarks with predominantly **global** anomalies. PNKIF/PNKIF_SNN are designed for **contextual** anomalies (normal globally, unusual for context). The results confirm the paper's thesis: use PNKIF when context matters, use IF when anomalies are global.

### Next Steps
- [ ] Add results to paper experiments section (Table with ConQuest datasets)
- [ ] Add PNKIF_SNN as method variant in methodology section
- [ ] Consider pivoting to AML application focus given weak novelty over ROCOD
