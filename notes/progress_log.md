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
- [x] Add results to paper experiments section (Table with ConQuest datasets)
- [x] Add PNKIF_SNN as method variant in methodology section
- [ ] Consider pivoting to AML application focus given weak novelty over ROCOD

---

## 2026-02-01: Last Sprint - Conditional Isolation Forest Research

### The Problem
- PNKIF is weak over ROCOD (minimal novelty)
- IF is extremely strong on global anomalies
- CVAE models x = f(context, noise) but needs lots of data and can overfit

### Goal
Bring the "contextual conditioning" idea directly into Isolation Forest, without deep learning overhead.

### Research Ideas to Explore

**1. Residual IF (Simple baseline)**
- Train regression: predicted_behavior = f(context)
- Run IF on residuals: behavior - predicted_behavior
- Pros: Simple, interpretable
- Cons: Assumes linear/smooth relationship

**2. Local IF (IF on context-neighbors)**
- For each test point, find K context-neighbors
- Build IF only on those K neighbors' behaviors
- Score = IF anomaly score within local group
- Pros: True local context conditioning
- Cons: Expensive (N separate IF models), small sample for each IF

**3. Context-Weighted IF**
- Modify IF subsample selection to weight by context similarity
- Points with similar context to test point are more likely sampled
- Pros: Single IF, context-aware
- Cons: Requires modifying IF internals

**4. Context-Stratified IF**
- Cluster context space into regions
- Build separate IF per region
- Score using the IF from matching region
- Pros: Simple ensemble approach
- Cons: Hard boundaries between regions

**5. Conditional Splits IF**
- Modify split criterion to consider context
- Split on behavior features but threshold depends on context
- Like decision tree with context as conditioning variable
- Cons: Major algorithmic change

**6. Two-Stage IF**
- Stage 1: IF on context to find "unusual contexts"
- Stage 2: IF on behavior within context groups
- Combine scores

### Literature Review Results

**CADI (SAC 2024)** - "Contextual Anomaly Detection using Isolation Forest"
- Authors: Yepmo, Smits, Lesot, Pivert
- Key idea: Modified IF that distinguishes local vs global anomalies
- Uses density-aware splits and three node types: IN, DN, DLN
- Provides explanations by comparing anomalies to reconstructed clusters
- Follow-up: myCADI (CIKM 2024) adds interactive visualization
- **Limitation**: "Contextual" here means local clusters, NOT conditioning on separate context features

**Survey finding (arXiv 2403.10802)**
- Contextual/conditional IF is **underexplored**
- Most IF extensions focus on streaming, time series, trajectory
- No standard method for x|context conditioning

**Residual-based approach (TDS article)**
- Use Random Forest Regression to predict behavior from context
- Anomalies = points with high residual error (>3 std)
- Simple, interpretable, leverages RF's resistance to overfitting

### Key Insight
**CADI is NOT what we want** - it detects "local anomalies" within clusters found by IF itself, not anomalies conditioned on external context features.

**What we want**: IF that operates on behavior GIVEN context, like CVAE but without deep learning.

### Approaches to Implement

**1. Residual IF** (simplest)
```
predicted_behavior = regression(context)  # RF, linear, etc.
residuals = behavior - predicted_behavior
scores = IsolationForest(residuals)
```

**2. Local IF** (true contextual)
```
for each point i:
    neighbors = KNN(context, K)
    local_IF = IsolationForest(behavior[neighbors])
    scores[i] = local_IF.score(behavior[i])
```

**3. Context-Weighted Subsampling IF**
```
for each tree:
    weights = RBF_kernel(context, reference_point)
    subsample = weighted_sample(data, weights)
    tree = build_isolation_tree(subsample.behavior)
```

### Quick Tests to Run
- [x] Residual IF: linear regression + IF on residuals
- [x] Local IF: K-NN + IF on neighbors only
- [x] Compare on Syn-Cluster (where context matters most)

### Initial Results (2026-02-01)

**Syn-Cluster** (context defines clusters):
| Method | AUROC | Time |
|--------|-------|------|
| IF | 0.468 | 0.12s |
| ROCOD | **0.915** | 0.03s |
| PNKIF | 0.911 | 0.15s |
| ResidualIF_RF | 0.885 | 0.27s |
| ResidualIF_Ridge | 0.650 | 0.13s |
| LocalIF | 0.915 | 103s |

**Syn-HighDim** (20D context, 2 informative):
| Method | AUROC | Time |
|--------|-------|------|
| IF | 0.895 | 0.12s |
| ROCOD | 0.884 | 1.88s |
| PNKIF | 0.939 | 0.18s |
| **ResidualIF_Ridge** | **0.998** | 0.12s |
| ResidualIF_RF | 0.992 | 0.93s |
| LocalIF | 0.952 | 124s |

### Key Insights

1. **ResidualIF_Ridge dominates Syn-HighDim** - Linear regression captures the relationship perfectly. The synthetic data has a linear context→behavior mapping, so residuals isolate true anomalies.

2. **LocalIF is slow but effective** - 1000x slower than PNKIF for similar results. Not practical.

3. **ResidualIF fails on Syn-Cluster with Ridge** - Cluster structure is non-linear, so ridge regression can't capture it. RF version does better.

4. **PNKIF/ROCOD are robust across both** - Peer normalization works regardless of linear vs cluster structure.

### Extended Results (5 datasets)

| Dataset | IF | ROCOD | PNKIF | ResidualIF_Ridge | ResidualIF_RF | Best |
|---------|-----|-------|-------|------------------|---------------|------|
| Syn-Cluster | 0.468 | **0.915** | 0.911 | 0.650 | 0.885 | ROCOD |
| Syn-HighDim | 0.895 | 0.884 | 0.939 | **0.998** | 0.992 | ResidualIF_Ridge |
| Cardio | **0.940** | 0.724 | 0.706 | 0.877 | 0.608 | IF (global) |
| Ionosphere | 0.814 | 0.900 | 0.834 | 0.844 | **0.928** | ResidualIF_RF |
| Vowels | 0.697 | 0.822 | 0.848 | **0.880** | 0.678 | ResidualIF_Ridge |

### Key Findings

1. **No single method wins all** - Different data structures favor different approaches

2. **ResidualIF is competitive** - Wins 3/5 datasets, especially when:
   - Linear context→behavior relationship exists
   - High-dimensional context with informative features

3. **HybridCIF doesn't help** - Combining residual + peer doesn't beat individual methods

4. **Data structure determines best method**:
   - **Linear relationship** → ResidualIF_Ridge
   - **Cluster structure** → ROCOD/PNKIF
   - **Global anomalies** → plain IF

### Potential Paper Contribution

Instead of claiming one method beats all, we could:

1. **Diagnostic framework**: Run multiple methods, disagreement signals data structure
2. **Adaptive selection**: Auto-detect which method suits the data
3. **Ensemble**: Combine methods with confidence weighting

### The Real Question

Is ResidualIF novel enough? It's literally:
- Train regression (context → behavior)
- Run IF on residuals

This is a well-known pattern ("residual analysis"). But combining it with IF for anomaly detection might be underexplored in the CAD literature.

### Next Steps

- [ ] Check if "Residual IF" or similar exists in literature
- [ ] Test on SAML-D with injected anomalies
- [ ] Consider pivoting paper to "diagnostic framework" angle
