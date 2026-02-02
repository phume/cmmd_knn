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

---

## 2026-02-01: Decision to Pivot to Option B (AML Application Focus)

### Why We're Pivoting

**The core problem: Weak novelty over existing methods**

1. **PNKIF vs ROCOD**: Our method is essentially ROCOD with two swaps:
   - RBF kernel weighting (instead of uniform K-NN)
   - Isolation Forest scoring (instead of max z-score)

   Experiments show these changes provide **marginal or no benefit**. ROCOD actually beats PNKIF on several datasets (Glass, Ionosphere).

2. **PNKIF-SNN didn't help**: Adding SNN weighting from ConQuest provided negligible improvement.

3. **Standard benchmarks favor IF**: On 7/10 ODDS datasets, plain Isolation Forest wins. These datasets have global anomalies where context doesn't matter.

4. **ResidualIF is strong but too simple**: Just regression + IF on residuals. Works great but probably not publishable as a novel method.

5. **No single method dominates**: Different data structures favor different approaches. This is interesting but not a strong novelty claim.

### Why Option B (AML Focus) Makes Sense

1. **SAML-D is our strongest result**: When we inject contextual anomalies (domestic accounts with cross-border behavior), PNKIF achieves 0.948 vs IF's 0.901. This is a meaningful improvement on a real-world inspired scenario.

2. **Domain expertise adds value**: AML/financial crime is a high-impact application area. Explaining WHY context matters (geography, customer type, transaction patterns) adds practical value.

3. **Regulatory relevance**: Financial institutions must explain their detection methods. PNKIF's interpretability (peer comparison) aligns with explainability requirements.

4. **Less competition on method novelty**: An application paper is judged on domain contribution, not algorithmic novelty.

5. **We have the data**: SAML-D is a realistic AML simulation dataset with 293K accounts.

### What Option B Paper Would Look Like

**Title**: "Contextual Anomaly Detection for Anti-Money Laundering: When Geography Matters"

**Contributions**:
1. Demonstrate that AML anomalies are often contextual (normal globally, unusual for customer type/geography)
2. Show that standard IF misses these; PNKIF catches them
3. Provide practical framework for AML practitioners
4. Discuss interpretability and regulatory compliance

**Target Venues**:
- ACM KDD (Applied Data Science track)
- AAAI (AI for Financial Services)
- IEEE Intelligent Systems
- Expert Systems with Applications
- Journal of Financial Crime

### Literature Review (Completed 2026-02-01)

#### 1. ML for AML/Transaction Monitoring

**Key surveys:**
- Chen et al. (2018): "ML techniques for AML solutions in suspicious transaction detection" - foundational survey
- Deep Learning for AML (March 2025): CRP-AML model achieved F1=82.51% on minority class
- Comparative analysis (2025): XGBoost outperforms IF, KNN, RF, SVM on AML data

**Industry state:**
- SAS/ACAMS Survey (Feb 2025): Only 18% have AI/ML in production, 40% no plans
- PwC (2023): 62% use AI/ML for AML, expected 90% by 2025
- ML reduces false positives by up to 40%

**Key methods used:**
- Rule-based (traditional, high false positives)
- Isolation Forest (general anomaly detection)
- Graph-based methods (network analysis)
- Deep learning (emerging but black-box concerns)

**Sources:**
- [Deep Learning for AML](https://arxiv.org/html/2503.10058v1)
- [Transaction monitoring qualitative analysis](https://www.sciencedirect.com/science/article/pii/S0167739X24002607)
- [SAS AML Survey 2025](https://www.sas.com/en_us/news/press-releases/2025/february/anti-money-laundering-survey-ai-machine-learning.html)

#### 2. Contextual Anomaly Detection in Finance

**Anomaly types in AML:**
- **Point anomalies**: Single unusual transaction (large amount)
- **Contextual anomalies**: Unusual only within context (high freq during non-business hours)
- **Collective anomalies**: Pattern of transactions that together form suspicious behavior

**Context variables:**
- Customer type/segment
- Geographic location
- Time of day/week
- Transaction history baseline

**Key insight:** "AI systems understand WHY something looks unusual, not just THAT it does" - contextual reasoning is essential

**Sources:**
- [Anomaly detection for fraud prevention](https://www.fraud.com/post/anomaly-detection)
- [BIS ML framework for anomaly detection](https://www.bis.org/publ/work1188.pdf)

#### 3. Explainable AI (XAI) for AML Compliance

**Regulatory drivers:**
- **FATF**: Requires risk-based approach, transparency in decision-making
- **EU AI Act (2024)**: Mandates transparency in AI systems for financial institutions
- **FinCEN**: Emphasizes auditable, adaptive AML frameworks

**The problem:**
- Deep learning = black box
- Regulators need to understand WHY alerts are generated
- Institutions risk regulatory findings without explainability

**XAI techniques:**
- TreeSHAP, KernelSHAP for feature importance
- Rule extraction from models
- Counterfactual explanations

**PNKIF advantage:** Peer-based comparison is inherently interpretable!
- "This account is flagged because its behavior differs from similar accounts"
- Natural explanation aligned with peer group analysis

**Sources:**
- [XAI in AML - AMLWatcher](https://amlwatcher.com/blog/explainable-ai-in-aml/)
- [SAS XAI for AML whitepaper](https://www.sas.com/en/whitepapers/explainable-artificial-intelligence-for-anti-money-laundering.html)
- [XAI4AML research chair](https://anr.fr/Project-ANR-20-CHIA-0023)

#### 4. SAML-D Dataset

**Citation:** Oztas et al. (2023), "Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset", IEEE ICEBE

**Features:**
- 12 features, 28 typologies (11 normal, 17 suspicious)
- ~0.1% suspicious transactions (realistic imbalance)
- Multiple currencies, geographic locations, high-risk countries
- 15 network structures for graph analysis

**Why synthetic data:**
- Real AML data is confidential (legal/privacy)
- Lack of ground truth labels in real data
- Need diversity of money laundering patterns

**Availability:** Kaggle, GitHub

**Sources:**
- [SAML-D Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- [SAML-D GitHub](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)
- [IEEE paper](https://ieeexplore.ieee.org/document/10356193/)

#### 5. Peer Group Analysis in Finance (KEY!)

**This is directly relevant to PNKIF!**

**Foundational work:**
- Bolton & Hand: "Peer Group Analysis - Local Anomaly Detection in Longitudinal Data"
- Plastic card fraud detection using peer group analysis (Springer 2008)
- Stock fraud detection using peer group analysis

**How it works:**
1. Find peer group (similar objects based on context)
2. Summarize peer behavior over time
3. Compare target to peer group summary
4. Flag deviations

**Strengths:**
- Adaptable to new fraud types
- Detects LOCAL anomalies (contextual)
- "Context anomaly is data points considered abnormal when compared with peer group"

**Limitations:**
- Peer groups may change over time
- Static peer groups decrease detection probability

**PNKIF contribution:** RBF kernel weighting addresses the "hard boundary" problem of traditional peer groups!

**Sources:**
- [Peer Group Analysis seminal paper](https://www.semanticscholar.org/paper/Peer-Group-Analysis-Local-Anomaly-Detection-in-Data-Bolton-Hand/bf7f98eaf32453fab3042c533f0929624faffbf1)
- [Plastic card fraud peer group](https://link.springer.com/article/10.1007/s11634-008-0021-8)
- [Peer group analysis for AML](https://www.solytics-partners.com/resources/case-studies/peer-group-analysis-for-aml-transaction-monitoring)

### Paper Positioning for Option B

**Key angle:** PNKIF is a modern, principled approach to peer group analysis for AML

**Contributions:**
1. **Soft peer groups**: RBF kernel weighting instead of hard K-NN cutoffs
2. **Isolation Forest scoring**: More robust than z-score for multivariate behavior
3. **Training-free**: No risk of overfitting on limited labeled data
4. **Interpretable**: "Account flagged because behavior differs from geographic/customer-type peers"

**Differentiation from prior peer group work:**
- Bolton & Hand used simple statistical summaries
- We use modern ML (Isolation Forest) for scoring
- Kernel weighting is more principled than arbitrary peer selection

**Target story:**
"Domestic accounts exhibiting cross-border transaction patterns" - a clear, domain-relevant contextual anomaly that IF misses but PNKIF catches
