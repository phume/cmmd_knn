# Experimental Plan: PNKDIF Evaluation

## 1. Methods Under Comparison

### Baselines
| Method | Context-Aware | Non-Linear | Training-Free | Implementation |
|--------|---------------|------------|---------------|----------------|
| **IF** | No | No | Yes | sklearn |
| **IF-concat** | Partial* | No | Yes | sklearn (concat C+Y) |
| **DIF** | No | Yes | Yes | PyTorch (original paper) |
| **DIF-concat** | Partial* | Yes | Yes | PyTorch (concat C+Y) |
| **LOF** | No | No | Yes | sklearn |
| **QCAD** | Yes | No | Yes | Custom impl |
| **ROCOD** | Yes | No | Yes | Custom impl |

*Partial = context used but not for normalization

### Proposed Method Variants
| Variant | Description |
|---------|-------------|
| **PNKDIF** | Full method (K-NN + RBF kernel + MLP + IF) |
| **PNKDIF-uniform** | Uniform peer weights instead of RBF kernel |
| **PNKDIF-noMLP** | Direct IF on z-scores (no random projection) |
| **PNKDIF-single** | Single projection (M=1) instead of ensemble |
| **PNKDIF-global** | Global normalization instead of peer-based |

---

## 2. Datasets

### 2.1 Synthetic Datasets (Controlled Ground Truth)

#### Syn-Linear: Linear Context-Behavior Relationship
```
Context: c ~ Uniform(0, 10)
Normal:  y = 2*c + 5 + N(0, 1)
Anomaly: y = 2*c + 5 + N(0, 1) + Uniform(5, 10) * sign
```
- **Purpose**: Validate that peer normalization works for simple shift relationships
- **Expected**: All contextual methods should excel; IF/DIF should struggle

#### Syn-Scale: Context Affects Variance
```
Context: c ~ Uniform(1, 10)
Normal:  y ~ N(50, c²)  # variance scales with context
Anomaly: y ~ N(50, c²) + 5*c  # anomaly magnitude scales too
```
- **Purpose**: Test if z-score normalization handles heteroscedasticity
- **Expected**: PNKDIF should handle this; QCAD may struggle with variance modeling

#### Syn-Nonlinear: Non-Linear Anomaly Boundary
```
Context: c ~ Uniform(0, 10), 2D
Normal:  y ~ N(0, I) within radius 2 of context-dependent center
Anomaly: Points on curved manifold in Y-space
```
- **Purpose**: Test if random MLP projections capture non-linear patterns
- **Expected**: PNKDIF and DIF should outperform linear methods

#### Syn-Multimodal: Context-Dependent Distribution Shape
```
Context: c ∈ {0, 1} (binary)
c=0: y ~ N(0, 1)  (unimodal)
c=1: y ~ 0.5*N(-3, 0.5) + 0.5*N(3, 0.5)  (bimodal)
Anomaly: Points at distribution modes for wrong context
```
- **Purpose**: Test limitation - PNKDIF assumes shift/scale, not shape change
- **Expected**: PNKDIF should fail; this is a known limitation

#### Syn-HighDimContext: Curse of Dimensionality
```
Context: c ∈ R^d, d ∈ {5, 10, 20, 50, 100}
Sparse context with only 2-3 informative dimensions
```
- **Purpose**: Test degradation as context dimension increases
- **Expected**: All K-NN methods should degrade; measure rate of degradation

#### Syn-Cluster: Clustered Context Space
```
Context: 5 Gaussian clusters with different behavior distributions
Anomaly: Normal behavior for wrong cluster
```
- **Purpose**: Test if K-NN finds meaningful peers in clustered spaces
- **Expected**: PNKDIF should work if K captures within-cluster peers

### 2.2 Semi-Synthetic Datasets (Real Context, Synthetic Anomalies)

#### Credit-Semi
- **Source**: UCI Credit Card dataset or similar
- **Context**: Customer demographics (age, income, account type, etc.)
- **Behavior**: Inject synthetic transaction anomalies
- **Anomaly injection**:
  - Type A: Global outliers (extreme values)
  - Type B: Contextual outliers (normal globally, anomalous for peer group)
- **Purpose**: Test on realistic context structure with controlled anomalies

### 2.3 Real Datasets with Context/Behavior Split

#### Financial / Fraud Detection

| Dataset | N | Features | Anomaly % | Source |
|---------|---|----------|-----------|--------|
| **SAML-D** | ~500K | 12 | varies | [Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) |
| **IEEE-CIS Fraud** | 590K | 394 | 3.5% | [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) |
| **PaySim** | 6.3M | 11 | 0.13% | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Credit Card** | 284K | 31 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

##### SAML-D (Synthetic AML Dataset)
- **Source**: [Nature Scientific Data](https://www.nature.com/articles/s41597-023-02569-2), [Kaggle](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- **Context features**: Customer type, account age, geographic location, high-risk country flag, payment type
- **Behavioral features**: Transaction amount, frequency, velocity, time patterns
- **Feature engineering needed**:
  - Aggregate transaction-level data to customer-level
  - Compute velocity features (txn count per day/week)
  - Create peer groups by customer segment
- **Anomaly**: 28 money laundering typologies

##### IEEE-CIS Fraud Detection
- **Source**: [Kaggle Competition](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Context features**: Card info (card1-6), address (addr1-2), device info, ProductCD
- **Behavioral features**: TransactionAmt, D1-D15 (time deltas), C1-C14 (counts), V1-V339
- **Feature engineering needed**:
  - Create entity fingerprint (UUID) from card + address + time
  - Aggregate V-features by category
  - Handle 50%+ missing values in identity table
- **Anomaly**: isFraud label (3.5%)

##### PaySim (Mobile Money)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), [GitHub](https://github.com/EdgarLopezPhD/PaySim)
- **Context features**: Transaction type (CASH_IN, CASH_OUT, TRANSFER, PAYMENT, DEBIT), nameOrig, nameDest
- **Behavioral features**: Amount, oldbalanceOrg, newbalanceOrg, oldbalanceDest, newbalanceDest
- **Feature engineering needed**:
  - Aggregate to account-level
  - Compute balance change ratios
  - Create sender/receiver profiles
- **Anomaly**: isFraud (0.13%), isFlaggedFraud (transfers > 200K)

#### Tabular / Census

| Dataset | N | Features | Anomaly % | Source |
|---------|---|----------|-----------|--------|
| **Adult Census** | 48K | 14 | 24%* | [UCI](https://archive.ics.uci.edu/dataset/2/adult) |
| **Bank Marketing** | 41K | 20 | 11.3%* | [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) |

*Note: These are classification datasets; minority class treated as "anomaly" for benchmarking

##### Adult Census Income
- **Source**: [UCI](https://archive.ics.uci.edu/dataset/2/adult)
- **Context features**: age, workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Behavioral features**: hours-per-week, capital-gain, capital-loss, fnlwgt
- **Anomaly definition options**:
  1. Original: income >50K as minority class (24%)
  2. Synthetic injection: Add outliers to behavioral features for random subset
  3. Contextual: Flag high capital-gain for young/low-education individuals

##### Bank Marketing
- **Source**: [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Context features**: age, job, marital, education, default, housing, loan, contact, month, day_of_week
- **Behavioral features**: duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
- **Anomaly definition options**:
  1. Original: subscription (y=yes) as minority class (11.3%)
  2. Synthetic injection: Add outliers to economic indicators
  3. Contextual: Flag unusual call durations for customer segments

#### Healthcare

| Dataset | N | Features | Anomaly % | Source |
|---------|---|----------|-----------|--------|
| **Cardio** | 1831 | 11 | 9.6% | [ODDS](http://odds.cs.stonybrook.edu/) |
| **Thyroid** | 3772 | 21 | 2.5% | [ODDS](http://odds.cs.stonybrook.edu/) |
| **Healthcare Providers** | varies | varies | varies | [Kaggle](https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data) |

##### Healthcare Provider Anomaly
- **Context features**: Provider specialty, location, patient demographics
- **Behavioral features**: Billing amounts, procedure counts, claim patterns
- **Anomaly**: Unusual billing patterns for provider type

#### ADBench Datasets (57 datasets)
- **Source**: [GitHub](https://github.com/Minqi824/ADBench), [NeurIPS 2022](https://arxiv.org/abs/2206.09426)
- **Includes**: 48 real-world + 9 synthetic datasets across healthcare, finance, NLP, CV
- **Use case**: Comprehensive benchmark for overall method comparison
- **Note**: Need to define context/behavior split per dataset

### 2.4 Synthetic Anomaly Injection Protocol

For datasets without natural anomalies or to control anomaly types:

#### Injection Types

| Type | Description | Implementation |
|------|-------------|----------------|
| **Global** | Extreme values regardless of context | `y_anom = y + k * global_std` |
| **Contextual-Shift** | Normal globally, abnormal for peer group | `y_anom = y + k * peer_std` (for random subset) |
| **Contextual-Swap** | Swap behavior between different contexts | `y_anom[ctx=A] = y_normal[ctx=B]` |
| **Cluster-Based** | Anomalies at cluster boundaries | Points between cluster centers |

#### Injection Protocol
```python
def inject_anomalies(X, y, context, injection_type, anomaly_rate=0.05):
    n_anomalies = int(len(X) * anomaly_rate)
    anomaly_idx = np.random.choice(len(X), n_anomalies, replace=False)

    if injection_type == "global":
        y[anomaly_idx] += np.random.choice([-1, 1]) * 5 * y.std()
    elif injection_type == "contextual_shift":
        for idx in anomaly_idx:
            peer_mask = get_peers(context, idx)
            peer_std = y[peer_mask].std()
            y[idx] += np.random.choice([-1, 1]) * 3 * peer_std
    elif injection_type == "contextual_swap":
        # Swap with different context cluster
        ...

    labels = np.zeros(len(X))
    labels[anomaly_idx] = 1
    return y, labels
```

#### Datasets with Injection
| Base Dataset | Injection Types | Expected Outcome |
|--------------|-----------------|------------------|
| Adult Census | Global, Contextual-Shift | PNKDIF better on Contextual-Shift |
| Bank Marketing | Global, Contextual-Shift | PNKDIF better on Contextual-Shift |
| Cardio | Global, Contextual-Shift | Compare with original labels |

### 2.5 Dataset Summary Table

| Dataset | Domain | N | d_c | d_y | Anomaly % | Context/Behavior Split |
|---------|--------|---|-----|-----|-----------|------------------------|
| Syn-Linear | Synthetic | 10K | 1 | 1 | 5% | Explicit |
| Syn-Scale | Synthetic | 10K | 1 | 1 | 5% | Explicit |
| Syn-Nonlinear | Synthetic | 10K | 2 | 2 | 5% | Explicit |
| Syn-Multimodal | Synthetic | 10K | 1 | 1 | 5% | Explicit |
| SAML-D | Finance/AML | ~500K | 5 | 4 | varies | Domain-defined |
| IEEE-CIS | E-commerce | 590K | ~50 | ~340 | 3.5% | Domain-defined |
| PaySim | Mobile Money | 6.3M | 3 | 4 | 0.13% | Domain-defined |
| Adult | Census | 48K | 9 | 4 | 5%* | Domain-defined |
| Bank | Marketing | 41K | 10 | 9 | 5%* | Domain-defined |
| Cardio | Healthcare | 1.8K | 5 | 6 | 9.6% | Domain-defined |

*With synthetic injection

**Challenge**: Most benchmark datasets don't have explicit context/behavior separation. We need to:
1. Define which features are context vs. behavior based on domain knowledge
2. Validate that the split makes semantic sense
3. Run sensitivity analysis on different splits
4. Acknowledge this as a limitation

---

## 3. Evaluation Metrics

### Primary Metrics
| Metric | Description | Use Case |
|--------|-------------|----------|
| **AUROC** | Area under ROC curve | Overall ranking quality |
| **AUPRC** | Area under PR curve | Imbalanced data (preferred) |
| **P@k** | Precision at top-k | Practical: "top 100 alerts" |
| **R@k** | Recall at top-k | Practical: "catch rate in top 100" |

### Secondary Metrics
| Metric | Description | Use Case |
|--------|-------------|----------|
| **Adjusted AUROC** | AUROC adjusted for class imbalance | Cross-dataset comparison |
| **Average Precision** | Mean of P@k for all k | Summary of ranking |
| **Time (fit)** | Wall-clock training time | Scalability |
| **Time (score)** | Wall-clock scoring time | Practical deployment |
| **Memory** | Peak memory usage | Resource constraints |

### Statistical Testing
- Report mean ± std over 10 random seeds
- Paired Wilcoxon signed-rank test for pairwise comparison
- Bonferroni correction for multiple comparisons
- Critical difference diagram for overall ranking

---

## 4. Ablation Studies

### 4.1 Component Ablations
| Ablation | What it tests |
|----------|---------------|
| Remove RBF kernel (uniform weights) | Value of soft peer weighting |
| Remove MLP projection | Value of non-linear transformation |
| Remove peer normalization | Value of contextual preprocessing |
| Single projection (M=1) | Value of ensemble diversity |
| Remove IF (use distance-based score) | Value of isolation-based scoring |

### 4.2 Hyperparameter Sensitivity

#### K (number of neighbors)
- Range: [10, 25, 50, 100, 200, 500]
- Expected: Performance plateau after sufficient K; degradation if K too small (noisy) or too large (loses locality)

#### γ (kernel bandwidth)
- Range: [0.1×median, 0.5×median, median, 2×median, 10×median]
- Expected: Robust around median heuristic; extreme values degrade

#### M (number of projections)
- Range: [1, 2, 4, 6, 8, 10, 15, 20]
- Expected: Diminishing returns after M≈6-10

#### d_h (hidden dimension)
- Range: [16, 32, 64, 128, 256, 512]
- Expected: Sweet spot around 64-256; too small loses expressiveness, too large may overfit to noise

#### T (IF trees)
- Range: [50, 100, 200, 300, 500]
- Expected: Standard IF behavior; 100-200 usually sufficient

### 4.3 Scaling Studies

#### Dataset Size
- N ∈ [1K, 5K, 10K, 50K, 100K, 500K, 1M]
- Measure: Wall-clock time, memory, AUROC stability
- Expected: O(N log N) scaling

#### Context Dimensionality
- d_c ∈ [2, 5, 10, 20, 50, 100]
- Measure: AUROC degradation, peer quality (neighbor distance distribution)
- Expected: Degradation due to curse of dimensionality

#### Behavior Dimensionality
- d_y ∈ [1, 2, 5, 10, 20, 50]
- Measure: AUROC, MLP projection effectiveness
- Expected: More dimensions may help MLP find patterns; too many may dilute signal

---

## 5. Expected Failure Modes

### 5.1 Where PNKDIF Should Fail

| Failure Mode | Why | How to Detect |
|--------------|-----|---------------|
| **High-dim context** | K-NN breaks down, unstable peer stats | AUROC drops as d_c increases |
| **Non-shift/scale context effects** | Z-score assumes linear relationship | Syn-Multimodal dataset |
| **Insufficient peers** | Small N or sparse context regions | High variance in peer stats |
| **Adversarial context** | Malicious manipulation of context | Security evaluation (out of scope?) |
| **Homogeneous peers** | σ=0 leads to unstable z-scores | Check σ distribution |

### 5.2 Where Baselines Should Fail

| Method | Expected Failure | Test Dataset |
|--------|------------------|--------------|
| IF/DIF | Contextual anomalies (normal globally) | Syn-Linear, Syn-Scale |
| IF-concat | Context-behavior interaction ignored | Same as above |
| QCAD/ROCOD | Non-linear anomaly boundaries | Syn-Nonlinear |
| LOF | High dimensions, doesn't separate C/Y | All contextual datasets |

### 5.3 Tie Cases
- When anomalies are both global AND contextual: all methods may perform similarly
- When context is uninformative: PNKDIF may not improve over DIF

---

## 6. Reproducibility Considerations

### 6.1 Random Seeds
```python
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
```
- Use same seeds across all methods
- Report: mean, std, min, max
- Store per-seed results for statistical testing

### 6.2 Data Splits
- Synthetic: Generate fresh per seed
- Real: Fixed train/test split, vary only method randomness
- No validation set needed (training-free methods)

### 6.3 Implementation Details
- **PNKDIF**: Our implementation
- **IF**: sklearn.ensemble.IsolationForest (default params except n_estimators)
- **DIF**: Official implementation or faithful reproduction
- **QCAD/ROCOD**: Reproduce from paper description
- **LOF**: sklearn.neighbors.LocalOutlierFactor

### 6.4 Hardware
- Report: CPU model, cores, RAM, GPU (if used)
- Timing: Average of 3 runs after 1 warmup run

### 6.5 Code & Data Release
- GitHub repo with:
  - All method implementations
  - Synthetic data generators
  - Evaluation scripts
  - Config files for all experiments
  - requirements.txt / environment.yml
- Data: Synthetic generators included; real data links + preprocessing scripts

---

## 7. Experiment Execution Plan

### Phase 1: Synthetic Validation
1. Implement synthetic data generators
2. Run all methods on Syn-Linear, Syn-Scale
3. Verify PNKDIF beats non-contextual methods
4. Verify on Syn-Multimodal that PNKDIF fails as expected

### Phase 2: Ablation Studies
1. Run all ablations on Syn-Linear and Syn-Nonlinear
2. Identify which components contribute most
3. Generate sensitivity plots for hyperparameters

### Phase 3: Scaling Studies
1. Run timing experiments across dataset sizes
2. Verify O(N log N) scaling
3. Identify memory bottlenecks

### Phase 4: Real/Semi-Synthetic Data
1. Prepare Credit-Semi with injected anomalies
2. Run on available real benchmarks with context/behavior split
3. Document context/behavior split rationale

### Phase 5: Statistical Analysis
1. Aggregate all results
2. Run statistical tests
3. Generate tables and figures for paper

---

## 8. Expected Results Summary

| Dataset Type | Winner | Why |
|--------------|--------|-----|
| Syn-Linear | PNKDIF ≈ QCAD > DIF > IF | Contextual methods handle shift |
| Syn-Scale | PNKDIF > QCAD > DIF > IF | PNKDIF handles variance scaling |
| Syn-Nonlinear | PNKDIF > DIF > QCAD > IF | Non-linear boundaries needed |
| Syn-Multimodal | DIF/IF > PNKDIF | PNKDIF limitation: shape changes |
| Syn-HighDimContext | All degrade | Curse of dimensionality |
| Real (if context helps) | PNKDIF ≥ DIF | Depends on data structure |

---

## 9. Open Questions

1. **Context/behavior split on real data**: How to systematically define this? Domain expertise required.

2. **Baseline implementations**: Use official code where available, or reimplement for fair comparison?

3. **Computational budget**: How many total experiment hours? Prioritize which experiments if limited?

4. **Streaming/online setting**: Out of scope for initial paper, but worth mentioning as future work?

5. **Adversarial robustness**: Should we test manipulation of context features?
