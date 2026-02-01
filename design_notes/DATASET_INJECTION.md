# Dataset Injection Strategies for Contextual Anomaly Detection

This document describes the contextual anomaly injection strategies used to evaluate PNKDIF on real-world datasets.

---

## Why Injection?

Real-world anomaly datasets often contain **global anomalies** (unusual everywhere) rather than **contextual anomalies** (unusual for their specific context). To properly evaluate contextual anomaly detection methods, we inject synthetic contextual anomalies using behavior swapping strategies.

**Key principle:** Swap behavior between different context groups so that the injected values are **normal globally** but **unusual for their context**.

---

## General Injection Framework

```python
def inject_contextual_anomalies(context, behavior, labels,
                                 source_mask, target_mask,
                                 injection_rate=0.02, seed=42):
    """
    Inject contextual anomalies by swapping behavior between groups.

    Args:
        source_mask: Boolean mask for source group (will receive target behavior)
        target_mask: Boolean mask for target group (provides behavior statistics)
        injection_rate: Fraction of source samples to inject
    """
    rng = np.random.RandomState(seed)

    source_idx = np.where(source_mask & (labels == 0))[0]  # Normal source samples
    target_idx = np.where(target_mask)[0]

    n_inject = int(len(labels) * injection_rate)
    inject_idx = rng.choice(source_idx, min(n_inject, len(source_idx)), replace=False)

    # Calculate target behavior statistics
    target_mean = behavior[target_idx].mean(axis=0)
    target_std = behavior[target_idx].std(axis=0)

    # Replace source behavior with target-like behavior
    for i in inject_idx:
        behavior[i] = target_mean + rng.randn(behavior.shape[1]) * target_std * 0.3
        labels[i] = 1

    return behavior, labels
```

---

## Dataset-Specific Strategies

### 1. SAML-D (Anti-Money Laundering)

**Dataset:** Account-level aggregated transaction data (292,715 accounts)

| Feature Type | Features |
|--------------|----------|
| **Context** | Distribution profiles: `loc_{country}`, `curr_{currency}`, `ptype_{type}` (38 features) |
| **Behavior** | Transaction statistics: tx_count, tx_sum, tx_mean, tx_std, tx_max, tx_min, tx_range, unique_receivers, active_days, avg_tx_per_day, avg_receivers_per_day (11 features) |

**Injection Strategy: Domestic → Cross-border Behavior Swap**

| Account Type | Identification | Typical Behavior |
|--------------|----------------|------------------|
| Domestic | Low entropy in location distribution (>80% single country) | Lower amounts, fewer receivers |
| Cross-border | High entropy in location distribution | Higher amounts, more receivers |

**Rationale:** A domestic account with cross-border-like transaction patterns is suspicious in AML.

```python
# Identify by location entropy
loc_entropy = -sum(p * log(p)) for location probabilities
domestic_mask = (loc_entropy <= 25th percentile) & (labels == 0)
crossborder_mask = loc_entropy >= 75th percentile
```

**Result:**
- Source: 71,789 domestic accounts
- Injected: 5,854 anomalies (2% rate)
- Total: 4,950 real + 5,854 injected = 10,804 anomalies

---

### 2. Adult Dataset (UCI Census)

**Dataset:** Census income data (30,162 samples)

| Feature Type | Features |
|--------------|----------|
| **Context** | workclass, education, marital_status, relationship, race, sex, native_country |
| **Behavior** | age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week |

**Injection Strategy: Low → High Capital Gains Swap**

| Group | Identification | Behavior Pattern |
|-------|----------------|------------------|
| Low capital | capital_gain < 100 | Low financial activity |
| High capital | capital_gain > 10,000 | High financial activity |

**Rationale:** A low-income demographic with high capital gains is contextually unusual.

---

### 3. Bank Dataset (UCI Marketing)

**Dataset:** Bank marketing campaign data (30,488 samples)

| Feature Type | Features |
|--------------|----------|
| **Context** | age, job, marital, education |
| **Behavior** | duration, campaign, pdays, previous, balance, ... |

**Injection Strategy: Young → Senior Behavior Swap**

| Group | Identification | Behavior Pattern |
|-------|----------------|------------------|
| Young | age < 30 | Short calls, frequent campaigns |
| Senior | age > 55 | Long calls, infrequent campaigns |

**Rationale:** A young customer with senior-like engagement patterns is contextually unusual.

---

## Evaluation Framework

For each dataset, we evaluate on three conditions:

| Condition | Description | Tests |
|-----------|-------------|-------|
| **Pure Synthetic** | Synthetic datasets with controlled anomalies | Method's theoretical capability |
| **Real Only** | Original dataset labels | Real-world performance |
| **Real + Injection** | Original + injected contextual anomalies | Contextual detection capability |

This allows us to:
1. Demonstrate PNKDIF works on true contextual anomalies (synthetic)
2. Compare honestly on real data where anomalies may be global
3. Show contextual detection capability on realistic data structure

---

## Injection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `injection_rate` | 0.02 (2%) | Fraction of data to inject |
| `noise_scale` | 0.3 | Noise added as fraction of target std |
| `seed` | 42 | Random seed for reproducibility |

---

## Code Location

- `src/data/fraud_datasets.py`: `load_saml_with_injection()`
- `src/data/real_datasets.py`: `load_adult()`, `load_bank()` with `anomaly_type='synthetic_shift'`
