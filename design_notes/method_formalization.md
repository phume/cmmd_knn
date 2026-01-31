# Method Formalization: Peer-Normalized Kernel Deep Isolation Forest (PNKDIF)

## 1. Mathematical Notation

### Input Space
- **Dataset**: X = {x_1, ..., x_N} where each x_i = (c_i, y_i)
- **Context features**: c_i ∈ ℝ^{d_c}
- **Behavioral features**: y_i ∈ ℝ^{d_y}

### Hyperparameters
| Symbol | Description | Typical Range |
|--------|-------------|---------------|
| K | Number of nearest neighbors | 50-200 |
| γ | RBF kernel bandwidth | Data-dependent |
| M | Number of random MLP projections | 6-10 |
| d_h | Hidden dimension of each MLP | 64-256 |
| T | Number of trees per Isolation Forest | 100-300 |
| ψ | Subsample size for IF | 256 |
| ε | Floor for numerical stability | 1e-8 |

### Intermediate Quantities
| Symbol | Description | Dimension |
|--------|-------------|-----------|
| N_K(i) | Index set of K nearest neighbors of point i in context space (excluding i) | set of size K |
| w_{ij} | Kernel weight between points i and j | scalar |
| W_i | Sum of kernel weights for point i | scalar |
| μ_i | Weighted peer mean for point i | ℝ^{d_y} |
| σ_i | Weighted peer standard deviation for point i | ℝ^{d_y} |
| z_i | Normalized behavioral features for point i | ℝ^{d_y} |
| h_i^{(m)} | m-th random projection of z_i | ℝ^{d_h} |
| s_i^{(m)} | IF anomaly score on m-th representation | [0, 1] |
| s_i | Final aggregated anomaly score | [0, 100] |

### Random MLP Parameters (frozen at initialization)
- W^{(m)} ∈ ℝ^{d_y × d_h}: weight matrix, sampled from N(0, 2/d_y) (Kaiming init)
- b^{(m)} = 0 ∈ ℝ^{d_h}: bias vector (zero-initialized)

---

## 2. Algorithm Pseudocode

```
Algorithm 1: Peer-Normalized Kernel Deep Isolation Forest (PNKDIF)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:
    X = {(c_i, y_i)}_{i=1}^N    // dataset with context and behavioral features
    K                            // number of neighbors
    γ                            // kernel bandwidth
    M                            // number of random projections
    d_h                          // projection dimension
    T, ψ                         // IF parameters (trees, subsample size)

Output:
    S = {s_i}_{i=1}^N           // anomaly scores in [0, 100]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INITIALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1:  for m = 1 to M do
2:      W^{(m)} ← sample from N(0, 2/d_y)^{d_y × d_h}
3:      b^{(m)} ← 0^{d_h}
4:  end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PEER NORMALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5:  Build spatial index (KD-tree / ball-tree) on {c_i}_{i=1}^N

6:  for i = 1 to N do
7:      // Find K nearest neighbors, EXCLUDING self
8:      N_K(i) ← KNN(c_i, K, exclude={i})
9:
10:     // Compute RBF kernel weights
11:     for j ∈ N_K(i) do
12:         w_{ij} ← exp(-||c_i - c_j||² / (2γ²))
13:     end for
14:     W_i ← Σ_{j ∈ N_K(i)} w_{ij}
15:
16:     // Weighted peer statistics (element-wise)
17:     μ_i ← (1/W_i) · Σ_{j ∈ N_K(i)} w_{ij} · y_j
18:     σ_i ← sqrt((1/W_i) · Σ_{j ∈ N_K(i)} w_{ij} · (y_j - μ_i)²)
19:
20:     // Z-score with numerical stability floor
21:     σ_i ← max(σ_i, ε)
22:     z_i ← (y_i - μ_i) ⊘ σ_i                    // ⊘ = element-wise division
23: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RANDOM PROJECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

24: for m = 1 to M do
25:     for i = 1 to N do
26:         h_i^{(m)} ← LeakyReLU(z_i · W^{(m)} + b^{(m)}, α=0.01)
27:     end for
28: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ISOLATION FOREST SCORING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

29: for m = 1 to M do
30:     IF^{(m)} ← IsolationForest({h_i^{(m)}}_{i=1}^N, n_trees=T, subsample=ψ)
31:     for i = 1 to N do
32:         s_i^{(m)} ← IF^{(m)}.score(h_i^{(m)})   // raw score in [0, 1]
33:     end for
34: end for

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SCORE AGGREGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

35: for i = 1 to N do
36:     s_i^{raw} ← (1/M) · Σ_{m=1}^M s_i^{(m)}
37: end for

38: S^{raw} ← {s_i^{raw}}_{i=1}^N
39: S ← rescale(S^{raw}, target=[0, 100])          // distribution-preserving scaling

40: return S
```

---

## 3. Computational Complexity Analysis

### Per-Component Breakdown

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| KD-tree construction | O(N · d_c · log N) | O(N · d_c) |
| K-NN queries (all N) | O(N · K · log N) | O(N · K) |
| Kernel weights | O(N · K · d_c) | O(N · K) |
| Peer statistics | O(N · K · d_y) | O(N · d_y) |
| Z-score normalization | O(N · d_y) | O(N · d_y) |
| Random projection (M total) | O(N · M · d_y · d_h) | O(N · M · d_h) |
| IF construction (M total) | O(M · T · ψ · log ψ) | O(M · T · ψ) |
| IF scoring (M total) | O(N · M · T · log ψ) | O(1) |

### Total Complexity

**Time**:
```
O(N · log N · d_c)           // KD-tree + queries
+ O(N · K · (d_c + d_y))     // kernel weights + peer stats
+ O(N · M · d_y · d_h)       // random projection
+ O(N · M · T · log ψ)       // IF scoring
```

**Simplified** (assuming K, M, T, d_h are constants w.r.t. N):
```
O(N · log N)   for small d_c, d_y
O(N)           dominant term is linear in N
```

**Space**: O(N · (d_c + d_y + M · d_h) + M · T · ψ)

### Parallelization Opportunities
- K-NN queries: independent per point → parallelize over i
- Random projections: independent per m → parallelize over M
- IF construction: independent per m → parallelize over M
- IF scoring: independent per (i, m) → parallelize over both

---

## 4. Design Rationale

### Why K-NN for peer selection?
- **Non-parametric**: No assumption about context distribution shape
- **Local adaptivity**: Automatically adjusts to varying context density
- **Interpretable**: "Similar entities" is domain-meaningful
- **Alternative rejected**: Radius-based neighbors — peer count varies wildly with local density

### Why RBF kernel weighting (not uniform)?
- **Soft boundaries**: Avoids discontinuity at K-th neighbor cutoff
- **Distance-aware**: Closer peers contribute more to statistics
- **Smooth score surface**: Anomaly score changes continuously as point moves in context space
- **Trade-off**: Adds hyperparameter γ, but K-NN already requires K

### Why z-score normalization?
- **Location-scale removal**: Eliminates additive shift (μ) and multiplicative scaling (σ) effects of context
- **Assumption encoded**: Context affects behavior through shift and scale
- **Interpretable**: z = 2.5 means "2.5 standard deviations from peer average"
- **Limitation**: Cannot capture complex conditional relationships (e.g., context changing distribution shape)
- **Alternative rejected**: Quantile normalization — more robust but loses magnitude information

### Why frozen random MLP?
- **Training-free**: No optimization, no overfitting, no learning rate tuning
- **Non-linear boundaries**: Creates curved decision surfaces in original z-space
- **Theoretical basis**: Random features approximate kernel methods (Rahimi & Recht, 2007)
- **Alternative rejected**: Learned MLP — requires labels or proxy objective; may learn to ignore rare anomalies

### Why multiple MLPs (M > 1)?
- **Diverse views**: Each random initialization emphasizes different feature interactions
- **Robustness**: If one projection collapses a dimension, others preserve it
- **Variance reduction**: Averaging reduces sensitivity to random seed
- **Diminishing returns**: Marginal gain drops after M ≈ 6-10

### Why Isolation Forest as final scorer?
- **No density estimation**: Works in high dimensions where KDE fails
- **Efficient**: O(log ψ) per query
- **Non-linear in original space**: Axis-aligned cuts in MLP-transformed space = curved boundaries in z-space
- **Well-understood**: Extensively validated, interpretable path-length semantics

### Why exclude query from its own peer set?
- **Leakage prevention**: Including x_i in computing μ_i, σ_i would bias statistics toward query
- **Anomaly dilution**: Extreme outlier would inflate its own σ_i, hiding itself
- **Standard practice**: Leave-one-out is standard in local outlier methods (LOF, KNN-based)

---

## 5. Limitations

### Inherent Limitations

1. **Curse of dimensionality in context space**
   - High-dimensional context → sparse neighborhoods → unstable peer statistics
   - Mitigation: Dimensionality reduction on context features, or feature selection

2. **Simple context-behavior relationship assumed**
   - Z-score assumes context only shifts/scales behavior
   - Cannot learn complex non-linear context functions (unlike deep learning approaches)
   - Trade-off: Simplicity and interpretability vs. expressiveness

3. **Pre-defined distance metric required**
   - K-NN requires meaningful distance in context space
   - Mixed feature types (categorical + continuous) require careful encoding

4. **Homogeneous peer edge case**
   - When all K peers have identical behavioral values, σ_i = 0
   - Current handling: Floor at ε = 1e-8 to prevent division by zero
   - Effect: z-scores become very large, flagging deviation from homogeneous peer group

### Hyperparameter Sensitivity (to be validated empirically)
- K: Too small → noisy statistics; too large → loses locality
- γ: Too small → only nearest neighbor matters; too large → approaches uniform weighting
- M: Diminishing returns beyond certain point
- d_h: Trade-off between expressiveness and overfitting to noise

---

## 6. Notation Summary

| Symbol | Meaning |
|--------|---------|
| N | Number of samples |
| d_c | Context feature dimension |
| d_y | Behavioral feature dimension |
| K | Number of neighbors |
| γ | RBF kernel bandwidth |
| M | Number of random projections |
| d_h | Hidden dimension |
| T | Number of IF trees |
| ψ | IF subsample size |
| ε | Numerical stability floor (1e-8) |
| c_i | Context features of point i |
| y_i | Behavioral features of point i |
| N_K(i) | K nearest neighbors of i (excluding i) |
| w_{ij} | Kernel weight between i and j |
| μ_i | Weighted peer mean |
| σ_i | Weighted peer std |
| z_i | Normalized behavioral features |
| h_i^{(m)} | m-th random projection |
| s_i | Final anomaly score [0-100] |
