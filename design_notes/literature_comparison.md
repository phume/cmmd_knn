# Technical Analysis: Peer Normalize Kernel Based Deep Isolation Forest

## 1. Technical Summary

### Method Pipeline

**Input**: Dataset with context features $C$ and behavioral features $Y$

**Step 1 - Peer Selection**: For each point $x_i$, find $K$ nearest neighbors in context space $C$ using distance metric $d(c_i, c_j)$

**Step 2 - Kernel Weighting**: Apply RBF kernel weights to peers:
$$w_j = \exp\left(-\frac{d(c_i, c_j)^2}{2\sigma^2}\right)$$

**Step 3 - Peer Statistics**: Compute weighted peer statistics:
$$\mu_{peer} = \frac{\sum_j w_j \cdot y_j}{\sum_j w_j}, \quad \sigma_{peer} = \sqrt{\frac{\sum_j w_j (y_j - \mu_{peer})^2}{\sum_j w_j}}$$

**Step 4 - Z-Score Normalization**: Transform behavioral features:
$$z_i = \frac{y_i - \mu_{peer}}{\sigma_{peer}}$$

**Step 5 - Random MLP Projection**: Apply $M$ frozen randomly-initialized MLPs:
$$h_i^{(m)} = \text{LeakyReLU}(z_i \cdot W^{(m)} + b^{(m)})$$
where $W^{(m)}$ are random weights, $b^{(m)} = 0$

**Step 6 - Isolation Forest Ensemble**: Run IF on each representation, aggregate:
$$\text{score}(x_i) = \frac{1}{M} \sum_{m=1}^{M} \text{IF}(h_i^{(m)})$$

---

## 2. Comparison Table

| Property | IF | DIF | ROCOD | QCAD | CWAE | NS | **Proposed** |
|----------|:--:|:---:|:-----:|:----:|:----:|:--:|:------------:|
| Context-aware | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Non-linear boundaries | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| Training-free | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Handles high-dim context | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Distributional modeling | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Uncertainty quantification | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Local normalization | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |
| Ensemble diversity | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Interpretable scores | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | Partial |
| Handles missing context | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Scalable to large N | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Requires hyperparameter tuning | Low | Low | High | Med | High | High | Low |

---

## 3. Inspiration Mapping

### From Deep Isolation Forest (DIF)
- Random representation ensemble via casually initialized neural networks
- Non-linear isolation through axis-parallel cuts in transformed spaces
- Aggregation of multiple IF scores

### From ROCOD
- Local peer-based expected behavior modeling
- Weighted combination based on neighborhood density
- Concept of "contextually similar" reference points

### From QCAD
- Non-parametric peer selection in context space
- K-NN as basis for conditional distribution estimation
- Separation of context and behavioral features

### From CWAE/CVAE
- Explicit conditioning on context variables
- Concept of modeling P(Y|C) for anomaly detection

### From NS (Normalcy Score)
- Z-score transformation as anomaly indicator
- Peer-relative deviation measurement

### Potentially Novel Combinations
- **Kernel-weighted peer statistics** → soft K-NN instead of hard cutoff
- **Z-score feeding into random MLP** → normalized features as DIF input
- **Training-free contextual detection** → no learned context encoding

---

## 4. Candidate Novelty Claims

### Strong Claims
1. **First training-free contextual anomaly detector with non-linear decision boundaries**
   - ROCOD/QCAD: training-free but linear
   - CWAE/CVAE/NS: non-linear but require training
   - Proposed: training-free AND non-linear via frozen MLP + IF

2. **Kernel-weighted peer normalization for anomaly detection**
   - Soft peer weighting via RBF kernel (vs hard K-NN cutoff)
   - Weighted mean/variance computation for z-scores

### Moderate Claims
3. **Integration of local normalization with random representation ensemble**
   - DIF uses global features; proposed uses peer-normalized features
   - Combines contextual preprocessing with DIF architecture

4. **Elimination of training while preserving contextual sensitivity**
   - All existing CAD methods either train (CWAE, NS) or lack non-linearity (ROCOD, QCAD)

### Weak Claims (Incremental)
5. Using z-scores as input to neural representations (straightforward extension)
6. Multiple random projections for robustness (standard ensemble technique)

---

## 5. Assumptions Analysis

### Assumptions Made
1. **Gaussian-like peer distributions**: Z-score normalization assumes mean/variance are meaningful statistics
2. **Euclidean context similarity**: K-NN assumes distance in context space reflects behavioral similarity
3. **Additive/multiplicative context effects**: Normalization assumes context shifts/scales behavior
4. **Sufficient peer density**: Requires enough neighbors in context space for stable statistics
5. **Feature independence after normalization**: IF assumes features can be isolated independently

### Assumptions Deliberately Avoided
1. **Labeled anomaly data**: Fully unsupervised
2. **Specific anomaly patterns**: Random projections are anomaly-agnostic
3. **Learned context representations**: No neural encoding of context
4. **Parametric behavioral distributions**: Non-parametric peer statistics
5. **Global behavioral norms**: Only peer-relative comparisons

---

## 6. Identified Limitations (from concept.md)

1. **Curse of dimensionality in context space**: High-dimensional context → sparse neighborhoods
2. **Simple context-behavior relationship**: Cannot learn complex non-linear context functions
3. **Pre-defined distance metric**: Requires manual kernel/neighbor specification
