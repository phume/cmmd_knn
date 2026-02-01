"""
Synthetic dataset generators for contextual anomaly detection evaluation.
"""

import numpy as np
from typing import Tuple, Optional


def generate_syn_linear(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Linear: Linear context-behavior relationship.

    Normal:  y = 2*c + 5 + N(0, 1)
    Anomaly: y = 2*c + 5 + N(0, 1) + Uniform(5, 10) * sign

    Returns:
        context: (n_samples, 1) - context features
        behavior: (n_samples, 1) - behavioral features
        labels: (n_samples,) - 0=normal, 1=anomaly
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Context: Uniform(0, 10)
    context = rng.uniform(0, 10, size=(n_samples, 1))

    # Normal behavior: y = 2*c + 5 + noise
    behavior = 2 * context + 5 + rng.randn(n_samples, 1)

    # Labels
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Add anomaly shift
    signs = rng.choice([-1, 1], size=n_anomalies)
    shifts = rng.uniform(5, 10, size=n_anomalies)
    behavior[anomaly_idx, 0] += signs * shifts

    return context, behavior, labels


def generate_syn_scale(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Scale: Context affects variance (heteroscedasticity).

    Normal:  y ~ N(50, c²)  where c ∈ [1, 10]
    Anomaly: y ~ N(50, c²) + 5*c (shift scales with context)

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)

    # Context: Uniform(1, 10) - avoid 0 for variance
    context = rng.uniform(1, 10, size=(n_samples, 1))

    # Normal behavior: N(50, c²)
    behavior = 50 + context * rng.randn(n_samples, 1)

    # Labels
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Anomaly: add context-scaled shift
    signs = rng.choice([-1, 1], size=n_anomalies)
    behavior[anomaly_idx, 0] += signs * 5 * context[anomaly_idx, 0]

    return context, behavior, labels


def generate_syn_multimodal(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Multimodal: Context changes distribution shape.

    Context c ∈ {0, 1} (binary)
    c=0: y ~ N(0, 1)  (unimodal)
    c=1: y ~ 0.5*N(-3, 0.5) + 0.5*N(3, 0.5)  (bimodal)

    Anomaly: Points at wrong mode for context
    - c=0 with y near ±3 (bimodal modes)
    - c=1 with y near 0 (unimodal mode)

    This tests the LIMITATION of PNKDIF - it assumes shift/scale, not shape change.

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Split normal samples between contexts
    n_c0 = n_normal // 2
    n_c1 = n_normal - n_c0

    # Context 0: unimodal N(0, 1)
    context_c0 = np.zeros((n_c0, 1))
    behavior_c0 = rng.randn(n_c0, 1)

    # Context 1: bimodal mixture
    context_c1 = np.ones((n_c1, 1))
    mixture = rng.rand(n_c1) < 0.5
    behavior_c1 = np.where(
        mixture.reshape(-1, 1),
        -3 + 0.5 * rng.randn(n_c1, 1),
        3 + 0.5 * rng.randn(n_c1, 1)
    )

    # Anomalies: wrong distribution for context
    n_anom_c0 = n_anomalies // 2
    n_anom_c1 = n_anomalies - n_anom_c0

    # c=0 anomalies: bimodal values (should be unimodal)
    context_anom_c0 = np.zeros((n_anom_c0, 1))
    mixture_anom = rng.rand(n_anom_c0) < 0.5
    behavior_anom_c0 = np.where(
        mixture_anom.reshape(-1, 1),
        -3 + 0.5 * rng.randn(n_anom_c0, 1),
        3 + 0.5 * rng.randn(n_anom_c0, 1)
    )

    # c=1 anomalies: unimodal values (should be bimodal)
    context_anom_c1 = np.ones((n_anom_c1, 1))
    behavior_anom_c1 = rng.randn(n_anom_c1, 1)

    # Combine all
    context = np.vstack([context_c0, context_c1, context_anom_c0, context_anom_c1])
    behavior = np.vstack([behavior_c0, behavior_c1, behavior_anom_c0, behavior_anom_c1])
    labels = np.concatenate([
        np.zeros(n_c0 + n_c1),
        np.ones(n_anom_c0 + n_anom_c1)
    ])

    # Shuffle
    perm = rng.permutation(len(labels))
    context = context[perm]
    behavior = behavior[perm]
    labels = labels[perm]

    return context, behavior, labels


def generate_syn_nonlinear(
    n_samples: int = 10000,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Nonlinear: Non-linear anomaly boundary in behavior space.

    Context: c ∈ R² ~ Uniform([0,10], [0,10])
    Normal: y ∈ R² within radius 2 of context-dependent center
    Anomaly: Points on curved manifold outside normal region

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Context: 2D uniform
    context = rng.uniform(0, 10, size=(n_samples, 2))

    # Context-dependent center: sin/cos pattern
    center_y1 = np.sin(context[:, 0] * 0.5) * 3
    center_y2 = np.cos(context[:, 1] * 0.5) * 3

    # Normal: within radius 1.5 of center
    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii = rng.uniform(0, 1.5, n_samples)
    behavior = np.column_stack([
        center_y1 + radii * np.cos(angles),
        center_y2 + radii * np.sin(angles)
    ])

    # Labels
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Anomalies: push outside radius (3-5 from center)
    anom_radii = rng.uniform(3, 5, n_anomalies)
    anom_angles = rng.uniform(0, 2 * np.pi, n_anomalies)
    behavior[anomaly_idx, 0] = center_y1[anomaly_idx] + anom_radii * np.cos(anom_angles)
    behavior[anomaly_idx, 1] = center_y2[anomaly_idx] + anom_radii * np.sin(anom_angles)

    return context, behavior, labels


def generate_syn_highdim_context(
    n_samples: int = 10000,
    d_context: int = 20,
    d_informative: int = 2,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-HighDimContext: Test curse of dimensionality.

    Context: d_context dimensions, but only d_informative are relevant
    Tests K-NN degradation in high-dimensional spaces.

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)

    # High-dim context with sparse informative structure
    context = rng.randn(n_samples, d_context)

    # Only first d_informative dimensions affect behavior
    informative_context = context[:, :d_informative]

    # Behavior depends on informative context
    behavior = np.column_stack([
        2 * informative_context[:, 0] + rng.randn(n_samples) * 0.5,
        informative_context.sum(axis=1) + rng.randn(n_samples) * 0.5,
        np.sin(informative_context[:, 0]) * 3 + rng.randn(n_samples) * 0.3
    ])

    # Labels
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Anomaly: shift behavior relative to informative context peers
    for idx in anomaly_idx:
        # Find similar points in informative space
        dists = np.linalg.norm(informative_context - informative_context[idx], axis=1)
        peer_idx = np.argsort(dists)[1:51]  # 50 nearest
        peer_std = behavior[peer_idx].std(axis=0)
        shift = rng.choice([-1, 1], size=3) * 3 * peer_std
        behavior[idx] += shift

    return context, behavior, labels


def generate_syn_cluster(
    n_samples: int = 10000,
    n_clusters: int = 5,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Cluster: Clustered context space.

    Context: 5 Gaussian clusters with different behavior distributions
    Anomaly: Normal behavior for WRONG cluster (contextual swap)

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_per_cluster = n_samples // n_clusters

    # Cluster centers in context space
    cluster_centers = rng.uniform(-5, 5, size=(n_clusters, 2))

    # Each cluster has different behavior distribution
    cluster_behavior_means = rng.uniform(-3, 3, size=(n_clusters, 3))
    cluster_behavior_stds = rng.uniform(0.3, 1.0, size=(n_clusters, 3))

    contexts = []
    behaviors = []
    cluster_labels = []

    for i in range(n_clusters):
        n_this = n_per_cluster if i < n_clusters - 1 else n_samples - len(contexts) * n_per_cluster // n_clusters
        if i == n_clusters - 1:
            n_this = n_samples - sum(len(c) for c in contexts)

        # Context: Gaussian around cluster center
        ctx = cluster_centers[i] + rng.randn(n_this, 2) * 0.5
        contexts.append(ctx)

        # Behavior: cluster-specific distribution
        beh = cluster_behavior_means[i] + rng.randn(n_this, 3) * cluster_behavior_stds[i]
        behaviors.append(beh)

        cluster_labels.extend([i] * n_this)

    context = np.vstack(contexts)
    behavior = np.vstack(behaviors)
    cluster_labels = np.array(cluster_labels)

    # Labels: anomalies are points with behavior from WRONG cluster
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Swap behavior with a different cluster
    for idx in anomaly_idx:
        current_cluster = cluster_labels[idx]
        other_cluster = (current_cluster + rng.randint(1, n_clusters)) % n_clusters
        # Replace with typical behavior from other cluster
        behavior[idx] = cluster_behavior_means[other_cluster] + \
                        rng.randn(3) * cluster_behavior_stds[other_cluster]

    # Shuffle
    perm = rng.permutation(n_samples)
    context = context[perm]
    behavior = behavior[perm]
    labels = labels[perm]

    return context, behavior, labels


def generate_syn_complex(
    n_samples: int = 20000,
    d_context: int = 50,
    d_behavior: int = 20,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Complex: Super high-dimensional non-linear contextual anomalies.

    Designed to showcase PNKDIF's power with MLP projections:
    - 50D context with complex manifold structure
    - 20D behavior with non-linear context dependencies
    - Multiple interacting non-linear functions
    - Anomalies require learning non-linear projections to detect

    Context structure:
    - 5 latent factors generate 50D observed context via non-linear mixing
    - Context lies on a complex manifold

    Behavior structure:
    - Behavior depends non-linearly on latent context factors
    - Includes interactions, polynomials, and periodic functions

    Anomalies:
    - Subtle shifts in behavior that are normal GLOBALLY
    - But unusual for the local context manifold region
    - Requires learning the non-linear context-behavior mapping

    Returns:
        context: (n_samples, 50)
        behavior: (n_samples, 20)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    d_latent = 5  # Latent factors

    # Generate latent context factors (the "true" context)
    latent = rng.randn(n_samples, d_latent)

    # Non-linear mixing to create observed high-dim context
    # Each observed dim is a random non-linear combination of latents
    context = np.zeros((n_samples, d_context))

    for j in range(d_context):
        # Random coefficients for this dimension
        coeffs = rng.randn(d_latent) * 0.5

        # Mix of linear, quadratic, and interaction terms
        linear = latent @ coeffs

        # Quadratic terms
        quad_idx = rng.choice(d_latent, 2, replace=False)
        quadratic = latent[:, quad_idx[0]] * latent[:, quad_idx[1]] * rng.randn() * 0.3

        # Periodic terms
        periodic_idx = rng.choice(d_latent)
        periodic = np.sin(latent[:, periodic_idx] * rng.uniform(0.5, 2)) * rng.randn() * 0.5

        # Combine with noise
        context[:, j] = linear + quadratic + periodic + rng.randn(n_samples) * 0.1

    # Generate behavior as non-linear function of latent factors
    behavior = np.zeros((n_samples, d_behavior))

    # Store the "expected" behavior for each sample (for anomaly injection)
    expected_behavior = np.zeros((n_samples, d_behavior))

    for j in range(d_behavior):
        # Complex non-linear function of latent factors
        coeffs = rng.randn(d_latent)

        # Base: polynomial of degree 2-3
        base = np.zeros(n_samples)
        for k in range(d_latent):
            base += coeffs[k] * latent[:, k]
            if rng.rand() > 0.5:
                base += coeffs[k] * 0.3 * latent[:, k] ** 2
            if rng.rand() > 0.7:
                base += coeffs[k] * 0.1 * latent[:, k] ** 3

        # Interaction terms (2-way and 3-way)
        for _ in range(3):
            idx = rng.choice(d_latent, 2, replace=False)
            base += rng.randn() * 0.4 * latent[:, idx[0]] * latent[:, idx[1]]

        if rng.rand() > 0.5:
            idx = rng.choice(d_latent, 3, replace=False)
            base += rng.randn() * 0.2 * latent[:, idx[0]] * latent[:, idx[1]] * latent[:, idx[2]]

        # Non-linear activations
        if j % 4 == 0:
            base = np.tanh(base)
        elif j % 4 == 1:
            base = np.sin(base * 0.5) * 2
        elif j % 4 == 2:
            base = np.sign(base) * np.log1p(np.abs(base))
        # else: keep as is (polynomial)

        expected_behavior[:, j] = base

        # Add context-dependent noise (heteroscedastic)
        noise_scale = 0.3 + 0.2 * np.abs(latent[:, j % d_latent])
        behavior[:, j] = base + rng.randn(n_samples) * noise_scale

    # Create anomalies: behavior from DIFFERENT region of latent space
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        # Find the latent position
        current_latent = latent[idx]

        # Find samples with DIFFERENT latent position (distant in latent space)
        latent_dists = np.linalg.norm(latent - current_latent, axis=1)

        # Get samples from far away in latent space (75th-95th percentile distance)
        dist_threshold_low = np.percentile(latent_dists, 75)
        dist_threshold_high = np.percentile(latent_dists, 95)
        far_mask = (latent_dists > dist_threshold_low) & (latent_dists < dist_threshold_high)
        far_indices = np.where(far_mask)[0]

        if len(far_indices) > 0:
            # Swap behavior with a distant sample
            donor_idx = rng.choice(far_indices)
            # Use the EXPECTED behavior from donor (not noisy version)
            # This makes the anomaly "normal globally" but wrong for context
            behavior[idx] = expected_behavior[donor_idx] + rng.randn(d_behavior) * 0.2

    return context, behavior, labels


def generate_syn_deep_manifold(
    n_samples: int = 15000,
    d_context: int = 30,
    d_behavior: int = 15,
    n_layers: int = 3,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-DeepManifold: Data lies on a deep non-linear manifold.

    Simulates data that requires deep learning to model:
    - Context and behavior are generated through multiple layers of non-linear transforms
    - The relationship is a "deep" function that shallow methods can't capture

    Returns:
        context: (n_samples, 30)
        behavior: (n_samples, 15)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    d_latent = 8

    # Latent codes
    latent = rng.randn(n_samples, d_latent)

    def random_layer(x, out_dim, rng, activation='relu'):
        """Apply a random non-linear layer."""
        in_dim = x.shape[1]
        W = rng.randn(in_dim, out_dim) / np.sqrt(in_dim)
        b = rng.randn(out_dim) * 0.1
        h = x @ W + b

        if activation == 'relu':
            return np.maximum(0, h)
        elif activation == 'tanh':
            return np.tanh(h)
        elif activation == 'sin':
            return np.sin(h)
        elif activation == 'leaky':
            return np.where(h > 0, h, 0.1 * h)
        return h

    # Generate context through deep network
    h = latent
    for i in range(n_layers):
        hidden_dim = d_latent * 2 if i < n_layers - 1 else d_context
        activation = ['relu', 'tanh', 'leaky'][i % 3]
        h = random_layer(h, hidden_dim, rng, activation)
    context = h + rng.randn(n_samples, d_context) * 0.1

    # Generate behavior through different deep network (sharing latent)
    h = latent
    for i in range(n_layers):
        hidden_dim = d_latent * 2 if i < n_layers - 1 else d_behavior
        activation = ['tanh', 'sin', 'relu'][i % 3]
        h = random_layer(h, hidden_dim, rng, activation)
    expected_behavior = h
    behavior = h + rng.randn(n_samples, d_behavior) * 0.2

    # Anomalies: swap behavior between distant latent regions
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        latent_dists = np.linalg.norm(latent - latent[idx], axis=1)
        far_mask = latent_dists > np.percentile(latent_dists, 80)
        far_indices = np.where(far_mask)[0]
        if len(far_indices) > 0:
            donor = rng.choice(far_indices)
            behavior[idx] = expected_behavior[donor] + rng.randn(d_behavior) * 0.15

    return context, behavior, labels


def generate_syn_hidden_anomaly(
    n_samples: int = 20000,
    d_context: int = 20,
    d_behavior: int = 30,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-HiddenAnomaly: Anomaly visible only after non-linear transformation.

    The anomaly is in the PRODUCT of behavior dimensions, not individual dims.
    - Normal: y1 * y2 * y3 ≈ context-dependent target (varies by context)
    - Anomaly: y1 * y2 * y3 deviates from expected

    Individual dimensions look normal, but the non-linear combination differs.
    This tests if MLP can learn the relevant non-linear projection.

    Returns:
        context: (n_samples, 20)
        behavior: (n_samples, 30)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)

    # Context: continuous 20D
    context = rng.randn(n_samples, d_context)

    # The "true signal" depends on context via non-linear function
    # Target = tanh(sum of first 5 context dims)
    target = np.tanh(context[:, :5].sum(axis=1))

    # Behavior: 30D, mostly noise
    behavior = rng.randn(n_samples, d_behavior)

    # But dims 0, 1, 2 satisfy: y0 * y1 + y2^2 ≈ target for normal points
    # We set y2 to achieve this
    for i in range(n_samples):
        product = behavior[i, 0] * behavior[i, 1]
        # y2^2 = target - product, so y2 = sign * sqrt(|target - product|)
        diff = target[i] - product
        if diff >= 0:
            behavior[i, 2] = np.sqrt(diff) + rng.randn() * 0.1
        else:
            behavior[i, 2] = -np.sqrt(-diff) + rng.randn() * 0.1

    # Second hidden constraint: y3*y4 - sin(y5) ≈ context[:, 5:10].mean()
    target2 = context[:, 5:10].mean(axis=1)
    for i in range(n_samples):
        # Set y5 to satisfy
        product = behavior[i, 3] * behavior[i, 4]
        behavior[i, 5] = np.arcsin(np.clip(product - target2[i], -1, 1)) + rng.randn() * 0.1

    # Create anomalies: break BOTH hidden constraints
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        # Perturb the constrained dimensions to break the relationship
        # But keep them within normal range for each dimension
        behavior[idx, 2] += rng.choice([-1, 1]) * (1.5 + rng.rand())
        behavior[idx, 5] += rng.choice([-1, 1]) * (1.0 + rng.rand() * 0.5)

    return context, behavior, labels


def generate_syn_xor_context(
    n_samples: int = 20000,
    d_context: int = 40,
    d_behavior: int = 10,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-XOR: XOR-like context structure that REQUIRES non-linear projections.

    The "true" context is a 2D XOR pattern embedded in high-dim space:
    - Quadrant (++): behavior type A
    - Quadrant (--): behavior type A (same as ++)
    - Quadrant (+-): behavior type B
    - Quadrant (-+): behavior type B (same as +-)

    This is NOT linearly separable - requires non-linear transformation.
    Anomalies: behavior from wrong quadrant pair.

    Linear methods (including PNKIF without MLP) cannot distinguish
    (++) from (+-) using only linear projections of context.

    Returns:
        context: (n_samples, 40)
        behavior: (n_samples, 10)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies

    # Generate 2D latent XOR coordinates
    latent_2d = rng.randn(n_samples, 2) * 2

    # Determine quadrant (XOR pattern)
    # Type A: both same sign (++ or --)
    # Type B: different signs (+- or -+)
    is_type_A = (latent_2d[:, 0] * latent_2d[:, 1]) > 0

    # Embed 2D into high-dim context via random non-linear projection
    # This makes the XOR structure hidden
    context = np.zeros((n_samples, d_context))

    # Random embedding matrix
    embed_W1 = rng.randn(2, 20) / np.sqrt(2)
    embed_W2 = rng.randn(20, d_context) / np.sqrt(20)

    # Non-linear embedding: two-layer with tanh
    h1 = np.tanh(latent_2d @ embed_W1)
    context = h1 @ embed_W2

    # Add noise and random projections of nuisance dimensions
    nuisance = rng.randn(n_samples, 10)
    nuisance_proj = rng.randn(10, d_context) / np.sqrt(10)
    context += nuisance @ nuisance_proj * 0.5
    context += rng.randn(n_samples, d_context) * 0.3

    # Behavior distributions
    behavior_A_mean = rng.randn(d_behavior) * 2
    behavior_B_mean = -behavior_A_mean  # Opposite
    behavior_std = 0.5

    # Generate normal behavior based on type
    behavior = np.zeros((n_samples, d_behavior))
    behavior[is_type_A] = behavior_A_mean + rng.randn(is_type_A.sum(), d_behavior) * behavior_std
    behavior[~is_type_A] = behavior_B_mean + rng.randn((~is_type_A).sum(), d_behavior) * behavior_std

    # Anomalies: swap behavior type
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        if is_type_A[idx]:
            # Should be type A, give type B behavior
            behavior[idx] = behavior_B_mean + rng.randn(d_behavior) * behavior_std
        else:
            # Should be type B, give type A behavior
            behavior[idx] = behavior_A_mean + rng.randn(d_behavior) * behavior_std

    # Shuffle
    perm = rng.permutation(n_samples)
    return context[perm], behavior[perm], labels[perm]


def generate_syn_spiral_context(
    n_samples: int = 15000,
    d_context: int = 30,
    d_behavior: int = 8,
    n_spirals: int = 3,
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Syn-Spiral: Interleaved spiral context structure.

    Context lies on interleaved spirals in high-dim space.
    Each spiral has different behavior distribution.
    Anomalies: behavior from wrong spiral.

    The spirals are NOT linearly separable, requiring non-linear
    projections to identify which spiral a point belongs to.

    Returns:
        context: (n_samples, 30)
        behavior: (n_samples, 8)
        labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)

    n_anomalies = int(n_samples * anomaly_rate)
    n_per_spiral = n_samples // n_spirals

    contexts = []
    behaviors = []
    spiral_ids = []

    # Behavior means for each spiral
    spiral_behavior_means = rng.randn(n_spirals, d_behavior) * 2

    for s in range(n_spirals):
        n_this = n_per_spiral if s < n_spirals - 1 else n_samples - len(spiral_ids)

        # Generate spiral in 2D
        t = np.linspace(0, 4 * np.pi, n_this) + rng.randn(n_this) * 0.3
        r = t / (4 * np.pi) * 5 + 1

        # Rotate each spiral
        angle_offset = 2 * np.pi * s / n_spirals
        x = r * np.cos(t + angle_offset) + rng.randn(n_this) * 0.3
        y = r * np.sin(t + angle_offset) + rng.randn(n_this) * 0.3

        spiral_2d = np.column_stack([x, y])

        # Embed to high-dim via random non-linear projection
        W1 = rng.randn(2, 15) / np.sqrt(2)
        W2 = rng.randn(15, d_context) / np.sqrt(15)

        h = np.tanh(spiral_2d @ W1 + rng.randn(15) * 0.1)
        ctx = h @ W2 + rng.randn(n_this, d_context) * 0.2

        contexts.append(ctx)

        # Behavior for this spiral
        beh = spiral_behavior_means[s] + rng.randn(n_this, d_behavior) * 0.4
        behaviors.append(beh)

        spiral_ids.extend([s] * n_this)

    context = np.vstack(contexts)
    behavior = np.vstack(behaviors)
    spiral_ids = np.array(spiral_ids)

    # Anomalies: wrong spiral behavior
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    for idx in anomaly_idx:
        current_spiral = spiral_ids[idx]
        wrong_spiral = (current_spiral + rng.randint(1, n_spirals)) % n_spirals
        behavior[idx] = spiral_behavior_means[wrong_spiral] + rng.randn(d_behavior) * 0.4

    # Shuffle
    perm = rng.permutation(n_samples)
    return context[perm], behavior[perm], labels[perm]


# Dataset registry
SYNTHETIC_DATASETS = {
    'syn_linear': generate_syn_linear,
    'syn_scale': generate_syn_scale,
    'syn_multimodal': generate_syn_multimodal,
    'syn_nonlinear': generate_syn_nonlinear,
    'syn_highdim_context': generate_syn_highdim_context,
    'syn_cluster': generate_syn_cluster,
    'syn_complex': generate_syn_complex,
    'syn_deep_manifold': generate_syn_deep_manifold,
    'syn_xor': generate_syn_xor_context,
    'syn_spiral': generate_syn_spiral_context,
    'syn_hidden_anomaly': generate_syn_hidden_anomaly,
}


def get_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get a synthetic dataset by name."""
    if name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(SYNTHETIC_DATASETS.keys())}")
    return SYNTHETIC_DATASETS[name](**kwargs)
