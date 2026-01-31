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


# Dataset registry
SYNTHETIC_DATASETS = {
    'syn_linear': generate_syn_linear,
    'syn_scale': generate_syn_scale,
    'syn_multimodal': generate_syn_multimodal,
    'syn_nonlinear': generate_syn_nonlinear,
}


def get_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get a synthetic dataset by name."""
    if name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(SYNTHETIC_DATASETS.keys())}")
    return SYNTHETIC_DATASETS[name](**kwargs)
