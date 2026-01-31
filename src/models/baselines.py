"""
Baseline anomaly detection methods for comparison.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from typing import Optional


class IFBaseline:
    """Standard Isolation Forest on behavioral features only."""

    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        """Score using only behavioral features."""
        behavior = np.asarray(behavior)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        model.fit(behavior)
        return -model.score_samples(behavior)


class IFConcat:
    """Isolation Forest on concatenated context + behavior."""

    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        """Score using concatenated features."""
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        X = np.hstack([context, behavior])
        model = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        model.fit(X)
        return -model.score_samples(X)


class DIFBaseline:
    """Deep Isolation Forest - random MLP + IF on behavior only."""

    def __init__(
        self,
        n_projections: int = 6,
        hidden_dim: int = 128,
        n_estimators: int = 100,
        random_state: Optional[int] = None
    ):
        self.n_projections = n_projections
        self.hidden_dim = hidden_dim
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        """Score using random projections of behavior."""
        behavior = np.asarray(behavior)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples, d_y = behavior.shape

        all_scores = []
        for _ in range(self.n_projections):
            W = self._rng.randn(d_y, self.hidden_dim) * np.sqrt(2.0 / d_y)
            h = behavior @ W
            h = np.where(h > 0, h, 0.01 * h)  # LeakyReLU

            model = IsolationForest(
                n_estimators=self.n_estimators,
                random_state=self._rng.randint(0, 2**31),
                contamination='auto'
            )
            model.fit(h)
            scores = -model.score_samples(h)
            all_scores.append(scores)

        return np.mean(all_scores, axis=0)


class DIFConcat:
    """Deep Isolation Forest on concatenated context + behavior."""

    def __init__(
        self,
        n_projections: int = 6,
        hidden_dim: int = 128,
        n_estimators: int = 100,
        random_state: Optional[int] = None
    ):
        self.n_projections = n_projections
        self.hidden_dim = hidden_dim
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        X = np.hstack([context, behavior])
        n_samples, d = X.shape

        all_scores = []
        for _ in range(self.n_projections):
            W = self._rng.randn(d, self.hidden_dim) * np.sqrt(2.0 / d)
            h = X @ W
            h = np.where(h > 0, h, 0.01 * h)

            model = IsolationForest(
                n_estimators=self.n_estimators,
                random_state=self._rng.randint(0, 2**31),
                contamination='auto'
            )
            model.fit(h)
            scores = -model.score_samples(h)
            all_scores.append(scores)

        return np.mean(all_scores, axis=0)


class LOFBaseline:
    """Local Outlier Factor on concatenated features."""

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        X = np.hstack([context, behavior])
        model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            novelty=False
        )
        # LOF returns negative scores (more negative = more outlier)
        scores = -model.fit_predict(X)  # Convert -1/1 to scores
        # Use negative_outlier_factor_ for continuous scores
        return -model.negative_outlier_factor_


class QCAD:
    """
    Quantile-based Conditional Anomaly Detection.

    Simplified version: Uses K-NN peers and computes deviation from
    peer quantiles in behavioral space.
    """

    def __init__(self, n_neighbors: int = 100, random_state: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples = len(context)
        K = min(self.n_neighbors, n_samples - 1)

        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        _, indices = nn.kneighbors(context)
        peer_idx = indices[:, 1:]  # Exclude self

        # For each point, compute how extreme it is relative to peers
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            peer_values = behavior[peer_idx[i]]
            # Compute empirical quantile of point among peers
            for d in range(behavior.shape[1]):
                peer_d = peer_values[:, d]
                val = behavior[i, d]
                # Two-tailed: how far from median in quantile terms
                quantile = np.mean(peer_d <= val)
                # Convert to anomaly score: distance from 0.5
                scores[i] += abs(quantile - 0.5) * 2

        return scores / behavior.shape[1]  # Average over dimensions


class ROCOD:
    """
    Robust Conditional Outlier Detection.

    Simplified version: Uses robust statistics (median, MAD) for peer normalization.
    """

    def __init__(self, n_neighbors: int = 100, random_state: Optional[int] = None):
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples = len(context)
        K = min(self.n_neighbors, n_samples - 1)

        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        _, indices = nn.kneighbors(context)
        peer_idx = indices[:, 1:]

        peer_behavior = behavior[peer_idx]  # (n, K, d_y)

        # Robust statistics: median and MAD
        median = np.median(peer_behavior, axis=1)
        mad = np.median(np.abs(peer_behavior - median[:, np.newaxis, :]), axis=1)
        mad = np.maximum(mad, 1e-8)  # Avoid division by zero

        # Robust z-score
        z = np.abs(behavior - median) / (1.4826 * mad)  # 1.4826 for normal consistency

        # Score: max absolute z-score across dimensions
        scores = np.max(z, axis=1)
        return scores


# Method registry
METHODS = {
    'IF': IFBaseline,
    'IF_concat': IFConcat,
    'DIF': DIFBaseline,
    'DIF_concat': DIFConcat,
    'LOF': LOFBaseline,
    'QCAD': QCAD,
    'ROCOD': ROCOD,
}


def get_method(name: str, random_state: Optional[int] = None, **kwargs):
    """Get a method instance by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")

    cls = METHODS[name]
    if 'random_state' in cls.__init__.__code__.co_varnames:
        return cls(random_state=random_state, **kwargs)
    return cls(**kwargs)
