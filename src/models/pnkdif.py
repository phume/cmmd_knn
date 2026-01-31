"""
Peer-Normalized Kernel Deep Isolation Forest (PNKDIF)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from typing import Optional
from dataclasses import dataclass


@dataclass
class PNKDIFConfig:
    """Configuration for PNKDIF algorithm."""
    n_neighbors: int = 100
    kernel_bandwidth: Optional[float] = None  # Auto-compute if None
    n_projections: int = 6
    hidden_dim: int = 128
    n_trees: int = 100
    subsample_size: int = 256
    epsilon: float = 1e-8
    random_state: Optional[int] = None


class PNKDIF:
    """Peer-Normalized Kernel Deep Isolation Forest."""

    def __init__(self, config: Optional[PNKDIFConfig] = None):
        self.config = config or PNKDIFConfig()
        self._rng = np.random.RandomState(self.config.random_state)

    def _compute_bandwidth(self, context: np.ndarray) -> float:
        """Compute kernel bandwidth using median heuristic."""
        n_samples = min(1000, len(context))
        idx = self._rng.choice(len(context), n_samples, replace=False)
        sample = context[idx]
        from scipy.spatial.distance import pdist
        distances = pdist(sample)
        return np.median(distances) + 1e-8

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters:
            context: (n_samples, d_c) - context features
            behavior: (n_samples, d_y) - behavioral features

        Returns:
            scores: (n_samples,) - anomaly scores (higher = more anomalous)
        """
        context = np.asarray(context)
        behavior = np.asarray(behavior)

        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples = len(context)
        K = min(self.config.n_neighbors, n_samples - 1)

        # Bandwidth
        bandwidth = self.config.kernel_bandwidth
        if bandwidth is None:
            bandwidth = self._compute_bandwidth(context)

        # Find K nearest neighbors
        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        distances, indices = nn.kneighbors(context)

        # Exclude self
        peer_idx = indices[:, 1:]
        peer_dist = distances[:, 1:]

        # Kernel weights
        weights = np.exp(-peer_dist**2 / (2 * bandwidth**2))
        weight_sum = weights.sum(axis=1, keepdims=True) + self.config.epsilon
        norm_weights = weights / weight_sum

        # Peer statistics
        peer_behavior = behavior[peer_idx]  # (n, K, d_y)
        mu = (norm_weights[:, :, np.newaxis] * peer_behavior).sum(axis=1)
        deviations = peer_behavior - mu[:, np.newaxis, :]
        variance = (norm_weights[:, :, np.newaxis] * deviations**2).sum(axis=1)
        sigma = np.sqrt(variance) + self.config.epsilon

        # Z-score normalization
        z = (behavior - mu) / sigma

        # Random MLP projections
        d_y = behavior.shape[1]
        projections = []
        for _ in range(self.config.n_projections):
            W = self._rng.randn(d_y, self.config.hidden_dim) * np.sqrt(2.0 / d_y)
            h = z @ W
            h = np.where(h > 0, h, 0.01 * h)  # LeakyReLU
            projections.append(h)

        # Isolation Forest scoring
        all_scores = []
        for h in projections:
            if_model = IsolationForest(
                n_estimators=self.config.n_trees,
                max_samples=min(self.config.subsample_size, n_samples),
                random_state=self._rng.randint(0, 2**31),
                contamination='auto'
            )
            if_model.fit(h)
            # Convert to positive scores (higher = more anomalous)
            scores = -if_model.score_samples(h)
            all_scores.append(scores)

        # Aggregate
        final_scores = np.mean(all_scores, axis=0)
        return final_scores


class PNKDIFUniform(PNKDIF):
    """PNKDIF with uniform peer weights (ablation)."""

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples = len(context)
        K = min(self.config.n_neighbors, n_samples - 1)

        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        _, indices = nn.kneighbors(context)
        peer_idx = indices[:, 1:]

        # Uniform weights
        peer_behavior = behavior[peer_idx]
        mu = peer_behavior.mean(axis=1)
        sigma = peer_behavior.std(axis=1) + self.config.epsilon

        z = (behavior - mu) / sigma

        # Same projection and IF scoring
        d_y = behavior.shape[1]
        projections = []
        for _ in range(self.config.n_projections):
            W = self._rng.randn(d_y, self.config.hidden_dim) * np.sqrt(2.0 / d_y)
            h = z @ W
            h = np.where(h > 0, h, 0.01 * h)
            projections.append(h)

        all_scores = []
        for h in projections:
            if_model = IsolationForest(
                n_estimators=self.config.n_trees,
                max_samples=min(self.config.subsample_size, n_samples),
                random_state=self._rng.randint(0, 2**31),
                contamination='auto'
            )
            if_model.fit(h)
            scores = -if_model.score_samples(h)
            all_scores.append(scores)

        return np.mean(all_scores, axis=0)


class PNKDIFNoMLP(PNKDIF):
    """PNKDIF without MLP projection - direct IF on z-scores (ablation)."""

    def fit_predict(self, context: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_samples = len(context)
        K = min(self.config.n_neighbors, n_samples - 1)

        bandwidth = self.config.kernel_bandwidth
        if bandwidth is None:
            bandwidth = self._compute_bandwidth(context)

        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        distances, indices = nn.kneighbors(context)
        peer_idx = indices[:, 1:]
        peer_dist = distances[:, 1:]

        weights = np.exp(-peer_dist**2 / (2 * bandwidth**2))
        weight_sum = weights.sum(axis=1, keepdims=True) + self.config.epsilon
        norm_weights = weights / weight_sum

        peer_behavior = behavior[peer_idx]
        mu = (norm_weights[:, :, np.newaxis] * peer_behavior).sum(axis=1)
        deviations = peer_behavior - mu[:, np.newaxis, :]
        variance = (norm_weights[:, :, np.newaxis] * deviations**2).sum(axis=1)
        sigma = np.sqrt(variance) + self.config.epsilon

        z = (behavior - mu) / sigma

        # Direct IF on z-scores (no MLP)
        if_model = IsolationForest(
            n_estimators=self.config.n_trees,
            max_samples=min(self.config.subsample_size, n_samples),
            random_state=self._rng.randint(0, 2**31),
            contamination='auto'
        )
        if_model.fit(z)
        return -if_model.score_samples(z)
