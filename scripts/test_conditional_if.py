"""
Quick tests of Conditional Isolation Forest approaches.

Goal: Bring contextual conditioning to IF without deep learning overhead.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import time

from data.synthetic import generate_syn_cluster, generate_syn_highdim_context
from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline, ROCOD


class ResidualIF:
    """
    Residual Isolation Forest.

    1. Train regression: predicted_behavior = f(context)
    2. Compute residuals: r = behavior - predicted
    3. Run IF on residuals

    Idea: Anomalies are points whose behavior deviates from what's
    expected given their context.
    """

    def __init__(self, n_estimators=100, regressor='ridge', random_state=None):
        self.n_estimators = n_estimators
        self.regressor_type = regressor
        self.random_state = random_state

    def fit_predict(self, context, behavior):
        context = np.asarray(context)
        behavior = np.asarray(behavior)

        # Train regressor to predict behavior from context
        if self.regressor_type == 'ridge':
            self.regressor = Ridge(alpha=1.0)
        else:
            self.regressor = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=self.random_state
            )

        self.regressor.fit(context, behavior)
        predicted = self.regressor.predict(context)

        # Compute residuals
        residuals = behavior - predicted

        # Run IF on residuals
        iso = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        iso.fit(residuals)
        return -iso.score_samples(residuals)


class LocalIF:
    """
    Local Isolation Forest.

    For each point, build IF only on its K context-neighbors.
    Score = how anomalous the point is within its local context group.

    This is expensive (N separate IF evaluations) but truly contextual.
    """

    def __init__(self, n_neighbors=50, n_estimators=100, random_state=None):
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_predict(self, context, behavior):
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        n_samples = len(context)

        K = min(self.n_neighbors, n_samples - 1)

        # Find neighbors in context space
        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        _, indices = nn.kneighbors(context)

        scores = np.zeros(n_samples)
        rng = np.random.RandomState(self.random_state)

        for i in range(n_samples):
            # Get local neighborhood (including self)
            local_idx = indices[i]
            local_behavior = behavior[local_idx]

            # Build IF on local neighborhood
            iso = IsolationForest(
                n_estimators=min(self.n_estimators, 50),  # Fewer trees for speed
                max_samples=min(K, 256),
                random_state=rng.randint(0, 2**31),
                contamination='auto'
            )
            iso.fit(local_behavior)

            # Score the target point (index 0 is self)
            scores[i] = -iso.score_samples(local_behavior[[0]])[0]

        return scores


class HybridContextIF:
    """
    Hybrid Contextual Isolation Forest.

    Combines two signals:
    1. Residual IF: How much does behavior deviate from predicted?
    2. Peer IF: How unusual is behavior compared to context-neighbors?

    Final score = weighted combination (or max) of both.
    """

    def __init__(self, n_neighbors=50, n_estimators=100, random_state=None,
                 combine='max'):
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.combine = combine  # 'max', 'mean', 'product'

    def fit_predict(self, context, behavior):
        context = np.asarray(context)
        behavior = np.asarray(behavior)
        n_samples = len(context)

        # 1. Residual-based score
        regressor = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=self.random_state
        )
        regressor.fit(context, behavior)
        predicted = regressor.predict(context)
        residuals = behavior - predicted

        iso_residual = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        iso_residual.fit(residuals)
        residual_scores = -iso_residual.score_samples(residuals)

        # 2. Peer-based score (z-score normalized + IF)
        K = min(self.n_neighbors, n_samples - 1)
        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        distances, indices = nn.kneighbors(context)
        peer_idx = indices[:, 1:]
        peer_dist = distances[:, 1:]

        # RBF kernel weights
        bandwidth = np.median(peer_dist) + 1e-8
        weights = np.exp(-peer_dist**2 / (2 * bandwidth**2))
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        # Peer statistics
        peer_behavior = behavior[peer_idx]
        mu = (weights[:, :, np.newaxis] * peer_behavior).sum(axis=1)
        deviations = peer_behavior - mu[:, np.newaxis, :]
        variance = (weights[:, :, np.newaxis] * deviations**2).sum(axis=1)
        sigma = np.sqrt(variance) + 1e-8

        z = (behavior - mu) / sigma

        iso_peer = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        iso_peer.fit(z)
        peer_scores = -iso_peer.score_samples(z)

        # Normalize both to [0, 1]
        residual_scores = (residual_scores - residual_scores.min()) / (residual_scores.max() - residual_scores.min() + 1e-8)
        peer_scores = (peer_scores - peer_scores.min()) / (peer_scores.max() - peer_scores.min() + 1e-8)

        # Combine
        if self.combine == 'max':
            return np.maximum(residual_scores, peer_scores)
        elif self.combine == 'mean':
            return (residual_scores + peer_scores) / 2
        elif self.combine == 'product':
            return residual_scores * peer_scores
        else:
            return residual_scores + peer_scores


class LocalIF_Fast:
    """
    Faster Local IF using shared tree structure.

    Instead of building N separate IFs, we:
    1. Build one IF on all data
    2. For scoring, only consider paths through context-similar points

    Actually, simpler approach: use peer-normalized + IF (which is PNKIF).

    Alternative: Build IF on behavior, but weight by context similarity.
    """

    def __init__(self, n_neighbors=50, n_estimators=100, random_state=None):
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit_predict(self, context, behavior):
        # This is essentially PNKIF with z-score normalization
        # Let's try something different: weighted ensemble

        context = np.asarray(context)
        behavior = np.asarray(behavior)
        n_samples = len(context)
        K = min(self.n_neighbors, n_samples - 1)

        # Build global IF
        iso = IsolationForest(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            contamination='auto'
        )
        iso.fit(behavior)
        global_scores = -iso.score_samples(behavior)

        # Find neighbors
        nn = NearestNeighbors(n_neighbors=K + 1, algorithm='auto')
        nn.fit(context)
        distances, indices = nn.kneighbors(context)

        # For each point, compare its score to local distribution
        local_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbor_scores = global_scores[indices[i, 1:]]  # Exclude self
            # How unusual is this point's score compared to neighbors?
            local_mean = neighbor_scores.mean()
            local_std = neighbor_scores.std() + 1e-8
            local_scores[i] = (global_scores[i] - local_mean) / local_std

        return local_scores


def run_tests():
    """Run comparison tests on synthetic data."""

    print("="*70)
    print("CONDITIONAL ISOLATION FOREST EXPERIMENTS")
    print("="*70)

    methods = {
        'IF': IFBaseline(n_estimators=100, random_state=42),
        'ROCOD': ROCOD(n_neighbors=50, random_state=42),
        'PNKIF': PNKDIFNoMLP(PNKDIFConfig(n_neighbors=50, n_trees=100, random_state=42)),
        'ResidualIF_Ridge': ResidualIF(n_estimators=100, regressor='ridge', random_state=42),
        'ResidualIF_RF': ResidualIF(n_estimators=100, regressor='rf', random_state=42),
        'HybridCIF_max': HybridContextIF(n_neighbors=50, n_estimators=100, combine='max', random_state=42),
        'HybridCIF_mean': HybridContextIF(n_neighbors=50, n_estimators=100, combine='mean', random_state=42),
        # 'LocalIF': LocalIF(n_neighbors=50, n_estimators=50, random_state=42),  # Too slow
    }

    # Import real dataset loaders
    from data.conquest_datasets import load_cardio_odds, load_ionosphere_odds, load_vowels_odds

    datasets = {
        'Syn-Cluster': lambda: generate_syn_cluster(n_samples=2000, random_state=42),
        'Syn-HighDim': lambda: generate_syn_highdim_context(n_samples=2000, random_state=42),
        'Cardio': load_cardio_odds,
        'Ionosphere': load_ionosphere_odds,
        'Vowels': load_vowels_odds,
    }

    results = []

    for ds_name, ds_loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        context, behavior, y = ds_loader()
        print(f"N={len(y)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
              f"anom={100*y.sum()/len(y):.1f}%")

        for method_name, method in methods.items():
            try:
                start = time.time()
                scores = method.fit_predict(context, behavior)
                elapsed = time.time() - start

                auroc = roc_auc_score(y, scores)
                print(f"  {method_name:20s}: AUROC={auroc:.3f}  ({elapsed:.2f}s)")

                results.append({
                    'dataset': ds_name,
                    'method': method_name,
                    'auroc': auroc,
                    'time': elapsed
                })
            except Exception as e:
                print(f"  {method_name:20s}: ERROR - {e}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.pivot(index='method', columns='dataset', values='auroc').round(3)
    print(pivot.to_string())

    # Save results
    df.to_csv(Path(__file__).parent.parent / 'results' / 'conditional_if_test.csv', index=False)
    print("\nResults saved to results/conditional_if_test.csv")


if __name__ == "__main__":
    run_tests()
