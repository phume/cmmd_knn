"""
Real dataset loaders for contextual anomaly detection.
Includes UCI Adult, Bank Marketing, and synthetic anomaly injection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
import os

DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw'


def download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't exist."""
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")


# =============================================================================
# UCI ADULT DATASET
# =============================================================================

ADULT_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

ADULT_CONTEXT_COLS = ['age', 'education_num', 'sex', 'race', 'native_country']
ADULT_BEHAVIOR_COLS = ['hours_per_week', 'capital_gain', 'capital_loss', 'fnlwgt']


def load_adult(
    anomaly_type: str = 'synthetic_shift',
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UCI Adult dataset with context/behavior split.

    Parameters:
        anomaly_type: 'original' (income>50K), 'synthetic_shift', 'synthetic_swap'
        anomaly_rate: Rate for synthetic injection
        random_state: Random seed

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    # Download if needed
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    filepath = DATA_DIR / 'adult.data'
    download_file(url, filepath)

    # Load data
    df = pd.read_csv(filepath, names=ADULT_COLUMNS, skipinitialspace=True)
    df = df.replace('?', np.nan).dropna()

    # Encode categoricals for context
    context_df = df[ADULT_CONTEXT_COLS].copy()
    for col in context_df.select_dtypes(include=['object']).columns:
        context_df[col] = pd.Categorical(context_df[col]).codes

    context = context_df.values.astype(float)

    # Standardize context
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)

    # Behavior features
    behavior = df[ADULT_BEHAVIOR_COLS].values.astype(float)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    # Labels based on anomaly type
    if anomaly_type == 'original':
        labels = (df['income'].str.strip() == '>50K').astype(int).values
    else:
        labels = inject_anomalies(
            context, behavior, anomaly_type, anomaly_rate, rng
        )

    return context, behavior, labels


# =============================================================================
# UCI BANK MARKETING DATASET
# =============================================================================

BANK_CONTEXT_COLS = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
BANK_BEHAVIOR_COLS = ['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


def load_bank(
    anomaly_type: str = 'synthetic_shift',
    anomaly_rate: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UCI Bank Marketing dataset with context/behavior split.

    Parameters:
        anomaly_type: 'original' (subscription), 'synthetic_shift', 'synthetic_swap'
        anomaly_rate: Rate for synthetic injection
        random_state: Random seed

    Returns:
        context, behavior, labels
    """
    rng = np.random.RandomState(random_state)

    # Download if needed
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = DATA_DIR / 'bank-additional.zip'
    csv_path = DATA_DIR / 'bank-additional' / 'bank-additional-full.csv'

    if not csv_path.exists():
        download_file(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)

    # Load data
    df = pd.read_csv(csv_path, sep=';')
    df = df.replace('unknown', np.nan).dropna()

    # Encode categoricals for context
    context_df = df[BANK_CONTEXT_COLS].copy()
    for col in context_df.select_dtypes(include=['object']).columns:
        context_df[col] = pd.Categorical(context_df[col]).codes

    context = context_df.values.astype(float)
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)

    # Behavior features
    behavior = df[BANK_BEHAVIOR_COLS].values.astype(float)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    # Labels
    if anomaly_type == 'original':
        labels = (df['y'] == 'yes').astype(int).values
    else:
        labels = inject_anomalies(
            context, behavior, anomaly_type, anomaly_rate, rng
        )

    return context, behavior, labels


# =============================================================================
# SYNTHETIC ANOMALY INJECTION
# =============================================================================

def inject_anomalies(
    context: np.ndarray,
    behavior: np.ndarray,
    injection_type: str,
    anomaly_rate: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Inject synthetic anomalies into real data.

    Parameters:
        context: Context features
        behavior: Behavioral features (will be modified in place)
        injection_type: 'synthetic_shift' or 'synthetic_swap'
        anomaly_rate: Fraction of samples to make anomalous
        rng: Random state

    Returns:
        labels: Binary anomaly labels
    """
    n_samples = len(context)
    n_anomalies = int(n_samples * anomaly_rate)
    labels = np.zeros(n_samples)

    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    if injection_type == 'synthetic_shift':
        # Add context-dependent shift to behavior
        from sklearn.neighbors import NearestNeighbors
        K = min(50, n_samples // 20)
        nn = NearestNeighbors(n_neighbors=K + 1)
        nn.fit(context)
        _, indices = nn.kneighbors(context[anomaly_idx])

        for i, idx in enumerate(anomaly_idx):
            peer_idx = indices[i, 1:]  # Exclude self
            peer_std = behavior[peer_idx].std(axis=0)
            shift = rng.choice([-1, 1], size=behavior.shape[1]) * 3 * peer_std
            behavior[idx] += shift

    elif injection_type == 'synthetic_swap':
        # Swap behavior with a different context cluster
        from sklearn.cluster import KMeans
        n_clusters = min(5, n_samples // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=rng.randint(0, 10000), n_init=10)
        cluster_labels = kmeans.fit_predict(context)

        for idx in anomaly_idx:
            current_cluster = cluster_labels[idx]
            other_clusters = np.where(cluster_labels != current_cluster)[0]
            if len(other_clusters) > 0:
                swap_idx = rng.choice(other_clusters)
                behavior[idx] = behavior[swap_idx].copy()

    return labels


# =============================================================================
# CARDIO DATASET (ODDS)
# =============================================================================

def load_cardio(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Cardio dataset from ODDS.

    Context: First 5 features (patient demographics)
    Behavior: Remaining 6 features (measurements)
    """
    url = "https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1"
    filepath = DATA_DIR / 'cardio.mat'

    try:
        download_file(url, filepath)
        from scipy.io import loadmat
        data = loadmat(filepath)
        X = data['X']
        y = data['y'].ravel()

        # Split: first 5 = context, rest = behavior
        context = X[:, :5]
        behavior = X[:, 5:]

        # Standardize
        context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
        behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

        return context, behavior, y
    except Exception as e:
        print(f"Could not load cardio dataset: {e}")
        # Return synthetic fallback
        return generate_fallback_dataset(1831, 5, 6, 0.096, random_state)


def generate_fallback_dataset(
    n_samples: int,
    d_context: int,
    d_behavior: int,
    anomaly_rate: float,
    random_state: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic dataset as fallback."""
    rng = np.random.RandomState(random_state)

    context = rng.randn(n_samples, d_context)
    behavior = rng.randn(n_samples, d_behavior)

    # Add context-behavior relationship
    behavior += 0.5 * context[:, :min(d_context, d_behavior)]

    n_anomalies = int(n_samples * anomaly_rate)
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # Add anomaly shift
    behavior[anomaly_idx] += rng.randn(n_anomalies, d_behavior) * 3

    return context, behavior, labels


# =============================================================================
# DATASET REGISTRY
# =============================================================================

REAL_DATASETS = {
    'adult_original': lambda rs: load_adult('original', random_state=rs),
    'adult_shift': lambda rs: load_adult('synthetic_shift', 0.05, rs),
    'adult_swap': lambda rs: load_adult('synthetic_swap', 0.05, rs),
    'bank_original': lambda rs: load_bank('original', random_state=rs),
    'bank_shift': lambda rs: load_bank('synthetic_shift', 0.05, rs),
    'bank_swap': lambda rs: load_bank('synthetic_swap', 0.05, rs),
    'cardio': load_cardio,
}


def get_real_dataset(name: str, random_state: Optional[int] = None):
    """Get a real dataset by name."""
    if name not in REAL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(REAL_DATASETS.keys())}")
    return REAL_DATASETS[name](random_state)
