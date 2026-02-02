"""
ConQuest benchmark dataset loaders (DAMI and ODDS datasets).
These datasets are used for contextual anomaly detection benchmarking.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from scipy.io import loadmat

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw' / 'conquest'
ODDS_DIR = RAW_DIR  # ODDS .mat files
DAMI_DIR = RAW_DIR  # DAMI .arff files


def parse_arff(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse ARFF file and return features and labels."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse @ATTRIBUTE declarations to find outlier column
    attributes = []
    data_start = None
    outlier_col = None

    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        if line_stripped.startswith('@attribute'):
            # Extract attribute name
            parts = line.strip().split()
            if len(parts) >= 2:
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
                if 'outlier' in attr_name.lower() or 'class' in attr_name.lower():
                    outlier_col = len(attributes) - 1
        elif line_stripped == '@data':
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No @data section found in {filepath}")

    # Default to last column if no outlier column found
    if outlier_col is None:
        outlier_col = len(attributes) - 1

    # Parse data rows
    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if line and not line.startswith('%'):
            values = line.split(',')
            data.append(values)

    data = np.array(data)

    # Extract features (all columns except outlier and id)
    feature_cols = [i for i in range(data.shape[1])
                    if i != outlier_col and attributes[i].lower() != 'id']
    X = data[:, feature_cols].astype(float)
    y_raw = data[:, outlier_col]

    # Convert labels: 'yes'/'outlier'/1 -> 1, else -> 0
    y = np.zeros(len(y_raw), dtype=int)
    for i, val in enumerate(y_raw):
        val_lower = str(val).lower().strip()
        if val_lower in ['yes', 'outlier', '1', "'yes'", "'outlier'"]:
            y[i] = 1

    return X, y


def load_odds_mat(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ODDS dataset from .mat file."""
    filepath = ODDS_DIR / f"{name}.mat"
    if not filepath.exists():
        raise FileNotFoundError(f"ODDS dataset not found: {filepath}")

    mat = loadmat(filepath)
    X = mat['X']
    y = mat['y'].flatten()
    return X, y


def split_context_behavior(X: np.ndarray, n_context: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split features into context and behavior.
    Uses first n_context features as context, rest as behavior.
    """
    context = X[:, :n_context]
    behavior = X[:, n_context:]
    return context, behavior


# =============================================================================
# ODDS Datasets
# =============================================================================

def load_cardio_odds(n_context: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Cardio dataset from ODDS.
    Total: 21 features, 1831 samples, 9.6% anomalies
    """
    X, y = load_odds_mat('cardio')
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_ionosphere_odds(n_context: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Ionosphere dataset from ODDS.
    Total: 33 features, 351 samples, 35.9% anomalies
    """
    X, y = load_odds_mat('ionosphere')
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_arrhythmia_odds(n_context: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Arrhythmia dataset from ODDS.
    Total: 274 features, 452 samples, 14.6% anomalies
    """
    X, y = load_odds_mat('arrhythmia')
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_pima_odds(n_context: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Pima Indians Diabetes dataset from ODDS.
    Total: 8 features, 768 samples, 34.9% anomalies
    """
    X, y = load_odds_mat('pima')
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_vowels_odds(n_context: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Vowels dataset from ODDS.
    Total: 12 features, 1456 samples, 3.4% anomalies
    """
    X, y = load_odds_mat('vowels')
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_wbc_odds(n_context: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Wisconsin Breast Cancer dataset from ODDS.
    Total: 9 features, 223 samples, 4.5% anomalies (after removing NaN rows)
    """
    X, y = load_odds_mat('wbc')
    # Remove rows with NaN
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


# =============================================================================
# DAMI Datasets
# =============================================================================

def load_glass_dami(n_context: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Glass dataset from DAMI.
    Total: 9 features
    """
    filepath = DAMI_DIR / 'Glass.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_ionosphere_dami(n_context: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Ionosphere dataset from DAMI.
    """
    filepath = DAMI_DIR / 'Ionosphere.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_heartdisease_dami(n_context: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Heart Disease dataset from DAMI.
    """
    filepath = DAMI_DIR / 'HeartDisease.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_lymphography_dami(n_context: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Lymphography dataset from DAMI.
    Note: This version has only 3 features after categorical removal.
    """
    filepath = DAMI_DIR / 'Lymphography.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_wbc_dami(n_context: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load WBC dataset from DAMI.
    """
    filepath = DAMI_DIR / 'WBC.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


def load_wdbc_dami(n_context: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load WDBC dataset from DAMI.
    """
    filepath = DAMI_DIR / 'WDBC.arff'
    X, y = parse_arff(filepath)
    context, behavior = split_context_behavior(X, n_context)
    return context, behavior, y


# =============================================================================
# Dataset Registry
# =============================================================================

CONQUEST_DATASETS = {
    # ODDS datasets
    'cardio_odds': load_cardio_odds,
    'ionosphere_odds': load_ionosphere_odds,
    'arrhythmia_odds': load_arrhythmia_odds,
    'pima_odds': load_pima_odds,
    'vowels_odds': load_vowels_odds,
    'wbc_odds': load_wbc_odds,
    # DAMI datasets
    'glass_dami': load_glass_dami,
    'ionosphere_dami': load_ionosphere_dami,
    'heartdisease_dami': load_heartdisease_dami,
    'lymphography_dami': load_lymphography_dami,
    'wbc_dami': load_wbc_dami,
    'wdbc_dami': load_wdbc_dami,
}


def load_conquest_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a ConQuest dataset by name."""
    if name not in CONQUEST_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(CONQUEST_DATASETS.keys())}")
    return CONQUEST_DATASETS[name]()


def list_conquest_datasets() -> List[str]:
    """List all available ConQuest datasets."""
    return list(CONQUEST_DATASETS.keys())


if __name__ == "__main__":
    # Test loading
    print("Testing ConQuest dataset loaders...")
    for name in list_conquest_datasets():
        try:
            ctx, beh, y = load_conquest_dataset(name)
            n_anom = y.sum()
            print(f"{name:20s}: N={len(y):5d}, d_c={ctx.shape[1]:2d}, d_y={beh.shape[1]:2d}, anom={100*n_anom/len(y):.1f}%")
        except Exception as e:
            print(f"{name:20s}: ERROR - {e}")
