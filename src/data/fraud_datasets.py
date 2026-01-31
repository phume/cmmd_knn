"""
Fraud detection dataset loaders: SAML-D, IEEE-CIS, PaySim, Credit Card.
Features are cached after first extraction to avoid recomputation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import hashlib
import pickle
import zipfile
import os

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
CACHE_DIR = DATA_DIR / 'cache'

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_path(dataset_name: str, config_hash: str) -> Path:
    """Get path for cached features."""
    return CACHE_DIR / f"{dataset_name}_{config_hash}.pkl"

def load_cached(cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load cached features if they exist."""
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_cached(cache_path: Path, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Save extracted features to cache."""
    ensure_dirs()
    print(f"Caching features to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)


# =============================================================================
# SAML-D: Synthetic AML Dataset
# =============================================================================

def load_saml(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SAML-D (Synthetic Anti-Money Laundering Dataset).

    Source: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

    Context features: Customer type, account age, geographic risk, payment type
    Behavioral features: Transaction amount stats, frequency, velocity

    Parameters:
        data_path: Path to the SAML CSV file (if None, looks in RAW_DIR)
        random_state: Random seed for sampling
        max_samples: Maximum samples to use (for faster testing)

    Returns:
        context, behavior, labels
    """
    config = f"saml_rs{random_state}_max{max_samples}"
    config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
    cache_path = get_cache_path("saml", config_hash)

    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    # Find data file
    if data_path is None:
        # Try common names
        possible_names = [
            'SAML_D.csv', 'saml.csv', 'synthetic_aml.csv',
            'HI-Small_Trans.csv', 'transactions.csv'
        ]
        for name in possible_names:
            if (RAW_DIR / name).exists():
                data_path = RAW_DIR / name
                break

    if data_path is None or not Path(data_path).exists():
        print("SAML-D data not found. Please download from:")
        print("https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml")
        print(f"And place CSV file in: {RAW_DIR}")
        return generate_synthetic_fallback("saml", 10000, 5, 6, 0.05, random_state)

    print(f"Loading SAML-D from {data_path}...")
    df = pd.read_csv(data_path)

    # Sample if needed
    rng = np.random.RandomState(random_state)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state)

    # Identify columns (adapt based on actual schema)
    # Common SAML-D schema has: Account, Amount, Timestamp, Is_Laundering, etc.

    # Check for label column
    label_cols = ['Is_Laundering', 'is_laundering', 'isFraud', 'label', 'Label']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print(f"Warning: No label column found. Available: {df.columns.tolist()}")
        return generate_synthetic_fallback("saml", len(df), 5, 6, 0.05, random_state)

    labels = df[label_col].values.astype(int)

    # Extract features based on available columns
    # Context: categorical/identifier features
    # Behavior: numerical transaction features

    context_candidates = ['Account', 'Account_Type', 'Customer_Type', 'Bank',
                          'Payment_Type', 'Payment_Format', 'Is_High_Risk_Country']
    behavior_candidates = ['Amount', 'Amount_Received', 'Amount_Paid',
                           'Timestamp', 'Transaction_Count']

    context_cols = [c for c in context_candidates if c in df.columns]
    behavior_cols = [c for c in behavior_candidates if c in df.columns]

    # If we have account-level data, aggregate transactions
    if 'Account' in df.columns and len(context_cols) > 0:
        # Aggregate to account level
        agg_dict = {}
        for col in behavior_cols:
            if df[col].dtype in [np.float64, np.int64]:
                agg_dict[col] = ['mean', 'std', 'min', 'max', 'count']

        if agg_dict:
            account_df = df.groupby('Account').agg(agg_dict)
            account_df.columns = ['_'.join(col) for col in account_df.columns]
            account_labels = df.groupby('Account')[label_col].max()  # 1 if any transaction is fraud

            behavior = account_df.values
            labels = account_labels.values

            # Get context for each account (first occurrence)
            context_df = df.groupby('Account')[context_cols].first()
            for col in context_df.select_dtypes(include=['object']).columns:
                context_df[col] = pd.Categorical(context_df[col]).codes
            context = context_df.values.astype(float)
        else:
            # Fallback: use available numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != label_col]

            n_ctx = min(5, len(numeric_cols) // 2)
            context = df[numeric_cols[:n_ctx]].values.astype(float)
            behavior = df[numeric_cols[n_ctx:]].values.astype(float)
    else:
        # Direct feature extraction
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != label_col]

        n_ctx = min(5, len(numeric_cols) // 2)
        context = df[numeric_cols[:n_ctx]].values.astype(float)
        behavior = df[numeric_cols[n_ctx:]].values.astype(float)

    # Handle NaN
    context = np.nan_to_num(context, nan=0.0)
    behavior = np.nan_to_num(behavior, nan=0.0)

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    result = (context, behavior, labels)
    save_cached(cache_path, result)

    print(f"SAML-D loaded: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.3f}")

    return result


# =============================================================================
# IEEE-CIS Fraud Detection
# =============================================================================

def load_ieee_cis(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None,
    max_samples: Optional[int] = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load IEEE-CIS Fraud Detection dataset.

    Source: https://www.kaggle.com/competitions/ieee-fraud-detection

    Context: Card info, address, device info, ProductCD
    Behavior: TransactionAmt, time deltas (D1-D15), counts (C1-C14)

    Parameters:
        data_path: Path to train_transaction.csv
        random_state: Random seed
        max_samples: Max samples (dataset is 590K rows)

    Returns:
        context, behavior, labels
    """
    config = f"ieee_rs{random_state}_max{max_samples}"
    config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
    cache_path = get_cache_path("ieee_cis", config_hash)

    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    if data_path is None:
        data_path = RAW_DIR / 'train_transaction.csv'

    if not Path(data_path).exists():
        print("IEEE-CIS data not found. Please download from:")
        print("https://www.kaggle.com/competitions/ieee-fraud-detection")
        print(f"And place train_transaction.csv in: {RAW_DIR}")
        return generate_synthetic_fallback("ieee_cis", 10000, 10, 20, 0.035, random_state)

    print(f"Loading IEEE-CIS from {data_path}...")

    # Load with sampling for memory efficiency
    df = pd.read_csv(data_path, nrows=max_samples)

    labels = df['isFraud'].values

    # Context features: card, addr, ProductCD
    context_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
    context_cols = [c for c in context_cols if c in df.columns]

    context_df = df[context_cols].copy()
    for col in context_df.columns:
        if context_df[col].dtype == 'object':
            context_df[col] = pd.Categorical(context_df[col]).codes
        context_df[col] = context_df[col].fillna(-1)
    context = context_df.values.astype(float)

    # Behavior features: amount, time deltas, counts
    behavior_cols = ['TransactionAmt', 'TransactionDT']
    behavior_cols += [f'C{i}' for i in range(1, 15) if f'C{i}' in df.columns]
    behavior_cols += [f'D{i}' for i in range(1, 16) if f'D{i}' in df.columns]
    behavior_cols = [c for c in behavior_cols if c in df.columns]

    behavior = df[behavior_cols].fillna(0).values.astype(float)

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    result = (context, behavior, labels)
    save_cached(cache_path, result)

    print(f"IEEE-CIS loaded: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.3f}")

    return result


# =============================================================================
# PaySim
# =============================================================================

def load_paysim(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None,
    max_samples: Optional[int] = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PaySim mobile money fraud dataset.

    Source: https://www.kaggle.com/datasets/ealaxi/paysim1

    Context: Transaction type, name identifiers
    Behavior: Amount, balance changes

    Parameters:
        data_path: Path to PS_20174392719_1491204439457_log.csv
        random_state: Random seed
        max_samples: Max samples (dataset is 6.3M rows)

    Returns:
        context, behavior, labels
    """
    config = f"paysim_rs{random_state}_max{max_samples}"
    config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
    cache_path = get_cache_path("paysim", config_hash)

    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    if data_path is None:
        # Try common names
        possible_names = ['PS_20174392719_1491204439457_log.csv', 'paysim.csv', 'PaySim.csv']
        for name in possible_names:
            if (RAW_DIR / name).exists():
                data_path = RAW_DIR / name
                break

    if data_path is None or not Path(data_path).exists():
        print("PaySim data not found. Please download from:")
        print("https://www.kaggle.com/datasets/ealaxi/paysim1")
        print(f"And place CSV file in: {RAW_DIR}")
        return generate_synthetic_fallback("paysim", 10000, 3, 5, 0.001, random_state)

    print(f"Loading PaySim from {data_path}...")

    df = pd.read_csv(data_path, nrows=max_samples)

    labels = df['isFraud'].values

    # Context: transaction type
    context_df = pd.DataFrame()
    context_df['type'] = pd.Categorical(df['type']).codes
    context_df['step'] = df['step']  # Time step
    # Hash account names to numeric (preserve some identity info)
    context_df['nameOrig_hash'] = df['nameOrig'].apply(lambda x: hash(x) % 10000)
    context = context_df.values.astype(float)

    # Behavior: amounts and balances
    behavior_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    behavior = df[behavior_cols].values.astype(float)

    # Add derived features
    balance_change_orig = behavior[:, 2] - behavior[:, 1]  # newbalanceOrig - oldbalanceOrg
    balance_change_dest = behavior[:, 4] - behavior[:, 3]  # newbalanceDest - oldbalanceDest
    behavior = np.column_stack([behavior, balance_change_orig, balance_change_dest])

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    result = (context, behavior, labels)
    save_cached(cache_path, result)

    print(f"PaySim loaded: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.3f}")

    return result


# =============================================================================
# Credit Card Fraud (Kaggle)
# =============================================================================

def load_creditcard(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Credit Card Fraud Detection dataset.

    Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    Context: Time, Amount (as proxy for transaction context)
    Behavior: V1-V28 (PCA components)

    Returns:
        context, behavior, labels
    """
    config = f"creditcard_rs{random_state}"
    config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
    cache_path = get_cache_path("creditcard", config_hash)

    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    if data_path is None:
        data_path = RAW_DIR / 'creditcard.csv'

    if not Path(data_path).exists():
        print("Credit Card Fraud data not found. Please download from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"And place creditcard.csv in: {RAW_DIR}")
        return generate_synthetic_fallback("creditcard", 10000, 2, 28, 0.002, random_state)

    print(f"Loading Credit Card Fraud from {data_path}...")

    df = pd.read_csv(data_path)

    labels = df['Class'].values

    # Context: Time and Amount
    context = df[['Time', 'Amount']].values.astype(float)

    # Behavior: V1-V28 (PCA components)
    v_cols = [f'V{i}' for i in range(1, 29)]
    behavior = df[v_cols].values.astype(float)

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    result = (context, behavior, labels)
    save_cached(cache_path, result)

    print(f"Credit Card loaded: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={labels.mean():.4f}")

    return result


# =============================================================================
# Thyroid (ODDS)
# =============================================================================

def load_thyroid(random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Thyroid dataset from ODDS.

    Context: First 6 features (demographic/test type)
    Behavior: Remaining 15 features (measurements)
    """
    import urllib.request
    from scipy.io import loadmat

    config = f"thyroid_rs{random_state}"
    config_hash = hashlib.md5(config.encode()).hexdigest()[:8]
    cache_path = get_cache_path("thyroid", config_hash)

    cached = load_cached(cache_path)
    if cached is not None:
        return cached

    ensure_dirs()
    filepath = RAW_DIR / 'thyroid.mat'

    if not filepath.exists():
        url = "https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1"
        print(f"Downloading thyroid dataset...")
        urllib.request.urlretrieve(url, filepath)

    data = loadmat(filepath)
    X = data['X']
    y = data['y'].ravel()

    # Split: first 6 = context, rest = behavior
    context = X[:, :6]
    behavior = X[:, 6:]

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    result = (context, behavior, y)
    save_cached(cache_path, result)

    print(f"Thyroid loaded: N={len(y)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anomaly_rate={y.mean():.3f}")

    return result


# =============================================================================
# Fallback synthetic generator
# =============================================================================

def generate_synthetic_fallback(
    name: str,
    n_samples: int,
    d_context: int,
    d_behavior: int,
    anomaly_rate: float,
    random_state: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic fallback when real data unavailable."""
    print(f"Generating synthetic fallback for {name}...")

    rng = np.random.RandomState(random_state)

    context = rng.randn(n_samples, d_context)
    behavior = rng.randn(n_samples, d_behavior)
    behavior += 0.5 * context[:, :min(d_context, d_behavior)]

    n_anomalies = int(n_samples * anomaly_rate)
    labels = np.zeros(n_samples)
    anomaly_idx = rng.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_idx] = 1
    behavior[anomaly_idx] += rng.randn(n_anomalies, d_behavior) * 3

    return context, behavior, labels


# =============================================================================
# Dataset Registry
# =============================================================================

FRAUD_DATASETS = {
    'saml': load_saml,
    'ieee_cis': load_ieee_cis,
    'paysim': load_paysim,
    'creditcard': load_creditcard,
    'thyroid': load_thyroid,
}


def get_fraud_dataset(name: str, random_state: Optional[int] = None, **kwargs):
    """Get a fraud detection dataset by name."""
    if name not in FRAUD_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(FRAUD_DATASETS.keys())}")
    return FRAUD_DATASETS[name](random_state=random_state, **kwargs)
