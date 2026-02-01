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
PROCESSED_DIR = DATA_DIR / 'processed'

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
            'SAML-D.csv', 'SAML_D.csv', 'saml.csv', 'synthetic_aml.csv',
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

    # For very large files, use chunked reading with sampling
    if max_samples:
        df = pd.read_csv(data_path, nrows=max_samples * 2)  # Read extra for sampling
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=random_state)
    else:
        df = pd.read_csv(data_path)

    # Check for label column
    label_cols = ['Is_laundering', 'Is_Laundering', 'is_laundering', 'isFraud', 'label', 'Label']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print(f"Warning: No label column found. Available: {df.columns.tolist()}")
        return generate_synthetic_fallback("saml", 10000, 5, 6, 0.05, random_state)

    labels = df[label_col].values.astype(int)

    # SAML-D specific feature extraction
    # Columns: Time, Date, Sender_account, Receiver_account, Amount,
    #          Payment_currency, Received_currency, Sender_bank_location,
    #          Receiver_bank_location, Payment_type, Is_laundering, Laundering_type

    # Context: geographic and payment type info (categorical)
    context_cols = ['Sender_bank_location', 'Receiver_bank_location',
                    'Payment_currency', 'Received_currency', 'Payment_type']
    context_cols = [c for c in context_cols if c in df.columns]

    if len(context_cols) > 0:
        context_df = df[context_cols].copy()
        for col in context_df.columns:
            context_df[col] = pd.Categorical(context_df[col]).codes
        context = context_df.values.astype(float)
    else:
        # Fallback
        context = np.zeros((len(df), 1))

    # Behavior: Amount and derived features
    behavior_cols = []
    if 'Amount' in df.columns:
        behavior_cols.append('Amount')

    if len(behavior_cols) > 0:
        behavior = df[behavior_cols].values.astype(float)
        # Add log-transformed amount as additional feature
        log_amount = np.log1p(np.abs(behavior))
        behavior = np.column_stack([behavior, log_amount])
    else:
        behavior = np.zeros((len(df), 1))

    # Ensure we have features
    if context.shape[1] == 0:
        context = np.zeros((len(df), 1))
    if behavior.shape[1] == 0:
        behavior = np.zeros((len(df), 1))

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
# SAML-D: Account-Level Aggregation (per design doc)
# =============================================================================

def load_saml_account_level(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None,
    max_samples: Optional[int] = None,  # Ignored - uses full pre-aggregated data
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SAML-D with account-level aggregation as per DATASET_FEATURES.md.

    Note: max_samples is ignored since we use pre-aggregated account-level data.

    This aggregates transaction-level data to account level:
    - Context: Distribution profile (one-hot counts of locations, currencies, payment types)
    - Behavior: Transaction statistics (count, sum, mean, std, max, min, range,
                unique receivers, active days, velocity features)
    - Label: max(Is_laundering) per account

    Processed data is cached in data/processed/ folder since aggregation is expensive.

    Returns:
        context, behavior, labels (all at account level)
    """
    ensure_dirs()

    # Check for processed file first (no random_state dependency - aggregation is deterministic)
    processed_path = PROCESSED_DIR / 'saml_account_level.pkl'

    if processed_path.exists():
        print(f"Loading processed SAML account-level data from {processed_path}")
        with open(processed_path, 'rb') as f:
            data = pickle.load(f)
        context, behavior, labels = data['context'], data['behavior'], data['labels']
        print(f"SAML (account-level) loaded: N={len(labels)}, d_c={context.shape[1]}, "
              f"d_y={behavior.shape[1]}, anomaly_rate={labels.mean():.4f}")
        return context, behavior, labels

    # Find raw data file
    if data_path is None:
        possible_names = [
            'SAML-D.csv', 'SAML_D.csv', 'saml.csv', 'synthetic_aml.csv',
            'HI-Small_Trans.csv', 'transactions.csv'
        ]
        for name in possible_names:
            if (RAW_DIR / name).exists():
                data_path = RAW_DIR / name
                break

    if data_path is None or not Path(data_path).exists():
        raise FileNotFoundError(
            f"SAML-D data not found. Please download from:\n"
            f"https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml\n"
            f"And place CSV file in: {RAW_DIR}"
        )

    print(f"Loading and aggregating SAML-D from {data_path}...")
    print("This may take a few minutes for 9.5M transactions...")

    # Read raw transaction data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} transactions")

    # Identify label column
    label_cols = ['Is_laundering', 'Is_Laundering', 'is_laundering', 'isFraud', 'label']
    label_col = None
    for col in label_cols:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No label column found. Available: {df.columns.tolist()}")

    # =========================================================================
    # Aggregate to account level
    # =========================================================================
    print("Aggregating to account level...")

    # Context: Distribution profiles (one-hot counts per category)
    # Bank location distribution
    loc_counts = df.groupby(['Sender_account', 'Sender_bank_location']).size().unstack(fill_value=0)
    loc_counts.columns = [f'loc_{c}' for c in loc_counts.columns]

    # Currency distribution
    curr_counts = df.groupby(['Sender_account', 'Payment_currency']).size().unstack(fill_value=0)
    curr_counts.columns = [f'curr_{c}' for c in curr_counts.columns]

    # Payment type distribution
    ptype_counts = df.groupby(['Sender_account', 'Payment_type']).size().unstack(fill_value=0)
    ptype_counts.columns = [f'ptype_{c}' for c in ptype_counts.columns]

    # Combine context features
    context_df = loc_counts.join(curr_counts, how='outer').join(ptype_counts, how='outer')
    context_df = context_df.fillna(0)

    # Behavior: Transaction statistics
    behavior_agg = df.groupby('Sender_account').agg({
        'Amount': ['count', 'sum', 'mean', 'std', 'max', 'min'],
        'Receiver_account': 'nunique',
        'Date': 'nunique',
        label_col: 'max',
    })
    behavior_agg.columns = ['tx_count', 'tx_sum', 'tx_mean', 'tx_std', 'tx_max', 'tx_min',
                            'unique_receivers', 'active_days', 'is_laundering']

    # Derived features
    behavior_agg['tx_range'] = behavior_agg['tx_max'] - behavior_agg['tx_min']
    behavior_agg['avg_tx_per_day'] = behavior_agg['tx_count'] / behavior_agg['active_days'].clip(lower=1)
    behavior_agg['avg_receivers_per_day'] = behavior_agg['unique_receivers'] / behavior_agg['active_days'].clip(lower=1)

    # Fill NaN in std (accounts with single transaction)
    behavior_agg['tx_std'] = behavior_agg['tx_std'].fillna(0)

    # Extract labels
    labels = behavior_agg['is_laundering'].values.astype(int)

    # Behavior features (exclude label)
    behavior_cols = ['tx_count', 'tx_sum', 'tx_mean', 'tx_std', 'tx_max', 'tx_min',
                     'tx_range', 'unique_receivers', 'active_days',
                     'avg_tx_per_day', 'avg_receivers_per_day']
    behavior_df = behavior_agg[behavior_cols]

    # Align indices
    common_accounts = context_df.index.intersection(behavior_df.index)
    context_df = context_df.loc[common_accounts]
    behavior_df = behavior_df.loc[common_accounts]
    labels = behavior_agg.loc[common_accounts, 'is_laundering'].values.astype(int)

    print(f"Aggregated to {len(common_accounts):,} accounts")
    print(f"Context features: {context_df.shape[1]}, Behavior features: {behavior_df.shape[1]}")
    print(f"Anomaly rate: {labels.mean():.4f}")

    # Convert to numpy
    context = context_df.values.astype(float)
    behavior = behavior_df.values.astype(float)

    # Standardize
    context = (context - context.mean(axis=0)) / (context.std(axis=0) + 1e-8)
    behavior = (behavior - behavior.mean(axis=0)) / (behavior.std(axis=0) + 1e-8)

    # Handle any remaining NaN
    context = np.nan_to_num(context, nan=0.0)
    behavior = np.nan_to_num(behavior, nan=0.0)

    # Save processed data
    print(f"Saving processed data to {processed_path}")
    with open(processed_path, 'wb') as f:
        pickle.dump({
            'context': context,
            'behavior': behavior,
            'labels': labels,
            'context_columns': list(context_df.columns),
            'behavior_columns': behavior_cols,
        }, f)

    print(f"SAML (account-level) processed: N={len(labels)}, d_c={context.shape[1]}, "
          f"d_y={behavior.shape[1]}, anomaly_rate={labels.mean():.4f}")

    return context, behavior, labels


def load_saml_with_injection(
    data_path: Optional[str] = None,
    random_state: Optional[int] = None,
    max_samples: Optional[int] = None,  # Ignored
    injection_rate: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SAML-D account-level data with contextual anomaly injection.

    Injection Strategy: Domestic → Cross-border Behavior Swap

    Accounts that are primarily domestic (>80% same-country transactions)
    are given the behavioral statistics of cross-border accounts.
    This creates contextual anomalies: the behavior values are normal globally
    but unusual for accounts with domestic transaction profiles.

    Parameters:
        injection_rate: Fraction of normal domestic accounts to inject (default 2%)
        random_state: Random seed for reproducibility

    Returns:
        context, behavior, labels (with injected contextual anomalies)
    """
    rng = np.random.RandomState(random_state)

    # Load the processed account-level data
    processed_path = PROCESSED_DIR / 'saml_account_level.pkl'

    if not processed_path.exists():
        # Generate it first
        load_saml_account_level()

    print(f"Loading processed SAML data for injection...")
    with open(processed_path, 'rb') as f:
        data = pickle.load(f)

    context = data['context'].copy()
    behavior = data['behavior'].copy()
    labels = data['labels'].copy()
    context_columns = data['context_columns']
    behavior_columns = data['behavior_columns']

    # =========================================================================
    # Identify Domestic vs Cross-border accounts using context features
    # =========================================================================
    # Context columns are like: loc_UK, loc_US, loc_DE, curr_UK pounds, ptype_Cash Deposit, etc.

    # Find location columns (they start with 'loc_')
    loc_cols = [i for i, col in enumerate(context_columns) if col.startswith('loc_')]

    if len(loc_cols) < 2:
        print("Warning: Not enough location columns for injection. Returning original data.")
        return context, behavior, labels

    # For each account, calculate "domestic ratio" based on location distribution
    # High values in a single location column = domestic, spread across = cross-border
    loc_features = context[:, loc_cols]  # Already standardized, so we work with relative values

    # Use entropy-like measure: domestic accounts have low entropy (concentrated in one location)
    # Add small epsilon to avoid log(0)
    loc_probs = np.abs(loc_features) / (np.abs(loc_features).sum(axis=1, keepdims=True) + 1e-8)
    loc_entropy = -np.sum(loc_probs * np.log(loc_probs + 1e-8), axis=1)

    # Low entropy = domestic, high entropy = cross-border
    entropy_threshold_domestic = np.percentile(loc_entropy, 25)  # Bottom 25% = most domestic
    entropy_threshold_crossborder = np.percentile(loc_entropy, 75)  # Top 25% = most cross-border

    domestic_mask = (loc_entropy <= entropy_threshold_domestic) & (labels == 0)
    crossborder_mask = loc_entropy >= entropy_threshold_crossborder

    domestic_idx = np.where(domestic_mask)[0]
    crossborder_idx = np.where(crossborder_mask)[0]

    print(f"Domestic accounts (low entropy, normal): {len(domestic_idx)}")
    print(f"Cross-border accounts (high entropy): {len(crossborder_idx)}")

    if len(domestic_idx) == 0 or len(crossborder_idx) == 0:
        print("Warning: Could not identify domestic/cross-border accounts. Returning original data.")
        return context, behavior, labels

    # =========================================================================
    # Inject contextual anomalies
    # =========================================================================
    n_inject = int(len(labels) * injection_rate)
    n_inject = min(n_inject, len(domestic_idx))

    # Select random domestic accounts to inject
    inject_idx = rng.choice(domestic_idx, n_inject, replace=False)

    # Calculate cross-border behavior statistics (in standardized space)
    crossborder_behavior_mean = behavior[crossborder_idx].mean(axis=0)
    crossborder_behavior_std = behavior[crossborder_idx].std(axis=0) + 1e-6

    # Replace domestic behavior with cross-border-like behavior
    # Add some noise to avoid exact duplicates
    for i in inject_idx:
        noise = rng.randn(behavior.shape[1]) * 0.3  # 30% of std as noise
        behavior[i] = crossborder_behavior_mean + noise * crossborder_behavior_std
        labels[i] = 1  # Mark as anomaly

    # Count original vs injected anomalies
    n_original_anomalies = data['labels'].sum()
    n_total_anomalies = labels.sum()

    print(f"Injected {n_inject} contextual anomalies (domestic accounts with cross-border behavior)")
    print(f"Original anomalies: {n_original_anomalies}, Total anomalies: {n_total_anomalies}")
    print(f"New anomaly rate: {labels.mean():.4f}")

    return context, behavior, labels


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
        # Check multiple possible locations
        possible_paths = [
            RAW_DIR / 'train_transaction.csv',
            RAW_DIR / 'IEEE-CIS Fraud Detection' / 'train_transaction.csv',
            RAW_DIR / 'ieee-cis' / 'train_transaction.csv',
        ]
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        if data_path is None:
            data_path = possible_paths[0]  # For error message

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
        possible_names = [
            'Synthetic Financial Datasets For Fraud Detection.csv',
            'PS_20174392719_1491204439457_log.csv',
            'paysim.csv', 'PaySim.csv'
        ]
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
        # Try common names
        possible_names = [
            'creditcard.csv',
            'Credit Card Fraud Detection.csv',
            'credit_card.csv'
        ]
        for name in possible_names:
            if (RAW_DIR / name).exists():
                data_path = RAW_DIR / name
                break
        if data_path is None:
            data_path = RAW_DIR / 'creditcard.csv'  # For error message

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

    The ODDS thyroid dataset has 6 features total.
    Context: First 3 features (demographic/test type)
    Behavior: Last 3 features (measurements)
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

    # Thyroid from ODDS has 6 features total
    # Split: first 3 = context (demographic), last 3 = behavior (measurements)
    n_features = X.shape[1]
    n_context = n_features // 2  # Use half for context
    context = X[:, :n_context]
    behavior = X[:, n_context:]

    # Ensure we have at least 1 feature in each
    if behavior.shape[1] == 0:
        # Fallback: use first feature as context, rest as behavior
        context = X[:, :1]
        behavior = X[:, 1:]

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
    'saml': load_saml_account_level,  # Account-level aggregation (real anomalies only)
    'saml_injected': load_saml_with_injection,  # With contextual anomaly injection
    'saml_tx': load_saml,  # Original transaction-level (deprecated)
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
