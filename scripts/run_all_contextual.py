"""
Comprehensive Contextual Anomaly Detection Experiments.
Test PNKIF vs IF vs ROCOD on ALL datasets with contextual structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time
import warnings
warnings.filterwarnings('ignore')

from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline, IFConcat, ROCOD

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

SEEDS = [42, 123, 456, 789, 1011]


def run_methods(name, context, behavior, labels, K=None):
    """Run all methods on a dataset."""
    if K is None:
        K = min(100, len(labels) // 20)
    K = max(5, K)  # minimum K

    print(f"\n{'='*60}")
    print(f"{name}: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, anom={100*labels.mean():.2f}%")
    print('='*60)

    methods = {
        'IF': lambda s: IFBaseline(n_estimators=100, random_state=s),
        'IF_concat': lambda s: IFConcat(n_estimators=100, random_state=s),
        'ROCOD': lambda s: ROCOD(n_neighbors=K, random_state=s),
        'PNKIF': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=s)),
    }

    results = []
    for method_name, method_fn in methods.items():
        aurocs = []
        for seed in SEEDS:
            try:
                model = method_fn(seed)
                scores = model.fit_predict(context, behavior)
                auroc = roc_auc_score(labels, scores)
                aurocs.append(auroc)
            except Exception as e:
                print(f"  {method_name} error: {e}")
                aurocs.append(np.nan)

        mean_auroc = np.nanmean(aurocs)
        std_auroc = np.nanstd(aurocs)
        print(f"  {method_name:12s}: AUROC={mean_auroc:.3f}±{std_auroc:.3f}")

        results.append({
            'dataset': name, 'method': method_name,
            'auroc_mean': mean_auroc, 'auroc_std': std_auroc,
            'n_samples': len(labels), 'd_context': context.shape[1],
            'd_behavior': behavior.shape[1], 'anom_rate': labels.mean()
        })

    return results


# =============================================================================
# ODDS DATASETS (.mat files)
# =============================================================================

def load_cardio_contextual():
    """Cardio: Context = patient demographics, Behavior = ECG measurements."""
    mat = loadmat(DATA_DIR / 'conquest' / 'cardio.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    # First 5 features: age, height, weight, gender, ap_hi (demographics/vitals)
    # Remaining: ECG and other measurements
    context = StandardScaler().fit_transform(X[:, :5])
    behavior = StandardScaler().fit_transform(X[:, 5:])

    return context, behavior, y


def load_arrhythmia_contextual():
    """Arrhythmia: Context = patient info (first 50), Behavior = ECG (remaining)."""
    mat = loadmat(DATA_DIR / 'conquest' / 'arrhythmia.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    # First ~15 features are patient characteristics
    n_context = 15
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


def load_pima_contextual():
    """Pima: Context = age, BMI, pregnancies. Behavior = glucose, bp, insulin."""
    mat = loadmat(DATA_DIR / 'conquest' / 'pima.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    # Features: Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age
    # Context: Pregnancies (0), BMI (5), Age (7)
    # Behavior: Glucose (1), BP (2), SkinThickness (3), Insulin (4), DPF (6)
    context_idx = [0, 5, 7]
    behavior_idx = [1, 2, 3, 4, 6]

    context = StandardScaler().fit_transform(X[:, context_idx])
    behavior = StandardScaler().fit_transform(X[:, behavior_idx])

    return context, behavior, y


def load_ionosphere_contextual():
    """Ionosphere: Context = first 8 features, Behavior = remaining."""
    mat = loadmat(DATA_DIR / 'conquest' / 'ionosphere.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    n_context = 8
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


def load_wbc_contextual():
    """WBC: Context = first 5 features, Behavior = remaining."""
    mat = loadmat(DATA_DIR / 'conquest' / 'wbc.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    n_context = 5
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


def load_vowels_contextual():
    """Vowels: Context = first 3 features, Behavior = remaining."""
    mat = loadmat(DATA_DIR / 'conquest' / 'vowels.mat')
    X = mat['X'].astype(np.float64)
    y = mat['y'].ravel().astype(int)

    n_context = 3
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


# =============================================================================
# DAMI DATASETS (.arff files)
# =============================================================================

def parse_arff(filepath):
    """Parse ARFF file and return X, y."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find data section
    data_start = None
    attributes = []
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if line_lower.startswith('@attribute'):
            parts = line.strip().split()
            attr_name = parts[1]
            attributes.append(attr_name)
        elif line_lower.startswith('@data'):
            data_start = i + 1
            break

    # Find outlier column
    outlier_idx = None
    for i, attr in enumerate(attributes):
        if 'outlier' in attr.lower() or attr.lower() == 'class':
            outlier_idx = i
            break

    if outlier_idx is None:
        outlier_idx = len(attributes) - 1

    # Parse data
    data_lines = [l.strip() for l in lines[data_start:] if l.strip() and not l.startswith('%')]

    X_list = []
    y_list = []
    for line in data_lines:
        values = line.split(',')
        row = []
        for i, v in enumerate(values):
            if i == outlier_idx:
                # Parse label
                v_clean = v.strip().strip("'\"").lower()
                if v_clean in ['yes', 'o', 'outlier', '1', 'true']:
                    y_list.append(1)
                else:
                    y_list.append(0)
            else:
                try:
                    row.append(float(v))
                except:
                    row.append(0.0)
        X_list.append(row)

    return np.array(X_list), np.array(y_list)


def load_glass_contextual():
    """Glass: Context = RI (refractive index), Behavior = element composition."""
    X, y = parse_arff(DATA_DIR / 'conquest' / 'Glass.arff')

    # RI is first feature (context), rest are element percentages (behavior)
    context = StandardScaler().fit_transform(X[:, :1])
    behavior = StandardScaler().fit_transform(X[:, 1:])

    return context, behavior, y


def load_heartdisease_contextual():
    """HeartDisease: Context = age, sex, cp. Behavior = clinical measurements."""
    X, y = parse_arff(DATA_DIR / 'conquest' / 'HeartDisease.arff')

    # First 3-4 features are demographics
    n_context = 3
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


def load_wdbc_contextual():
    """WDBC: Context = first 5 features, Behavior = remaining."""
    X, y = parse_arff(DATA_DIR / 'conquest' / 'WDBC.arff')

    n_context = 5
    context = StandardScaler().fit_transform(X[:, :n_context])
    behavior = StandardScaler().fit_transform(X[:, n_context:])

    return context, behavior, y


# =============================================================================
# FRAUD DATASETS
# =============================================================================

def load_saml_contextual(max_accounts=30000):
    """SAML-D: Context = geography/payment type, Behavior = transaction patterns."""
    print("Loading SAML-D...")
    df = pd.read_csv(DATA_DIR / 'SAML-D.csv')

    # Aggregate by sender
    agg = df.groupby('Sender_account').agg({
        'Sender_bank_location': lambda x: x.mode().iloc[0] if len(x) > 0 else 'UK',
        'Payment_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Cash',
        'Payment_currency': lambda x: x.mode().iloc[0] if len(x) > 0 else 'GBP',
        'Amount': ['mean', 'std', 'min', 'max', 'count'],
        'Is_laundering': 'max',
    })
    agg.columns = ['location', 'payment_type', 'currency',
                   'amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count', 'is_suspicious']
    agg = agg.reset_index()
    agg['amt_std'] = agg['amt_std'].fillna(0)

    # Cross-border ratio
    cb = df.groupby('Sender_account').apply(
        lambda x: (x['Sender_bank_location'] != x['Receiver_bank_location']).mean(),
        include_groups=False
    ).reset_index()
    cb.columns = ['Sender_account', 'cross_border_ratio']
    agg = agg.merge(cb, on='Sender_account')

    # Subsample
    if max_accounts and len(agg) > max_accounts:
        np.random.seed(42)
        susp = agg[agg['is_suspicious'] == 1]
        norm = agg[agg['is_suspicious'] == 0]
        n_susp = min(len(susp), int(max_accounts * len(susp) / len(agg)))
        agg = pd.concat([
            susp.sample(n=n_susp, random_state=42),
            norm.sample(n=max_accounts - n_susp, random_state=42)
        ])

    # Context: one-hot categoricals
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    context = enc.fit_transform(agg[['location', 'payment_type', 'currency']])

    # Behavior: transaction stats
    behavior = agg[['amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count', 'cross_border_ratio']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = agg['is_suspicious'].values
    print(f"  {len(labels)} accounts, {labels.sum()} suspicious ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def load_paysim_contextual(max_samples=30000):
    """PaySim: Context = transaction type, Behavior = amounts/balances."""
    print("Loading PaySim...")
    df = pd.read_csv(DATA_DIR / 'Synthetic Financial Datasets For Fraud Detection.csv')

    if max_samples and len(df) > max_samples:
        np.random.seed(42)
        fraud = df[df['isFraud'] == 1]
        normal = df[df['isFraud'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.03))
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=42),
            normal.sample(n=max_samples - n_fraud, random_state=42)
        ])

    # Context: transaction type (one-hot)
    enc = OneHotEncoder(sparse_output=False)
    context = enc.fit_transform(df[['type']])

    # Behavior: amounts and balances
    behavior = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = df['isFraud'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def load_creditcard_contextual(max_samples=30000):
    """CreditCard: Context = Time, Amount. Behavior = V1-V28."""
    print("Loading CreditCard...")
    df = pd.read_csv(DATA_DIR / 'Credit Card Fraud Detection.csv')

    if max_samples and len(df) > max_samples:
        np.random.seed(42)
        fraud = df[df['Class'] == 1]
        normal = df[df['Class'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.02))
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=42),
            normal.sample(n=max_samples - n_fraud, random_state=42)
        ])

    # Context: Time and Amount
    context = df[['Time', 'Amount']].values
    context = StandardScaler().fit_transform(context)

    # Behavior: V1-V28
    v_cols = [f'V{i}' for i in range(1, 29)]
    behavior = df[v_cols].values

    labels = df['Class'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def load_ieee_cis_contextual(max_samples=30000):
    """IEEE-CIS: Context = card/addr info, Behavior = transaction features."""
    print("Loading IEEE-CIS...")
    trans = pd.read_csv(DATA_DIR / 'IEEE-CIS Fraud Detection' / 'train_transaction.csv')

    if max_samples and len(trans) > max_samples:
        np.random.seed(42)
        fraud = trans[trans['isFraud'] == 1]
        normal = trans[trans['isFraud'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.035))
        trans = pd.concat([
            fraud.sample(n=n_fraud, random_state=42),
            normal.sample(n=max_samples - n_fraud, random_state=42)
        ])

    # Context: Card info (categorical)
    context_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain']
    for col in context_cols:
        trans[col] = trans[col].fillna('unknown')

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    context = enc.fit_transform(trans[context_cols])

    # Behavior: Transaction amount and C/D/V features
    behavior_cols = ['TransactionAmt'] + [f'C{i}' for i in range(1, 15)] + [f'D{i}' for i in range(1, 16)]
    behavior_cols = [c for c in behavior_cols if c in trans.columns]
    behavior = trans[behavior_cols].fillna(0).values
    behavior = StandardScaler().fit_transform(behavior)

    labels = trans['isFraud'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("COMPREHENSIVE CONTEXTUAL ANOMALY DETECTION EXPERIMENTS")
    print("="*60)

    all_results = []

    # ODDS datasets
    datasets = [
        ("Cardio", load_cardio_contextual),
        ("Arrhythmia", load_arrhythmia_contextual),
        ("Pima", load_pima_contextual),
        ("Ionosphere", load_ionosphere_contextual),
        ("WBC", load_wbc_contextual),
        ("Vowels", load_vowels_contextual),
    ]

    # DAMI datasets
    datasets += [
        ("Glass", load_glass_contextual),
        ("HeartDisease", load_heartdisease_contextual),
        ("WDBC", load_wdbc_contextual),
    ]

    # Fraud datasets
    datasets += [
        ("SAML-D", load_saml_contextual),
        ("PaySim", load_paysim_contextual),
        ("CreditCard", load_creditcard_contextual),
        ("IEEE-CIS", load_ieee_cis_contextual),
    ]

    for name, loader in datasets:
        try:
            context, behavior, labels = loader()
            results = run_methods(name, context, behavior, labels)
            all_results.extend(results)
        except Exception as e:
            print(f"\n{name} FAILED: {e}")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'all_contextual_experiments.csv', index=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: PNKIF vs IF")
    print("="*60)

    pivot = df.pivot(index='dataset', columns='method', values='auroc_mean')

    if 'PNKIF' in pivot.columns and 'IF' in pivot.columns:
        pivot['PNKIF_wins'] = pivot['PNKIF'] > pivot['IF']
        pivot['delta'] = pivot['PNKIF'] - pivot['IF']

        print("\nDatasets where PNKIF > IF:")
        wins = pivot[pivot['PNKIF_wins'] == True].sort_values('delta', ascending=False)
        if len(wins) > 0:
            for idx, row in wins.iterrows():
                print(f"  {idx}: PNKIF={row['PNKIF']:.3f}, IF={row['IF']:.3f}, Δ={row['delta']:+.3f}")
        else:
            print("  None")

        print("\nDatasets where IF > PNKIF:")
        losses = pivot[pivot['PNKIF_wins'] == False].sort_values('delta')
        for idx, row in losses.iterrows():
            print(f"  {idx}: IF={row['IF']:.3f}, PNKIF={row['PNKIF']:.3f}, Δ={row['delta']:+.3f}")

    print(f"\nResults saved to {RESULTS_DIR / 'all_contextual_experiments.csv'}")


if __name__ == "__main__":
    main()
