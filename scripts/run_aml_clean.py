"""
Clean AML Experiments - No injection, original labels only.
For workshop/lower-tier venue submission.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import time

from models.pnkdif import PNKDIFNoMLP, PNKDIF, PNKDIFConfig
from models.baselines import IFBaseline, IFConcat, ROCOD

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_saml_account_level(max_accounts=50000, random_state=42):
    """Load SAML-D aggregated to account level."""
    print("Loading SAML-D...")
    df = pd.read_csv(DATA_DIR / 'SAML-D.csv')
    print(f"  {len(df)} transactions")

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
        np.random.seed(random_state)
        susp = agg[agg['is_suspicious'] == 1]
        norm = agg[agg['is_suspicious'] == 0]
        n_susp = min(len(susp), int(max_accounts * len(susp) / len(agg)))
        agg = pd.concat([
            susp.sample(n=n_susp, random_state=random_state),
            norm.sample(n=max_accounts - n_susp, random_state=random_state)
        ])

    # Context: one-hot encoded categoricals
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    context = enc.fit_transform(agg[['location', 'payment_type', 'currency']])

    # Behavior
    behavior = agg[['amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count', 'cross_border_ratio']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = agg['is_suspicious'].values
    print(f"  {len(labels)} accounts, {labels.sum()} suspicious ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def load_paysim(max_samples=50000, random_state=42):
    """Load PaySim."""
    print("Loading PaySim...")
    df = pd.read_csv(DATA_DIR / 'Synthetic Financial Datasets For Fraud Detection.csv')
    print(f"  {len(df)} transactions")

    if max_samples and len(df) > max_samples:
        np.random.seed(random_state)
        fraud = df[df['isFraud'] == 1]
        normal = df[df['isFraud'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.03))
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=random_state),
            normal.sample(n=max_samples - n_fraud, random_state=random_state)
        ])

    # Context: transaction type
    enc = OneHotEncoder(sparse_output=False)
    context = enc.fit_transform(df[['type']])

    # Behavior
    behavior = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = df['isFraud'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def load_creditcard(max_samples=50000, random_state=42):
    """Load Credit Card Fraud dataset."""
    print("Loading Credit Card Fraud...")
    df = pd.read_csv(DATA_DIR / 'Credit Card Fraud Detection.csv')
    print(f"  {len(df)} transactions")

    if max_samples and len(df) > max_samples:
        np.random.seed(random_state)
        fraud = df[df['Class'] == 1]
        normal = df[df['Class'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.02))
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=random_state),
            normal.sample(n=max_samples - n_fraud, random_state=random_state)
        ])

    # Context: Time and Amount (the only non-PCA features)
    context = df[['Time', 'Amount']].values
    context = StandardScaler().fit_transform(context)

    # Behavior: V1-V28 (PCA components)
    v_cols = [f'V{i}' for i in range(1, 29)]
    behavior = df[v_cols].values

    labels = df['Class'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def run_experiment(name, context, behavior, labels, seeds=[42, 123, 456, 789, 1011]):
    """Run methods and return results."""
    print(f"\n{'='*60}")
    print(f"{name}: N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}, "
          f"anom={100*labels.mean():.2f}%")
    print('='*60)

    K = min(100, len(labels) // 20)

    methods = {
        'IF': lambda s: IFBaseline(n_estimators=100, random_state=s),
        'IF_concat': lambda s: IFConcat(n_estimators=100, random_state=s),
        'ROCOD': lambda s: ROCOD(n_neighbors=K, random_state=s),
        'PNKIF': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=s)),
    }

    results = []
    for method_name, method_fn in methods.items():
        aurocs, times = [], []
        for seed in seeds:
            try:
                model = method_fn(seed)
                start = time.time()
                scores = model.fit_predict(context, behavior)
                elapsed = time.time() - start
                aurocs.append(roc_auc_score(labels, scores))
                times.append(elapsed)
            except Exception as e:
                print(f"  {method_name} error: {e}")
                aurocs.append(np.nan)

        mean_auroc = np.nanmean(aurocs)
        std_auroc = np.nanstd(aurocs)
        print(f"  {method_name:12s}: AUROC={mean_auroc:.3f}±{std_auroc:.3f}")

        results.append({
            'dataset': name, 'method': method_name,
            'auroc_mean': mean_auroc, 'auroc_std': std_auroc,
            'time': np.nanmean(times)
        })

    return results


def main():
    print("="*60)
    print("CLEAN AML EXPERIMENTS - Original Labels Only")
    print("="*60)

    all_results = []

    # SAML-D
    ctx, beh, lbl = load_saml_account_level(max_accounts=50000)
    all_results.extend(run_experiment("SAML-D", ctx, beh, lbl))

    # PaySim
    ctx, beh, lbl = load_paysim(max_samples=50000)
    all_results.extend(run_experiment("PaySim", ctx, beh, lbl))

    # Credit Card
    ctx, beh, lbl = load_creditcard(max_samples=50000)
    all_results.extend(run_experiment("CreditCard", ctx, beh, lbl))

    # Summary
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'aml_clean.csv', index=False)

    print("\n" + "="*60)
    print("SUMMARY (AUROC)")
    print("="*60)
    pivot = df.pivot(index='method', columns='dataset', values='auroc_mean').round(3)
    print(pivot.to_string())

    # Best method per dataset
    print("\nBest method per dataset:")
    for col in pivot.columns:
        best = pivot[col].idxmax()
        score = pivot[col].max()
        print(f"  {col}: {best} ({score:.3f})")

    print(f"\nSaved to {RESULTS_DIR / 'aml_clean.csv'}")


if __name__ == "__main__":
    main()
