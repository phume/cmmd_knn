"""
AML Experiments for Option B pivot.

Tests PNKIF vs IF vs ROCOD on real AML datasets with proper context/behavior splits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline, ROCOD

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_saml_account_level(max_accounts=50000, random_state=42):
    """
    Load SAML-D and aggregate to account level.

    Context features: Geographic profile, dominant payment type
    Behavioral features: Transaction patterns, amounts, frequencies
    """
    print("Loading SAML-D...")
    df = pd.read_csv(DATA_DIR / 'SAML-D.csv')
    print(f"  Loaded {len(df)} transactions")

    # Aggregate by sender account
    print("  Aggregating to account level...")

    # Context features (what TYPE of account is this?)
    context_agg = df.groupby('Sender_account').agg({
        'Sender_bank_location': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
        'Payment_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
        'Payment_currency': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    }).reset_index()

    # Behavioral features (what does this account DO?)
    behavior_agg = df.groupby('Sender_account').agg({
        'Amount': ['mean', 'std', 'min', 'max', 'sum', 'count'],
        'Is_laundering': 'max',  # Account is suspicious if ANY transaction is
    })
    behavior_agg.columns = ['_'.join(col).strip() for col in behavior_agg.columns]
    behavior_agg = behavior_agg.reset_index()
    behavior_agg.rename(columns={'Is_laundering_max': 'is_suspicious'}, inplace=True)

    # Cross-border ratio (behavioral)
    cross_border = df.groupby('Sender_account').apply(
        lambda x: (x['Sender_bank_location'] != x['Receiver_bank_location']).mean()
    ).reset_index()
    cross_border.columns = ['Sender_account', 'cross_border_ratio']

    # Currency diversity (behavioral)
    currency_div = df.groupby('Sender_account').apply(
        lambda x: x['Received_currency'].nunique()
    ).reset_index()
    currency_div.columns = ['Sender_account', 'currency_diversity']

    # Payment type diversity (behavioral)
    payment_div = df.groupby('Sender_account').apply(
        lambda x: x['Payment_type'].nunique()
    ).reset_index()
    payment_div.columns = ['Sender_account', 'payment_type_diversity']

    # Merge all
    accounts = context_agg.merge(behavior_agg, on='Sender_account')
    accounts = accounts.merge(cross_border, on='Sender_account')
    accounts = accounts.merge(currency_div, on='Sender_account')
    accounts = accounts.merge(payment_div, on='Sender_account')

    # Fill NaN
    accounts['Amount_std'] = accounts['Amount_std'].fillna(0)

    # Subsample if needed
    if max_accounts and len(accounts) > max_accounts:
        np.random.seed(random_state)
        # Stratified sampling to preserve suspicious ratio
        suspicious = accounts[accounts['is_suspicious'] == 1]
        normal = accounts[accounts['is_suspicious'] == 0]
        n_suspicious = min(len(suspicious), int(max_accounts * len(suspicious) / len(accounts)))
        n_normal = max_accounts - n_suspicious

        accounts = pd.concat([
            suspicious.sample(n=n_suspicious, random_state=random_state),
            normal.sample(n=n_normal, random_state=random_state)
        ])

    print(f"  {len(accounts)} accounts, {accounts['is_suspicious'].sum()} suspicious ({100*accounts['is_suspicious'].mean():.2f}%)")

    # One-hot encode categorical context features (better for K-NN distance)
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    context_cats = accounts[['Sender_bank_location', 'Payment_type', 'Payment_currency']].values
    context = encoder.fit_transform(context_cats)

    # Behavioral features
    behavior_cols = ['Amount_mean', 'Amount_std', 'Amount_min', 'Amount_max',
                     'Amount_sum', 'Amount_count', 'cross_border_ratio',
                     'currency_diversity', 'payment_type_diversity']
    behavior = accounts[behavior_cols].values

    # Normalize behavior
    scaler = StandardScaler()
    behavior = scaler.fit_transform(behavior)

    labels = accounts['is_suspicious'].values

    return context, behavior, labels, accounts


def inject_contextual_anomalies(context, behavior, labels, injection_rate=0.03, random_state=42):
    """
    Inject contextual anomalies: take accounts from one context group and give them
    behavior typical of another context group.

    This creates anomalies that are:
    - Normal GLOBALLY (the behavior exists in the dataset)
    - Unusual FOR CONTEXT (unusual for THIS type of account)
    """
    np.random.seed(random_state)

    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * injection_rate)

    # Group accounts by context (location)
    unique_contexts = np.unique(context[:, 0])
    print(f"  Found {len(unique_contexts)} unique location contexts")

    if len(unique_contexts) < 2:
        print("  Warning: Need at least 2 context groups for injection")
        return context, behavior, labels

    # Find the two largest context groups
    context_counts = [(c, np.sum(context[:, 0] == c)) for c in unique_contexts]
    context_counts.sort(key=lambda x: -x[1])
    ctx_a, count_a = context_counts[0]
    ctx_b, count_b = context_counts[1]

    print(f"  Group A (ctx={ctx_a}): {count_a} accounts")
    print(f"  Group B (ctx={ctx_b}): {count_b} accounts")

    # Get normal accounts from each group
    group_a_idx = np.where((context[:, 0] == ctx_a) & (labels == 0))[0]
    group_b_idx = np.where((context[:, 0] == ctx_b) & (labels == 0))[0]

    if len(group_a_idx) < n_inject // 2 or len(group_b_idx) < n_inject // 2:
        print(f"  Warning: Not enough normal samples in groups")
        # Reduce injection to what's possible
        n_inject = min(len(group_a_idx), len(group_b_idx)) * 2

    n_each = n_inject // 2

    # Swap behaviors: A gets B's behavior, B gets A's behavior
    targets_a = np.random.choice(group_a_idx, size=n_each, replace=False)
    targets_b = np.random.choice(group_b_idx, size=n_each, replace=False)
    donors_a = np.random.choice(group_a_idx, size=n_each, replace=True)
    donors_b = np.random.choice(group_b_idx, size=n_each, replace=True)

    # Group A targets get Group B behavior
    for target, donor in zip(targets_a, donors_b):
        behavior[target] = behavior[donor]
        labels[target] = 1

    # Group B targets get Group A behavior
    for target, donor in zip(targets_b, donors_a):
        behavior[target] = behavior[donor]
        labels[target] = 1

    print(f"  Injected {n_inject} contextual anomalies (cross-context behavior swap)")
    print(f"  New suspicious rate: {100*labels.mean():.2f}%")

    return context, behavior, labels


def run_experiment(name, context, behavior, labels, seeds=[42, 123, 456]):
    """Run all methods and return results."""

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"N={len(labels)}, d_c={context.shape[1]}, d_y={behavior.shape[1]}")
    print(f"Suspicious: {labels.sum()} ({100*labels.mean():.2f}%)")
    print('='*60)

    K = min(100, len(labels) // 20)

    methods = {
        'IF': lambda s: IFBaseline(n_estimators=100, random_state=s),
        'ROCOD': lambda s: ROCOD(n_neighbors=K, random_state=s),
        'PNKIF': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=s)),
    }

    results = []

    for method_name, method_fn in methods.items():
        aurocs = []
        auprcs = []
        times = []

        for seed in seeds:
            try:
                model = method_fn(seed)
                start = time.time()
                scores = model.fit_predict(context, behavior)
                elapsed = time.time() - start

                auroc = roc_auc_score(labels, scores)
                precision, recall, _ = precision_recall_curve(labels, scores)
                auprc = auc(recall, precision)

                aurocs.append(auroc)
                auprcs.append(auprc)
                times.append(elapsed)
            except Exception as e:
                print(f"  {method_name} error: {e}")
                aurocs.append(np.nan)
                auprcs.append(np.nan)
                times.append(np.nan)

        mean_auroc = np.nanmean(aurocs)
        std_auroc = np.nanstd(aurocs)
        mean_auprc = np.nanmean(auprcs)
        mean_time = np.nanmean(times)

        print(f"  {method_name:10s}: AUROC={mean_auroc:.3f}±{std_auroc:.3f}, AUPRC={mean_auprc:.3f}, time={mean_time:.1f}s")

        results.append({
            'dataset': name,
            'method': method_name,
            'auroc_mean': mean_auroc,
            'auroc_std': std_auroc,
            'auprc_mean': mean_auprc,
            'time_mean': mean_time,
        })

    return results


def load_paysim(max_samples=50000, random_state=42):
    """
    Load PaySim synthetic financial dataset.

    Context: Transaction type
    Behavior: Amount, balance changes
    """
    print("Loading PaySim...")
    filepath = DATA_DIR / 'Synthetic Financial Datasets For Fraud Detection.csv'

    if not filepath.exists():
        print(f"  PaySim not found at {filepath}")
        return None, None, None

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} transactions")

    # Subsample
    if max_samples and len(df) > max_samples:
        np.random.seed(random_state)
        # Stratified
        fraud = df[df['isFraud'] == 1]
        normal = df[df['isFraud'] == 0]
        n_fraud = min(len(fraud), int(max_samples * 0.05))  # Cap at 5%
        n_normal = max_samples - n_fraud
        df = pd.concat([
            fraud.sample(n=n_fraud, random_state=random_state),
            normal.sample(n=n_normal, random_state=random_state)
        ])

    # Context: Transaction type (one-hot)
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    context = encoder.fit_transform(df[['type']].values)

    # Behavior: Amount and balance features
    behavior_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    behavior = df[behavior_cols].values

    # Normalize
    scaler = StandardScaler()
    behavior = scaler.fit_transform(behavior)

    labels = df['isFraud'].values

    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")
    return context, behavior, labels


def main():
    print("="*60)
    print("AML EXPERIMENTS - Option B Pivot")
    print("="*60)

    all_results = []

    # 1. SAML-D Original Labels
    print("\n[1] Loading SAML-D with original labels...")
    context, behavior, labels, _ = load_saml_account_level(max_accounts=30000)
    results = run_experiment("SAML-D (original)", context, behavior, labels)
    all_results.extend(results)

    # 2. SAML-D with Injected Contextual Anomalies (higher rate)
    print("\n[2] SAML-D with injected contextual anomalies...")
    context_inj, behavior_inj, labels_inj = inject_contextual_anomalies(
        context.copy(), behavior.copy(), labels.copy(), injection_rate=0.05
    )
    results = run_experiment("SAML-D (injected)", context_inj, behavior_inj, labels_inj)
    all_results.extend(results)

    # 3. PaySim
    print("\n[3] Loading PaySim...")
    ctx_pay, beh_pay, lbl_pay = load_paysim(max_samples=50000)
    if ctx_pay is not None:
        results = run_experiment("PaySim (original)", ctx_pay, beh_pay, lbl_pay)
        all_results.extend(results)

        # PaySim with injection
        print("\n[4] PaySim with injected contextual anomalies...")
        ctx_pay_inj, beh_pay_inj, lbl_pay_inj = inject_contextual_anomalies(
            ctx_pay.copy(), beh_pay.copy(), lbl_pay.copy(), injection_rate=0.03
        )
        results = run_experiment("PaySim (injected)", ctx_pay_inj, beh_pay_inj, lbl_pay_inj)
        all_results.extend(results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'aml_experiments.csv', index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    pivot = df.pivot(index='method', columns='dataset', values='auroc_mean').round(3)
    print(pivot.to_string())

    print("\nKey comparison (SAML-D injected):")
    inj = df[df['dataset'] == 'SAML-D (injected)'].set_index('method')
    if_score = inj.loc['IF', 'auroc_mean']
    pnkif_score = inj.loc['PNKIF', 'auroc_mean']
    print(f"  IF:    {if_score:.3f}")
    print(f"  PNKIF: {pnkif_score:.3f}")
    print(f"  Delta: {pnkif_score - if_score:+.3f} ({'PNKIF wins' if pnkif_score > if_score else 'IF wins'})")

    print(f"\nResults saved to {RESULTS_DIR / 'aml_experiments.csv'}")


if __name__ == "__main__":
    main()
