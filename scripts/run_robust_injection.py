"""
Robust Injection Experiments for AML Workshop Paper.

Multiple injection strategies + sensitivity analysis to make results defensible.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline, IFConcat, ROCOD

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

SEEDS = [42, 123, 456, 789, 1011]
INJECTION_RATES = [0.01, 0.03, 0.05, 0.10]


def run_methods(context, behavior, labels, K=None):
    """Run all methods and return AUROC scores."""
    if K is None:
        K = min(100, len(labels) // 20)
    K = max(5, K)

    methods = {
        'IF': lambda s: IFBaseline(n_estimators=100, random_state=s),
        'IF_concat': lambda s: IFConcat(n_estimators=100, random_state=s),
        'ROCOD': lambda s: ROCOD(n_neighbors=K, random_state=s),
        'PNKIF': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=s)),
    }

    results = {}
    for method_name, method_fn in methods.items():
        aurocs = []
        for seed in SEEDS:
            try:
                model = method_fn(seed)
                scores = model.fit_predict(context, behavior)
                auroc = roc_auc_score(labels, scores)
                aurocs.append(auroc)
            except Exception as e:
                aurocs.append(np.nan)
        results[method_name] = np.nanmean(aurocs)

    return results


# =============================================================================
# INJECTION STRATEGIES
# =============================================================================

def inject_geographic_swap(context, behavior, labels, rate, random_state=42):
    """
    Geographic Arbitrage: Swap behaviors between different geographic regions.

    AML Typology: Domestic accounts exhibiting cross-border transaction patterns
    typical of high-risk jurisdictions.
    """
    np.random.seed(random_state)
    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * rate)

    # Find two largest context clusters (by first context dimension)
    # For one-hot encoded context, find most common patterns
    context_patterns = [tuple(row) for row in context]
    unique_patterns = list(set(context_patterns))

    if len(unique_patterns) < 2:
        return context, behavior, labels, 0

    # Get indices for each pattern
    pattern_indices = {}
    for i, p in enumerate(context_patterns):
        if p not in pattern_indices:
            pattern_indices[p] = []
        pattern_indices[p].append(i)

    # Sort by size, get two largest
    sorted_patterns = sorted(pattern_indices.items(), key=lambda x: -len(x[1]))
    pattern_a, indices_a = sorted_patterns[0]
    pattern_b, indices_b = sorted_patterns[1]

    # Filter to normal samples only
    normal_a = [i for i in indices_a if labels[i] == 0]
    normal_b = [i for i in indices_b if labels[i] == 0]

    n_each = min(n_inject // 2, len(normal_a), len(normal_b))
    if n_each == 0:
        return context, behavior, labels, 0

    # Swap behaviors
    targets_a = np.random.choice(normal_a, size=n_each, replace=False)
    targets_b = np.random.choice(normal_b, size=n_each, replace=False)
    donors_a = np.random.choice(normal_a, size=n_each, replace=True)
    donors_b = np.random.choice(normal_b, size=n_each, replace=True)

    for target, donor in zip(targets_a, donors_b):
        behavior[target] = behavior[donor]
        labels[target] = 1

    for target, donor in zip(targets_b, donors_a):
        behavior[target] = behavior[donor]
        labels[target] = 1

    return context, behavior, labels, n_each * 2


def inject_velocity_anomaly(context, behavior, labels, rate, random_state=42):
    """
    Velocity Anomaly: Scale transaction frequency/counts by 2-5x.

    AML Typology: Structuring - normal amounts but unusually high frequency
    to avoid reporting thresholds.
    """
    np.random.seed(random_state)
    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * rate)

    # Find normal samples
    normal_idx = np.where(labels == 0)[0]
    if len(normal_idx) < n_inject:
        n_inject = len(normal_idx)

    targets = np.random.choice(normal_idx, size=n_inject, replace=False)

    # Scale behavior by 2-5x (simulating increased velocity)
    for target in targets:
        scale = np.random.uniform(2.0, 5.0)
        behavior[target] = behavior[target] * scale
        labels[target] = 1

    return context, behavior, labels, n_inject


def inject_temporal_shift(context, behavior, labels, rate, random_state=42):
    """
    Temporal Shift: Add noise to behavior that's unusual for the context.

    AML Typology: Unusual timing patterns - transactions at unexpected times
    for the account type.
    """
    np.random.seed(random_state)
    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * rate)

    normal_idx = np.where(labels == 0)[0]
    if len(normal_idx) < n_inject:
        n_inject = len(normal_idx)

    targets = np.random.choice(normal_idx, size=n_inject, replace=False)

    # Add systematic shift (2-3 std devs in random direction)
    behavior_std = behavior.std(axis=0)
    for target in targets:
        shift = np.random.choice([-1, 1], size=behavior.shape[1]) * behavior_std * np.random.uniform(2, 3)
        behavior[target] = behavior[target] + shift
        labels[target] = 1

    return context, behavior, labels, n_inject


def inject_context_mismatch(context, behavior, labels, rate, random_state=42):
    """
    Context Mismatch: Assign behavior from a random different context.

    AML Typology: Account takeover or misuse - behavior inconsistent with
    account profile.
    """
    np.random.seed(random_state)
    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * rate)

    normal_idx = np.where(labels == 0)[0]
    if len(normal_idx) < n_inject:
        n_inject = len(normal_idx)

    targets = np.random.choice(normal_idx, size=n_inject, replace=False)

    # For each target, find a donor from a DIFFERENT context
    context_patterns = [tuple(row) for row in context]

    for target in targets:
        target_pattern = context_patterns[target]
        # Find samples with different context
        different_context = [i for i in normal_idx if context_patterns[i] != target_pattern]
        if len(different_context) > 0:
            donor = np.random.choice(different_context)
            behavior[target] = behavior[donor]
            labels[target] = 1

    return context, behavior, labels, n_inject


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_saml(max_accounts=30000):
    """Load SAML-D aggregated to account level."""
    df = pd.read_csv(DATA_DIR / 'SAML-D.csv')

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

    cb = df.groupby('Sender_account').apply(
        lambda x: (x['Sender_bank_location'] != x['Receiver_bank_location']).mean(),
        include_groups=False
    ).reset_index()
    cb.columns = ['Sender_account', 'cross_border_ratio']
    agg = agg.merge(cb, on='Sender_account')

    if max_accounts and len(agg) > max_accounts:
        np.random.seed(42)
        susp = agg[agg['is_suspicious'] == 1]
        norm = agg[agg['is_suspicious'] == 0]
        n_susp = min(len(susp), int(max_accounts * len(susp) / len(agg)))
        agg = pd.concat([
            susp.sample(n=n_susp, random_state=42),
            norm.sample(n=max_accounts - n_susp, random_state=42)
        ])

    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    context = enc.fit_transform(agg[['location', 'payment_type', 'currency']])

    behavior = agg[['amt_mean', 'amt_std', 'amt_min', 'amt_max', 'tx_count', 'cross_border_ratio']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = agg['is_suspicious'].values
    return context, behavior, labels


def load_paysim(max_samples=30000):
    """Load PaySim."""
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

    enc = OneHotEncoder(sparse_output=False)
    context = enc.fit_transform(df[['type']])

    behavior = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = df['isFraud'].values
    return context, behavior, labels


def load_creditcard(max_samples=30000):
    """Load Credit Card Fraud."""
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

    context = df[['Time', 'Amount']].values
    context = StandardScaler().fit_transform(context)

    v_cols = [f'V{i}' for i in range(1, 29)]
    behavior = df[v_cols].values

    labels = df['Class'].values
    return context, behavior, labels


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("="*70)
    print("ROBUST INJECTION EXPERIMENTS FOR AML WORKSHOP PAPER")
    print("="*70)

    datasets = {
        'SAML-D': load_saml,
        'PaySim': load_paysim,
        'CreditCard': load_creditcard,
    }

    injection_strategies = {
        'geographic_swap': inject_geographic_swap,
        'velocity_anomaly': inject_velocity_anomaly,
        'temporal_shift': inject_temporal_shift,
        'context_mismatch': inject_context_mismatch,
    }

    all_results = []

    for dataset_name, loader in datasets.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print('='*70)

        # Load original data
        context_orig, behavior_orig, labels_orig = loader()
        print(f"Loaded: N={len(labels_orig)}, d_c={context_orig.shape[1]}, d_y={behavior_orig.shape[1]}")
        print(f"Original anomaly rate: {100*labels_orig.mean():.2f}%")

        # Run on original labels
        print(f"\n--- Original Labels ---")
        orig_results = run_methods(context_orig, behavior_orig, labels_orig)
        for method, auroc in orig_results.items():
            print(f"  {method:12s}: {auroc:.3f}")
            all_results.append({
                'dataset': dataset_name,
                'injection': 'none',
                'rate': 0.0,
                'method': method,
                'auroc': auroc,
            })

        # Run injection experiments
        for inj_name, inj_fn in injection_strategies.items():
            print(f"\n--- Injection: {inj_name} ---")

            for rate in INJECTION_RATES:
                context_inj, behavior_inj, labels_inj, n_injected = inj_fn(
                    context_orig.copy(), behavior_orig.copy(), labels_orig.copy(),
                    rate=rate, random_state=42
                )

                if n_injected == 0:
                    print(f"  Rate {rate:.0%}: No injection possible")
                    continue

                results = run_methods(context_inj, behavior_inj, labels_inj)

                pnkif_wins = results['PNKIF'] > results['IF']
                delta = results['PNKIF'] - results['IF']

                print(f"  Rate {rate:.0%}: IF={results['IF']:.3f}, PNKIF={results['PNKIF']:.3f}, "
                      f"delta={delta:+.3f} {'*' if pnkif_wins else ''}")

                for method, auroc in results.items():
                    all_results.append({
                        'dataset': dataset_name,
                        'injection': inj_name,
                        'rate': rate,
                        'method': method,
                        'auroc': auroc,
                    })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'robust_injection_experiments.csv', index=False)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: When does PNKIF beat IF?")
    print("="*70)

    df_inj = df[df['injection'] != 'none']
    pivot = df_inj.pivot_table(
        index=['dataset', 'injection', 'rate'],
        columns='method',
        values='auroc'
    )

    if 'PNKIF' in pivot.columns and 'IF' in pivot.columns:
        pivot['PNKIF_wins'] = pivot['PNKIF'] > pivot['IF']
        pivot['delta'] = pivot['PNKIF'] - pivot['IF']

        wins = pivot[pivot['PNKIF_wins']].sort_values('delta', ascending=False)
        print(f"\nPNKIF wins in {len(wins)}/{len(pivot)} injection scenarios")

        if len(wins) > 0:
            print("\nTop PNKIF wins:")
            for (ds, inj, rate), row in wins.head(10).iterrows():
                print(f"  {ds} + {inj} @ {rate:.0%}: delta={row['delta']:+.3f}")

    print(f"\nResults saved to {RESULTS_DIR / 'robust_injection_experiments.csv'}")


if __name__ == "__main__":
    main()
