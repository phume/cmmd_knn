"""
PaySim experiments with more seeds + PNKDIF (deep model).
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

from models.pnkdif import PNKDIFNoMLP, PNKDIF, PNKDIFConfig
from models.baselines import IFBaseline, IFConcat, ROCOD

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

# More seeds for statistical robustness
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
INJECTION_RATES = [0.01, 0.03, 0.05, 0.10]


def run_methods(context, behavior, labels, K=None):
    """Run all methods including PNKDIF (deep)."""
    if K is None:
        K = min(100, len(labels) // 20)
    K = max(5, K)

    methods = {
        'IF': lambda s: IFBaseline(n_estimators=100, random_state=s),
        'IF_concat': lambda s: IFConcat(n_estimators=100, random_state=s),
        'ROCOD': lambda s: ROCOD(n_neighbors=K, random_state=s),
        'PNKIF': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=s)),
        'PNKDIF': lambda s: PNKDIF(PNKDIFConfig(n_neighbors=K, n_trees=100, n_projections=3, hidden_dim=32, random_state=s)),
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
                print(f"  {method_name} seed {seed} error: {e}")
                aurocs.append(np.nan)

        mean_auroc = np.nanmean(aurocs)
        std_auroc = np.nanstd(aurocs)
        results[method_name] = {'mean': mean_auroc, 'std': std_auroc}

    return results


def inject_context_mismatch(context, behavior, labels, rate, random_state=42):
    """Context mismatch injection."""
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
    context_patterns = [tuple(row) for row in context]

    injected = 0
    for target in targets:
        target_pattern = context_patterns[target]
        different_context = [i for i in normal_idx if context_patterns[i] != target_pattern]
        if len(different_context) > 0:
            donor = np.random.choice(different_context)
            behavior[target] = behavior[donor]
            labels[target] = 1
            injected += 1

    return context, behavior, labels, injected


def inject_geographic_swap(context, behavior, labels, rate, random_state=42):
    """Geographic swap injection."""
    np.random.seed(random_state)
    context = context.copy()
    behavior = behavior.copy()
    labels = labels.copy()

    n_samples = len(labels)
    n_inject = int(n_samples * rate)

    context_patterns = [tuple(row) for row in context]
    unique_patterns = list(set(context_patterns))

    if len(unique_patterns) < 2:
        return context, behavior, labels, 0

    pattern_indices = {}
    for i, p in enumerate(context_patterns):
        if p not in pattern_indices:
            pattern_indices[p] = []
        pattern_indices[p].append(i)

    sorted_patterns = sorted(pattern_indices.items(), key=lambda x: -len(x[1]))
    pattern_a, indices_a = sorted_patterns[0]
    pattern_b, indices_b = sorted_patterns[1]

    normal_a = [i for i in indices_a if labels[i] == 0]
    normal_b = [i for i in indices_b if labels[i] == 0]

    n_each = min(n_inject // 2, len(normal_a), len(normal_b))
    if n_each == 0:
        return context, behavior, labels, 0

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


def load_paysim(max_samples=30000):
    """Load PaySim."""
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

    enc = OneHotEncoder(sparse_output=False)
    context = enc.fit_transform(df[['type']])

    behavior = df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].values
    behavior = StandardScaler().fit_transform(behavior)

    labels = df['isFraud'].values
    print(f"  {len(labels)} samples, {labels.sum()} fraud ({100*labels.mean():.2f}%)")

    return context, behavior, labels


def main():
    print("="*70)
    print("PAYSIM EXPERIMENTS WITH 10 SEEDS + PNKDIF (DEEP)")
    print("="*70)

    context_orig, behavior_orig, labels_orig = load_paysim()
    print(f"N={len(labels_orig)}, d_c={context_orig.shape[1]}, d_y={behavior_orig.shape[1]}")

    all_results = []

    # Original labels
    print("\n--- Original Labels ---")
    results = run_methods(context_orig, behavior_orig, labels_orig)
    for method, stats in results.items():
        print(f"  {method:12s}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
        all_results.append({
            'injection': 'none', 'rate': 0.0, 'method': method,
            'auroc_mean': stats['mean'], 'auroc_std': stats['std']
        })

    # Injection experiments
    injections = {
        'context_mismatch': inject_context_mismatch,
        'geographic_swap': inject_geographic_swap,
    }

    for inj_name, inj_fn in injections.items():
        print(f"\n--- {inj_name} ---")

        for rate in INJECTION_RATES:
            context_inj, behavior_inj, labels_inj, n_inj = inj_fn(
                context_orig.copy(), behavior_orig.copy(), labels_orig.copy(),
                rate=rate, random_state=42
            )

            if n_inj == 0:
                print(f"  Rate {rate:.0%}: No injection possible")
                continue

            results = run_methods(context_inj, behavior_inj, labels_inj)

            print(f"\n  Rate {rate:.0%} (n={n_inj}):")
            for method, stats in sorted(results.items(), key=lambda x: -x[1]['mean']):
                marker = "*" if method in ['PNKIF', 'PNKDIF'] and stats['mean'] > results['IF']['mean'] else ""
                print(f"    {method:12s}: {stats['mean']:.3f} +/- {stats['std']:.3f} {marker}")
                all_results.append({
                    'injection': inj_name, 'rate': rate, 'method': method,
                    'auroc_mean': stats['mean'], 'auroc_std': stats['std']
                })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'paysim_deep_experiments.csv', index=False)

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: PNKIF vs PNKDIF vs IF")
    print("="*70)

    df_inj = df[df['injection'] != 'none']
    for inj in df_inj['injection'].unique():
        print(f"\n{inj}:")
        for rate in INJECTION_RATES:
            subset = df_inj[(df_inj['injection'] == inj) & (df_inj['rate'] == rate)]
            if len(subset) == 0:
                continue

            if_score = subset[subset['method'] == 'IF']['auroc_mean'].values[0]
            pnkif_score = subset[subset['method'] == 'PNKIF']['auroc_mean'].values[0]
            pnkdif_score = subset[subset['method'] == 'PNKDIF']['auroc_mean'].values[0]

            best = 'PNKDIF' if pnkdif_score >= max(pnkif_score, if_score) else ('PNKIF' if pnkif_score > if_score else 'IF')
            print(f"  {rate:.0%}: IF={if_score:.3f}, PNKIF={pnkif_score:.3f}, PNKDIF={pnkdif_score:.3f} -> {best}")

    print(f"\nResults saved to {RESULTS_DIR / 'paysim_deep_experiments.csv'}")


if __name__ == "__main__":
    main()
