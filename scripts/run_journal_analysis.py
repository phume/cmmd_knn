"""
Extended analysis for journal version:
1. Precision@K at multiple K values (1%, 5%, 10%)
2. Statistical test for disagreement
3. Computational benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

SEEDS = [42, 123, 456, 789, 1011]
INJECTION_RATES = [0.0, 0.01, 0.03, 0.05, 0.10]
K_PERCENTS = [0.01, 0.05, 0.10]  # 1%, 5%, 10%


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


def compute_extended_metrics(context, behavior, labels, seed=42):
    """Compute extended metrics for journal version."""
    K = min(100, len(labels) // 20)
    K = max(5, K)

    # Fit models
    if_model = IFBaseline(n_estimators=100, random_state=seed)
    pnkif_model = PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=seed))

    # Time the models
    start = time.time()
    if_scores = if_model.fit_predict(context, behavior)
    if_time = time.time() - start

    start = time.time()
    pnkif_scores = pnkif_model.fit_predict(context, behavior)
    pnkif_time = time.time() - start

    n = len(labels)
    results = {
        'if_time': if_time,
        'pnkif_time': pnkif_time,
    }

    # Precision@K for multiple K values
    for k_pct in K_PERCENTS:
        k = int(n * k_pct)
        if k == 0:
            k = 1

        if_top_k = set(np.argsort(if_scores)[-k:])
        pnkif_top_k = set(np.argsort(pnkif_scores)[-k:])

        # Agreement
        overlap = len(if_top_k & pnkif_top_k)
        agreement = overlap / k

        # Precision
        if_tp = sum(labels[list(if_top_k)])
        pnkif_tp = sum(labels[list(pnkif_top_k)])

        results[f'agreement_{int(k_pct*100)}pct'] = agreement
        results[f'if_precision_{int(k_pct*100)}pct'] = if_tp / k
        results[f'pnkif_precision_{int(k_pct*100)}pct'] = pnkif_tp / k

    # Rank correlation
    rank_corr, _ = spearmanr(if_scores, pnkif_scores)
    results['rank_corr'] = rank_corr

    return results


def main():
    print("=" * 70)
    print("JOURNAL VERSION: Extended Analysis")
    print("=" * 70)

    context_orig, behavior_orig, labels_orig = load_paysim()
    print(f"PaySim: N={len(labels_orig)}, fraud={labels_orig.sum()}")

    all_results = []

    for rate in INJECTION_RATES:
        print(f"\n--- Injection Rate: {rate:.0%} ---")

        if rate == 0:
            context, behavior, labels = context_orig.copy(), behavior_orig.copy(), labels_orig.copy()
        else:
            context, behavior, labels, n_inj = inject_context_mismatch(
                context_orig.copy(), behavior_orig.copy(), labels_orig.copy(),
                rate=rate, random_state=42
            )
            print(f"  Injected: {n_inj}")

        # Run across seeds
        metrics_agg = {}
        for seed in SEEDS:
            result = compute_extended_metrics(context, behavior, labels, seed=seed)
            for key, val in result.items():
                if key not in metrics_agg:
                    metrics_agg[key] = []
                metrics_agg[key].append(val)

        # Aggregate
        row = {'rate': rate}
        for key, vals in metrics_agg.items():
            row[f'{key}_mean'] = np.mean(vals)
            row[f'{key}_std'] = np.std(vals)
        all_results.append(row)

        # Print summary
        print(f"  Precision@1%:  IF={row['if_precision_1pct_mean']:.3f}, PNKIF={row['pnkif_precision_1pct_mean']:.3f}")
        print(f"  Precision@5%:  IF={row['if_precision_5pct_mean']:.3f}, PNKIF={row['pnkif_precision_5pct_mean']:.3f}")
        print(f"  Precision@10%: IF={row['if_precision_10pct_mean']:.3f}, PNKIF={row['pnkif_precision_10pct_mean']:.3f}")
        print(f"  Agreement@5%:  {row['agreement_5pct_mean']:.1%}")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'journal_analysis.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'journal_analysis.csv'}")

    # Statistical test for disagreement
    print("\n" + "=" * 70)
    print("STATISTICAL TEST: Disagreement Threshold")
    print("=" * 70)

    orig = all_results[0]
    threshold = orig['agreement_5pct_mean'] - 2 * orig['agreement_5pct_std']
    print(f"Baseline Agreement@5%: {orig['agreement_5pct_mean']:.1%} +/- {orig['agreement_5pct_std']:.1%}")
    print(f"Proposed threshold (mean - 2*std): {threshold:.1%}")
    print("\nDecision Rule:")
    print(f"  If Agreement@5% < {threshold:.1%}, reject H0 (no contextual structure)")
    print(f"  Interpretation: contextual anomalies likely present")

    # Check which injection rates trigger the test
    print("\nTest Results by Injection Rate:")
    for row in all_results:
        agreement = row['agreement_5pct_mean']
        reject = agreement < threshold
        status = "REJECT H0 (context matters)" if reject else "FAIL TO REJECT H0"
        print(f"  {row['rate']:.0%}: Agreement={agreement:.1%} -> {status}")

    # Computational benchmark
    print("\n" + "=" * 70)
    print("COMPUTATIONAL BENCHMARK")
    print("=" * 70)
    avg_if_time = np.mean([r['if_time_mean'] for r in all_results])
    avg_pnkif_time = np.mean([r['pnkif_time_mean'] for r in all_results])
    print(f"IF time (30K samples):    {avg_if_time:.2f}s")
    print(f"PNKIF time (30K samples): {avg_pnkif_time:.2f}s")
    print(f"Overhead: {(avg_pnkif_time/avg_if_time - 1)*100:.0f}%")
    print(f"Throughput: ~{30000/avg_pnkif_time:.0f} transactions/second")

    # Table for paper
    print("\n" + "=" * 70)
    print("TABLE FOR PAPER: Extended Precision@K Analysis")
    print("=" * 70)
    print(f"{'Rate':<8} {'IF P@1%':<10} {'PNKIF P@1%':<12} {'IF P@5%':<10} {'PNKIF P@5%':<12} {'IF P@10%':<10} {'PNKIF P@10%':<12}")
    print("-" * 80)
    for row in all_results:
        print(f"{row['rate']:.0%:<8} {row['if_precision_1pct_mean']:.3f}      {row['pnkif_precision_1pct_mean']:.3f}        {row['if_precision_5pct_mean']:.3f}      {row['pnkif_precision_5pct_mean']:.3f}        {row['if_precision_10pct_mean']:.3f}      {row['pnkif_precision_10pct_mean']:.3f}")


if __name__ == "__main__":
    main()
