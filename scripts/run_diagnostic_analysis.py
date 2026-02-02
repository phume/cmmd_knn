"""
Diagnostic analysis: quantify IF vs PNKIF disagreement.

Computes:
1. Agreement rate at top-k (what % of top-k flagged by both methods)
2. Rank correlation (Spearman) between IF and PNKIF scores
3. Precision@k for each method
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

from models.pnkdif import PNKDIFNoMLP, PNKDIFConfig
from models.baselines import IFBaseline

DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

SEEDS = [42, 123, 456, 789, 1011]
INJECTION_RATES = [0.0, 0.01, 0.03, 0.05, 0.10]


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


def compute_diagnostics(context, behavior, labels, seed=42, k_percent=0.05):
    """Compute diagnostic metrics between IF and PNKIF."""
    K = min(100, len(labels) // 20)
    K = max(5, K)

    # Fit models
    if_model = IFBaseline(n_estimators=100, random_state=seed)
    pnkif_model = PNKDIFNoMLP(PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=seed))

    if_scores = if_model.fit_predict(context, behavior)
    pnkif_scores = pnkif_model.fit_predict(context, behavior)

    n = len(labels)
    k = int(n * k_percent)

    # Top-k indices
    if_top_k = set(np.argsort(if_scores)[-k:])
    pnkif_top_k = set(np.argsort(pnkif_scores)[-k:])

    # Agreement rate: overlap in top-k
    overlap = len(if_top_k & pnkif_top_k)
    agreement_rate = overlap / k

    # Rank correlation
    rank_corr, _ = spearmanr(if_scores, pnkif_scores)

    # Precision@k for each method
    if_pred = np.zeros(n)
    if_pred[list(if_top_k)] = 1
    pnkif_pred = np.zeros(n)
    pnkif_pred[list(pnkif_top_k)] = 1

    # True positives in top-k
    if_tp = sum(labels[list(if_top_k)])
    pnkif_tp = sum(labels[list(pnkif_top_k)])

    if_precision_k = if_tp / k
    pnkif_precision_k = pnkif_tp / k

    return {
        'agreement_rate': agreement_rate,
        'rank_corr': rank_corr,
        'if_precision_k': if_precision_k,
        'pnkif_precision_k': pnkif_precision_k,
        'overlap': overlap,
        'k': k
    }


def main():
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS: IF vs PNKIF Disagreement")
    print("=" * 70)

    context_orig, behavior_orig, labels_orig = load_paysim()
    print(f"PaySim: N={len(labels_orig)}, fraud={labels_orig.sum()}")

    all_results = []

    for rate in INJECTION_RATES:
        print(f"\n--- Injection Rate: {rate:.0%} ---")

        if rate == 0:
            context, behavior, labels = context_orig.copy(), behavior_orig.copy(), labels_orig.copy()
            n_inj = 0
        else:
            context, behavior, labels, n_inj = inject_context_mismatch(
                context_orig.copy(), behavior_orig.copy(), labels_orig.copy(),
                rate=rate, random_state=42
            )

        print(f"  Injected: {n_inj}, Total anomalies: {labels.sum()}")

        # Run across seeds
        metrics = {'agreement_rate': [], 'rank_corr': [],
                   'if_precision_k': [], 'pnkif_precision_k': []}

        for seed in SEEDS:
            result = compute_diagnostics(context, behavior, labels, seed=seed, k_percent=0.05)
            for key in metrics:
                metrics[key].append(result[key])

        # Aggregate
        row = {
            'rate': rate,
            'agreement_rate_mean': np.mean(metrics['agreement_rate']),
            'agreement_rate_std': np.std(metrics['agreement_rate']),
            'rank_corr_mean': np.mean(metrics['rank_corr']),
            'rank_corr_std': np.std(metrics['rank_corr']),
            'if_precision_k_mean': np.mean(metrics['if_precision_k']),
            'if_precision_k_std': np.std(metrics['if_precision_k']),
            'pnkif_precision_k_mean': np.mean(metrics['pnkif_precision_k']),
            'pnkif_precision_k_std': np.std(metrics['pnkif_precision_k']),
        }
        all_results.append(row)

        print(f"  Agreement@5%: {row['agreement_rate_mean']:.3f} +/- {row['agreement_rate_std']:.3f}")
        print(f"  Rank Corr:    {row['rank_corr_mean']:.3f} +/- {row['rank_corr_std']:.3f}")
        print(f"  IF P@5%:      {row['if_precision_k_mean']:.3f} +/- {row['if_precision_k_std']:.3f}")
        print(f"  PNKIF P@5%:   {row['pnkif_precision_k_mean']:.3f} +/- {row['pnkif_precision_k_std']:.3f}")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'diagnostic_analysis.csv', index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'diagnostic_analysis.csv'}")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("TABLE FOR PAPER: Diagnostic Metrics vs Injection Rate")
    print("=" * 70)
    print(f"{'Rate':<8} {'Agreement':<12} {'Rank Corr':<12} {'IF P@5%':<12} {'PNKIF P@5%':<12}")
    print("-" * 56)
    for row in all_results:
        print(f"{row['rate']:.0%:<8} {row['agreement_rate_mean']:.3f}        {row['rank_corr_mean']:.3f}        {row['if_precision_k_mean']:.3f}        {row['pnkif_precision_k_mean']:.3f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    orig = all_results[0]
    high = all_results[-1]  # 10% injection
    print(f"Agreement drops from {orig['agreement_rate_mean']:.1%} (no injection) to {high['agreement_rate_mean']:.1%} (10% injection)")
    print(f"Rank correlation drops from {orig['rank_corr_mean']:.3f} to {high['rank_corr_mean']:.3f}")
    print(f"PNKIF P@5% improves from {orig['pnkif_precision_k_mean']:.3f} to {high['pnkif_precision_k_mean']:.3f}")
    print(f"IF P@5% drops from {orig['if_precision_k_mean']:.3f} to {high['if_precision_k_mean']:.3f}")


if __name__ == "__main__":
    main()
