"""
Run PNKIF, PNKIF_SNN, ROCOD, IF on ConQuest benchmark datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.conquest_datasets import CONQUEST_DATASETS, load_conquest_dataset
from models.pnkdif import PNKDIF, PNKIF_SNN, PNKDIFConfig, PNKDIFNoMLP
from models.baselines import IFBaseline, ROCOD

SEEDS = [42, 123, 456, 789, 1011]
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_experiment(dataset_name: str, seed: int) -> dict:
    """Run all methods on a dataset with a given seed."""
    np.random.seed(seed)

    try:
        context, behavior, y = load_conquest_dataset(dataset_name)
    except Exception as e:
        print(f"  Error loading {dataset_name}: {e}")
        return None

    # Skip if too few samples or features
    if len(y) < 50 or behavior.shape[1] < 1:
        print(f"  Skipping {dataset_name}: N={len(y)}, d_y={behavior.shape[1]}")
        return None

    # Adjust K based on dataset size
    K = min(50, len(y) // 10)
    if K < 5:
        K = 5

    results = {'dataset': dataset_name, 'seed': seed, 'N': len(y),
               'd_c': context.shape[1], 'd_y': behavior.shape[1],
               'anom_pct': 100 * y.sum() / len(y)}

    # PNKIF (PNKDIF without MLP - direct IF on z-scores)
    try:
        config = PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=seed)
        pnkif = PNKDIFNoMLP(config)
        scores = pnkif.fit_predict(context, behavior)
        results['PNKIF'] = roc_auc_score(y, scores)
    except Exception as e:
        results['PNKIF'] = np.nan
        print(f"    PNKIF error: {e}")

    # PNKIF_SNN
    try:
        config = PNKDIFConfig(n_neighbors=K, n_trees=100, random_state=seed)
        pnkif_snn = PNKIF_SNN(config)
        scores = pnkif_snn.fit_predict(context, behavior)
        results['PNKIF_SNN'] = roc_auc_score(y, scores)
    except Exception as e:
        results['PNKIF_SNN'] = np.nan
        print(f"    PNKIF_SNN error: {e}")

    # ROCOD
    try:
        rocod = ROCOD(n_neighbors=K, random_state=seed)
        scores = rocod.fit_predict(context, behavior)
        results['ROCOD'] = roc_auc_score(y, scores)
    except Exception as e:
        results['ROCOD'] = np.nan
        print(f"    ROCOD error: {e}")

    # IF (on behavior only)
    try:
        iso = IFBaseline(n_estimators=100, random_state=seed)
        scores = iso.fit_predict(context, behavior)
        results['IF'] = roc_auc_score(y, scores)
    except Exception as e:
        results['IF'] = np.nan
        print(f"    IF error: {e}")

    return results


def main():
    all_results = []

    datasets = list(CONQUEST_DATASETS.keys())
    print(f"Running on {len(datasets)} datasets with {len(SEEDS)} seeds each...")

    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        for seed in SEEDS:
            print(f"  seed={seed}...", end=" ", flush=True)
            result = run_experiment(dataset_name, seed)
            if result:
                all_results.append(result)
                print(f"PNKIF={result.get('PNKIF', np.nan):.3f}, "
                      f"PNKIF_SNN={result.get('PNKIF_SNN', np.nan):.3f}, "
                      f"ROCOD={result.get('ROCOD', np.nan):.3f}, "
                      f"IF={result.get('IF', np.nan):.3f}")
            else:
                print("skipped")

    # Save raw results
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'conquest_raw.csv', index=False)
    print(f"\nRaw results saved to {RESULTS_DIR / 'conquest_raw.csv'}")

    # Aggregate by dataset
    if len(df) > 0:
        print("\n" + "="*80)
        print("AGGREGATED RESULTS (mean AUROC)")
        print("="*80)

        summary = df.groupby('dataset')[['PNKIF', 'PNKIF_SNN', 'ROCOD', 'IF']].mean().round(3)
        summary['Best'] = summary[['PNKIF', 'PNKIF_SNN', 'ROCOD', 'IF']].idxmax(axis=1)
        print(summary.to_string())

        summary.to_csv(RESULTS_DIR / 'conquest_summary.csv')
        print(f"\nSummary saved to {RESULTS_DIR / 'conquest_summary.csv'}")


if __name__ == "__main__":
    main()
