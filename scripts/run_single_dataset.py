"""
Run experiments for a single dataset.
Usage: python run_single_dataset.py <dataset_name> [max_samples]
"""

import sys
import os
import json
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from data.fraud_datasets import get_fraud_dataset, FRAUD_DATASETS
from data.synthetic import get_dataset, SYNTHETIC_DATASETS
from models.baselines import get_method
from models.pnkdif import (PNKDIF, PNKDIFConfig, PNKDIFUniform, PNKDIFNoMLP,
                            PNKDIFSingle, PNKDIFGlobal)
from evaluation.metrics import compute_metrics

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
K_FOR_P_AT_K = 100

RESULTS_DIR = ROOT / 'results'
RAW_HEADERS = ['dataset', 'method', 'seed', 'n_samples', 'n_context', 'n_behavior',
               'anomaly_rate', 'auroc', 'auprc', 'p_at_k', 'runtime_sec', 'timestamp']


def get_all_methods(seed, n_samples):
    K = min(100, n_samples // 20)
    config = PNKDIFConfig(n_neighbors=K, random_state=seed)
    return {
        'IF': get_method('IF', random_state=seed),
        'IF_concat': get_method('IF_concat', random_state=seed),
        'DIF': get_method('DIF', random_state=seed),
        'DIF_concat': get_method('DIF_concat', random_state=seed),
        'LOF': get_method('LOF'),
        'QCAD': get_method('QCAD', random_state=seed, n_neighbors=K),
        'ROCOD': get_method('ROCOD', random_state=seed, n_neighbors=K),
        'PNKDIF': PNKDIF(config),
        'PNKDIF_uniform': PNKDIFUniform(config),
        'PNKDIF_noMLP': PNKDIFNoMLP(config),
        'PNKDIF_single': PNKDIFSingle(config),
        'PNKDIF_global': PNKDIFGlobal(config),
    }


def run_dataset(dataset_name, max_samples=None):
    output_csv = RESULTS_DIR / f'phase4_{dataset_name}_raw.csv'
    state_file = RESULTS_DIR / f'phase4_{dataset_name}_state.json'

    # Load state
    completed = set()
    if state_file.exists():
        with open(state_file, 'r') as f:
            completed = set(json.load(f).get('completed', []))

    # Init CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    if not output_csv.exists():
        with open(output_csv, 'w', newline='') as f:
            csv.writer(f).writerow(RAW_HEADERS)

    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {max_samples}")
    print(f"Already completed: {len(completed)}")
    print(f"{'='*60}")

    total_runs = len(SEEDS) * 12
    run_idx = 0

    for seed in SEEDS:
        try:
            # Load dataset
            if dataset_name in FRAUD_DATASETS:
                if max_samples:
                    context, behavior, labels = get_fraud_dataset(
                        dataset_name, random_state=seed, max_samples=max_samples)
                else:
                    context, behavior, labels = get_fraud_dataset(dataset_name, random_state=seed)
            elif dataset_name in SYNTHETIC_DATASETS:
                context, behavior, labels = get_dataset(dataset_name, random_state=seed)
            else:
                print(f"Unknown dataset: {dataset_name}")
                return

            n_samples = len(labels)
            n_context = context.shape[1] if context.ndim > 1 else 1
            n_behavior = behavior.shape[1] if behavior.ndim > 1 else 1
            anomaly_rate = labels.mean()

            print(f"\nSeed {seed}: N={n_samples}, d_c={n_context}, d_y={n_behavior}, anom={anomaly_rate:.3f}")

            methods = get_all_methods(seed, n_samples)

            for method_name, model in methods.items():
                run_idx += 1
                key = f"{dataset_name}|{method_name}|{seed}"

                if key in completed:
                    continue

                try:
                    start = time.time()
                    scores = model.fit_predict(context, behavior)
                    runtime = time.time() - start

                    metrics = compute_metrics(labels, scores, k=K_FOR_P_AT_K)

                    row = [
                        dataset_name, method_name, seed, n_samples, n_context, n_behavior,
                        f"{anomaly_rate:.4f}",
                        f"{metrics['auroc']:.6f}",
                        f"{metrics['auprc']:.6f}",
                        f"{metrics['p_at_k']:.6f}",
                        f"{runtime:.3f}",
                        datetime.now().isoformat()
                    ]

                    with open(output_csv, 'a', newline='') as f:
                        csv.writer(f).writerow(row)

                    completed.add(key)
                    with open(state_file, 'w') as f:
                        json.dump({'completed': list(completed)}, f)

                    print(f"  [{run_idx}/{total_runs}] {method_name}: AUROC={metrics['auroc']:.4f}, Time={runtime:.1f}s")

                except Exception as e:
                    print(f"  [{run_idx}/{total_runs}] {method_name}: ERROR - {e}")
                    completed.add(key)
                    with open(state_file, 'w') as f:
                        json.dump({'completed': list(completed)}, f)

        except Exception as e:
            print(f"Failed to load dataset for seed {seed}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE: {dataset_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_single_dataset.py <dataset_name> [max_samples]")
        print("Datasets: ieee_cis, paysim, creditcard, thyroid")
        sys.exit(1)

    dataset = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None

    run_dataset(dataset, max_samples)
