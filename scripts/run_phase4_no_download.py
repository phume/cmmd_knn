"""
Phase 4 - Datasets that don't require manual download:
- Syn-HighDimContext
- Syn-Cluster
- Thyroid (auto-downloads from ODDS)

10 seeds, 12 methods.
"""

import sys
import os
import json
import csv
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from data.synthetic import get_dataset
from data.fraud_datasets import load_thyroid
from models.baselines import get_method
from models.pnkdif import (PNKDIF, PNKDIFConfig, PNKDIFUniform, PNKDIFNoMLP,
                            PNKDIFSingle, PNKDIFGlobal)
from evaluation.metrics import compute_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
K_FOR_P_AT_K = 100

DATASETS = ['syn_highdim_context', 'syn_cluster', 'thyroid']

# Paths
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
RAW_CSV = RESULTS_DIR / 'phase4_nodownload_raw.csv'
SUMMARY_CSV = RESULTS_DIR / 'phase4_nodownload_summary.csv'
ERRORS_CSV = RESULTS_DIR / 'phase4_nodownload_errors.csv'
STATE_FILE = RESULTS_DIR / 'phase4_nodownload_state.json'
LOG_FILE = LOGS_DIR / 'phase4_nodownload.log'

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'completed': []}

def save_state(state):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def run_key(*args):
    return '|'.join(str(a) for a in args)

# =============================================================================
# CSV
# =============================================================================

RAW_HEADERS = ['dataset', 'method', 'seed', 'n_samples', 'n_context', 'n_behavior',
               'anomaly_rate', 'auroc', 'auprc', 'p_at_k', 'runtime_sec', 'timestamp']
ERROR_HEADERS = ['dataset', 'method', 'seed', 'error', 'traceback', 'timestamp']

def init_csv(filepath, headers):
    if not filepath.exists():
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_csv(filepath, row):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# =============================================================================
# METHODS
# =============================================================================

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

# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_dataset(dataset_name, seed):
    if dataset_name == 'thyroid':
        return load_thyroid(random_state=seed)
    else:
        return get_dataset(dataset_name, random_state=seed)

# =============================================================================
# MAIN
# =============================================================================

def run_experiments():
    logger.info("=" * 60)
    logger.info("PHASE 4 (No Download Required)")
    logger.info("=" * 60)
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Seeds: {SEEDS}")

    state = load_state()
    completed = set(state['completed'])

    init_csv(RAW_CSV, RAW_HEADERS)
    init_csv(ERRORS_CSV, ERROR_HEADERS)

    n_methods = 12
    total_runs = len(DATASETS) * len(SEEDS) * n_methods
    run_idx = 0

    for dataset_name in DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info("=" * 60)

        for seed in SEEDS:
            try:
                context, behavior, labels = load_dataset(dataset_name, seed)
                n_samples = len(labels)
                n_context = context.shape[1] if context.ndim > 1 else 1
                n_behavior = behavior.shape[1] if behavior.ndim > 1 else 1
                anomaly_rate = labels.mean()

                logger.info(f"  Loaded: N={n_samples}, d_c={n_context}, d_y={n_behavior}, "
                           f"anomaly_rate={anomaly_rate:.3f}")

                methods = get_all_methods(seed, n_samples)

                for method_name, model in methods.items():
                    run_idx += 1
                    key = run_key(dataset_name, method_name, seed)

                    if key in completed:
                        continue

                    logger.info(f"  [{run_idx}/{total_runs}] {method_name} | seed={seed}")

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
                        append_csv(RAW_CSV, row)

                        completed.add(key)
                        state['completed'] = list(completed)
                        save_state(state)

                        logger.info(f"    -> AUROC={metrics['auroc']:.4f}, Time={runtime:.2f}s")

                    except Exception as e:
                        append_csv(ERRORS_CSV, [dataset_name, method_name, seed, str(e),
                                                traceback.format_exc(), datetime.now().isoformat()])
                        logger.error(f"    -> ERROR: {e}")
                        completed.add(key)
                        state['completed'] = list(completed)
                        save_state(state)

            except Exception as e:
                logger.error(f"  Failed to load dataset: {e}")
                logger.error(traceback.format_exc())

    logger.info("\n" + "=" * 60)
    logger.info("Generating summary...")
    generate_summary()
    logger.info("Complete!")


def generate_summary():
    import pandas as pd

    if not RAW_CSV.exists():
        logger.warning("No raw results found.")
        return

    df = pd.read_csv(RAW_CSV)

    summary_rows = []
    for dataset in df['dataset'].unique():
        ds_df = df[df['dataset'] == dataset]
        for method in ds_df['method'].unique():
            m_df = ds_df[ds_df['method'] == method]
            summary_rows.append({
                'dataset': dataset,
                'method': method,
                'n_runs': len(m_df),
                'auroc_mean': m_df['auroc'].mean(),
                'auroc_std': m_df['auroc'].std(),
                'auprc_mean': m_df['auprc'].mean(),
                'auprc_std': m_df['auprc'].std(),
                'p_at_k_mean': m_df['p_at_k'].mean(),
                'p_at_k_std': m_df['p_at_k'].std(),
                'runtime_mean': m_df['runtime_sec'].mean(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    logger.info(f"Summary saved to {SUMMARY_CSV}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset in df['dataset'].unique():
        print(f"\n{dataset.upper()}")
        print("-" * 60)
        ds_summary = summary_df[summary_df['dataset'] == dataset].sort_values('auroc_mean', ascending=False)
        for _, row in ds_summary.head(5).iterrows():
            print(f"  {row['method']:18s} | AUROC={row['auroc_mean']:.4f}±{row['auroc_std']:.4f}")


if __name__ == '__main__':
    try:
        run_experiments()
    except KeyboardInterrupt:
        logger.info("Interrupted. State saved.")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
