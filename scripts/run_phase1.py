"""
Phase 1 Runner: Synthetic Validation

Executes experiments on Syn-Linear, Syn-Scale, Syn-Multimodal across all methods.
Features:
- Incremental CSV writing (results/phase1_raw.csv)
- Checkpoint/resume (results/phase1_state.json)
- Summary statistics (results/phase1_summary.csv)
- Logging (logs/phase1.log)
- Error handling (results/phase1_errors.csv)
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

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from data.synthetic import get_dataset
from models.baselines import get_method
from models.pnkdif import PNKDIF, PNKDIFConfig, PNKDIFUniform, PNKDIFNoMLP
from evaluation.metrics import compute_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = ['syn_linear', 'syn_scale', 'syn_multimodal']
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
N_SAMPLES = 10000
ANOMALY_RATE = 0.05
K_FOR_P_AT_K = 100

# Paths
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
RAW_CSV = RESULTS_DIR / 'phase1_raw.csv'
SUMMARY_CSV = RESULTS_DIR / 'phase1_summary.csv'
ERRORS_CSV = RESULTS_DIR / 'phase1_errors.csv'
STATE_FILE = RESULTS_DIR / 'phase1_state.json'
LOG_FILE = LOGS_DIR / 'phase1.log'

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# METHOD FACTORY
# =============================================================================

def get_all_methods(seed):
    """Return dict of method_name -> model instance."""
    return {
        'IF': get_method('IF', random_state=seed),
        'IF_concat': get_method('IF_concat', random_state=seed),
        'DIF': get_method('DIF', random_state=seed),
        'DIF_concat': get_method('DIF_concat', random_state=seed),
        'LOF': get_method('LOF'),
        'QCAD': get_method('QCAD', random_state=seed),
        'ROCOD': get_method('ROCOD', random_state=seed),
        'PNKDIF': PNKDIF(PNKDIFConfig(n_neighbors=100, random_state=seed)),
        'PNKDIF_uniform': PNKDIFUniform(PNKDIFConfig(n_neighbors=100, random_state=seed)),
        'PNKDIF_noMLP': PNKDIFNoMLP(PNKDIFConfig(n_neighbors=100, random_state=seed)),
    }

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state():
    """Load checkpoint state."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'completed': []}

def save_state(state):
    """Save checkpoint state."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def run_key(dataset, method, seed):
    """Create unique key for a run."""
    return f"{dataset}|{method}|{seed}"

# =============================================================================
# CSV WRITING
# =============================================================================

def init_csv(filepath, headers):
    """Initialize CSV with headers if it doesn't exist."""
    if not filepath.exists():
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_csv(filepath, row):
    """Append a row to CSV."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

RAW_HEADERS = ['dataset', 'method', 'seed', 'auroc', 'auprc', 'p_at_k', 'runtime_sec', 'timestamp']
ERROR_HEADERS = ['dataset', 'method', 'seed', 'error', 'traceback', 'timestamp']

# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_single(dataset, method_name, seed, methods_cache):
    """Run a single experiment."""
    start_time = time.time()

    # Generate data
    context, behavior, labels = get_dataset(
        dataset,
        n_samples=N_SAMPLES,
        anomaly_rate=ANOMALY_RATE,
        random_state=seed
    )

    # Get method
    model = methods_cache[method_name]

    # Run
    scores = model.fit_predict(context, behavior)

    # Compute metrics
    metrics = compute_metrics(labels, scores, k=K_FOR_P_AT_K)

    runtime = time.time() - start_time

    return {
        'auroc': metrics['auroc'],
        'auprc': metrics['auprc'],
        'p_at_k': metrics['p_at_k'],
        'runtime': runtime
    }

def run_phase1():
    """Main Phase 1 runner."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Synthetic Validation")
    logger.info("=" * 60)
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"N_samples: {N_SAMPLES}, Anomaly rate: {ANOMALY_RATE}")

    # Initialize
    state = load_state()
    completed = set(state['completed'])

    init_csv(RAW_CSV, RAW_HEADERS)
    init_csv(ERRORS_CSV, ERROR_HEADERS)

    # Count total runs
    method_names = list(get_all_methods(42).keys())
    total_runs = len(DATASETS) * len(method_names) * len(SEEDS)
    completed_count = len(completed)

    logger.info(f"Total runs: {total_runs}, Already completed: {completed_count}")

    # Run experiments
    run_idx = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            # Create fresh methods for each seed
            methods_cache = get_all_methods(seed)

            for method_name in method_names:
                run_idx += 1
                key = run_key(dataset, method_name, seed)

                if key in completed:
                    continue

                logger.info(f"[{run_idx}/{total_runs}] {dataset} | {method_name} | seed={seed}")

                try:
                    start_ts = datetime.now().isoformat()
                    result = run_single(dataset, method_name, seed, methods_cache)
                    end_ts = datetime.now().isoformat()

                    # Write result
                    row = [
                        dataset,
                        method_name,
                        seed,
                        f"{result['auroc']:.6f}",
                        f"{result['auprc']:.6f}",
                        f"{result['p_at_k']:.6f}",
                        f"{result['runtime']:.3f}",
                        end_ts
                    ]
                    append_csv(RAW_CSV, row)

                    # Update state
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

                    logger.info(f"  -> AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, "
                               f"P@{K_FOR_P_AT_K}={result['p_at_k']:.4f}, Time={result['runtime']:.2f}s")

                except Exception as e:
                    error_ts = datetime.now().isoformat()
                    tb = traceback.format_exc()

                    error_row = [dataset, method_name, seed, str(e), tb, error_ts]
                    append_csv(ERRORS_CSV, error_row)

                    logger.error(f"  -> ERROR: {e}")

                    # Still mark as completed to avoid infinite retry
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

    logger.info("=" * 60)
    logger.info("All runs completed. Generating summary...")
    generate_summary()
    logger.info("Phase 1 complete!")

# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def generate_summary():
    """Generate summary statistics from raw results."""
    import pandas as pd

    if not RAW_CSV.exists():
        logger.warning("No raw results found.")
        return

    df = pd.read_csv(RAW_CSV)

    # Group by dataset and method
    summary_rows = []

    for dataset in DATASETS:
        ds_df = df[df['dataset'] == dataset]

        for method in ds_df['method'].unique():
            m_df = ds_df[ds_df['method'] == method]

            auroc_mean = m_df['auroc'].mean()
            auroc_std = m_df['auroc'].std()
            auprc_mean = m_df['auprc'].mean()
            auprc_std = m_df['auprc'].std()
            p_at_k_mean = m_df['p_at_k'].mean()
            p_at_k_std = m_df['p_at_k'].std()
            runtime_mean = m_df['runtime_sec'].mean()
            runtime_std = m_df['runtime_sec'].std()
            n_runs = len(m_df)

            summary_rows.append({
                'dataset': dataset,
                'method': method,
                'n_runs': n_runs,
                'auroc_mean': auroc_mean,
                'auroc_std': auroc_std,
                'auroc': f"{auroc_mean:.4f}±{auroc_std:.4f}",
                'auprc_mean': auprc_mean,
                'auprc_std': auprc_std,
                'auprc': f"{auprc_mean:.4f}±{auprc_std:.4f}",
                'p_at_k_mean': p_at_k_mean,
                'p_at_k_std': p_at_k_std,
                'p_at_k': f"{p_at_k_mean:.4f}±{p_at_k_std:.4f}",
                'runtime_mean': runtime_mean,
                'runtime_std': runtime_std,
                'runtime': f"{runtime_mean:.2f}±{runtime_std:.2f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    logger.info(f"Summary saved to {SUMMARY_CSV}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY (mean±std over 10 seeds)")
    print("=" * 80)

    for dataset in DATASETS:
        print(f"\n{dataset.upper()}")
        print("-" * 70)
        ds_summary = summary_df[summary_df['dataset'] == dataset].sort_values('auroc_mean', ascending=False)
        for _, row in ds_summary.iterrows():
            print(f"  {row['method']:15s} | AUROC={row['auroc']:15s} | AUPRC={row['auprc']:15s} | P@100={row['p_at_k']:15s}")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    try:
        run_phase1()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. State saved. Resume by running again.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
