"""
Phase 2 Runner: Ablation Studies

1. Run syn_nonlinear dataset (tests non-linear boundaries)
2. Hyperparameter sensitivity analysis (K, M, d_h)
3. Component ablations on syn_linear and syn_nonlinear
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
from models.baselines import get_method
from models.pnkdif import PNKDIF, PNKDIFConfig, PNKDIFUniform, PNKDIFNoMLP
from evaluation.metrics import compute_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1011]  # 5 seeds for ablations (faster)
N_SAMPLES = 10000
ANOMALY_RATE = 0.05
K_FOR_P_AT_K = 100

# Paths
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
RAW_CSV = RESULTS_DIR / 'phase2_raw.csv'
SUMMARY_CSV = RESULTS_DIR / 'phase2_summary.csv'
ERRORS_CSV = RESULTS_DIR / 'phase2_errors.csv'
STATE_FILE = RESULTS_DIR / 'phase2_state.json'
LOG_FILE = LOGS_DIR / 'phase2.log'

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
# CSV WRITING
# =============================================================================

RAW_HEADERS = ['experiment', 'dataset', 'method', 'param_name', 'param_value',
               'seed', 'auroc', 'auprc', 'p_at_k', 'runtime_sec', 'timestamp']
ERROR_HEADERS = ['experiment', 'dataset', 'method', 'seed', 'error', 'traceback', 'timestamp']

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
# EXPERIMENT RUNNERS
# =============================================================================

def run_single(context, behavior, labels, model):
    """Run a single experiment and return metrics."""
    start = time.time()
    scores = model.fit_predict(context, behavior)
    runtime = time.time() - start
    metrics = compute_metrics(labels, scores, k=K_FOR_P_AT_K)
    return {**metrics, 'runtime': runtime}

def run_syn_nonlinear_experiment(state, completed):
    """Run all methods on syn_nonlinear dataset."""
    logger.info("=" * 60)
    logger.info("Experiment 1: syn_nonlinear dataset")
    logger.info("=" * 60)

    methods = {
        'IF': lambda s: get_method('IF', random_state=s),
        'IF_concat': lambda s: get_method('IF_concat', random_state=s),
        'DIF': lambda s: get_method('DIF', random_state=s),
        'DIF_concat': lambda s: get_method('DIF_concat', random_state=s),
        'LOF': lambda s: get_method('LOF'),
        'QCAD': lambda s: get_method('QCAD', random_state=s),
        'ROCOD': lambda s: get_method('ROCOD', random_state=s),
        'PNKDIF': lambda s: PNKDIF(PNKDIFConfig(n_neighbors=100, random_state=s)),
        'PNKDIF_uniform': lambda s: PNKDIFUniform(PNKDIFConfig(n_neighbors=100, random_state=s)),
        'PNKDIF_noMLP': lambda s: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=100, random_state=s)),
    }

    for seed in SEEDS:
        context, behavior, labels = get_dataset('syn_nonlinear', n_samples=N_SAMPLES,
                                                 anomaly_rate=ANOMALY_RATE, random_state=seed)

        for method_name, method_fn in methods.items():
            key = run_key('syn_nonlinear', method_name, seed)
            if key in completed:
                continue

            logger.info(f"syn_nonlinear | {method_name} | seed={seed}")

            try:
                model = method_fn(seed)
                result = run_single(context, behavior, labels, model)

                row = ['syn_nonlinear_baseline', 'syn_nonlinear', method_name, 'default', 'default',
                       seed, f"{result['auroc']:.6f}", f"{result['auprc']:.6f}",
                       f"{result['p_at_k']:.6f}", f"{result['runtime']:.3f}",
                       datetime.now().isoformat()]
                append_csv(RAW_CSV, row)

                completed.add(key)
                state['completed'] = list(completed)
                save_state(state)

                logger.info(f"  -> AUROC={result['auroc']:.4f}")

            except Exception as e:
                append_csv(ERRORS_CSV, ['syn_nonlinear_baseline', 'syn_nonlinear', method_name,
                                        seed, str(e), traceback.format_exc(), datetime.now().isoformat()])
                logger.error(f"  -> ERROR: {e}")
                completed.add(key)
                state['completed'] = list(completed)
                save_state(state)

def run_k_sensitivity(state, completed):
    """Hyperparameter sensitivity: number of neighbors K."""
    logger.info("=" * 60)
    logger.info("Experiment 2: K (neighbors) sensitivity")
    logger.info("=" * 60)

    K_VALUES = [10, 25, 50, 100, 200, 500]
    datasets = ['syn_linear', 'syn_nonlinear']

    for dataset in datasets:
        for seed in SEEDS:
            context, behavior, labels = get_dataset(dataset, n_samples=N_SAMPLES,
                                                     anomaly_rate=ANOMALY_RATE, random_state=seed)

            for K in K_VALUES:
                key = run_key('K_sensitivity', dataset, K, seed)
                if key in completed:
                    continue

                logger.info(f"K={K} | {dataset} | seed={seed}")

                try:
                    model = PNKDIF(PNKDIFConfig(n_neighbors=K, random_state=seed))
                    result = run_single(context, behavior, labels, model)

                    row = ['K_sensitivity', dataset, 'PNKDIF', 'K', K,
                           seed, f"{result['auroc']:.6f}", f"{result['auprc']:.6f}",
                           f"{result['p_at_k']:.6f}", f"{result['runtime']:.3f}",
                           datetime.now().isoformat()]
                    append_csv(RAW_CSV, row)

                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

                    logger.info(f"  -> AUROC={result['auroc']:.4f}")

                except Exception as e:
                    append_csv(ERRORS_CSV, ['K_sensitivity', dataset, 'PNKDIF',
                                            seed, str(e), traceback.format_exc(), datetime.now().isoformat()])
                    logger.error(f"  -> ERROR: {e}")
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

def run_m_sensitivity(state, completed):
    """Hyperparameter sensitivity: number of projections M."""
    logger.info("=" * 60)
    logger.info("Experiment 3: M (projections) sensitivity")
    logger.info("=" * 60)

    M_VALUES = [1, 2, 4, 6, 8, 10]
    datasets = ['syn_linear', 'syn_nonlinear']

    for dataset in datasets:
        for seed in SEEDS:
            context, behavior, labels = get_dataset(dataset, n_samples=N_SAMPLES,
                                                     anomaly_rate=ANOMALY_RATE, random_state=seed)

            for M in M_VALUES:
                key = run_key('M_sensitivity', dataset, M, seed)
                if key in completed:
                    continue

                logger.info(f"M={M} | {dataset} | seed={seed}")

                try:
                    model = PNKDIF(PNKDIFConfig(n_projections=M, random_state=seed))
                    result = run_single(context, behavior, labels, model)

                    row = ['M_sensitivity', dataset, 'PNKDIF', 'M', M,
                           seed, f"{result['auroc']:.6f}", f"{result['auprc']:.6f}",
                           f"{result['p_at_k']:.6f}", f"{result['runtime']:.3f}",
                           datetime.now().isoformat()]
                    append_csv(RAW_CSV, row)

                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

                    logger.info(f"  -> AUROC={result['auroc']:.4f}")

                except Exception as e:
                    append_csv(ERRORS_CSV, ['M_sensitivity', dataset, 'PNKDIF',
                                            seed, str(e), traceback.format_exc(), datetime.now().isoformat()])
                    logger.error(f"  -> ERROR: {e}")
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

def run_hidden_dim_sensitivity(state, completed):
    """Hyperparameter sensitivity: hidden dimension d_h."""
    logger.info("=" * 60)
    logger.info("Experiment 4: d_h (hidden dim) sensitivity")
    logger.info("=" * 60)

    DH_VALUES = [16, 32, 64, 128, 256]
    datasets = ['syn_linear', 'syn_nonlinear']

    for dataset in datasets:
        for seed in SEEDS:
            context, behavior, labels = get_dataset(dataset, n_samples=N_SAMPLES,
                                                     anomaly_rate=ANOMALY_RATE, random_state=seed)

            for dh in DH_VALUES:
                key = run_key('dh_sensitivity', dataset, dh, seed)
                if key in completed:
                    continue

                logger.info(f"d_h={dh} | {dataset} | seed={seed}")

                try:
                    model = PNKDIF(PNKDIFConfig(hidden_dim=dh, random_state=seed))
                    result = run_single(context, behavior, labels, model)

                    row = ['dh_sensitivity', dataset, 'PNKDIF', 'd_h', dh,
                           seed, f"{result['auroc']:.6f}", f"{result['auprc']:.6f}",
                           f"{result['p_at_k']:.6f}", f"{result['runtime']:.3f}",
                           datetime.now().isoformat()]
                    append_csv(RAW_CSV, row)

                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

                    logger.info(f"  -> AUROC={result['auroc']:.4f}")

                except Exception as e:
                    append_csv(ERRORS_CSV, ['dh_sensitivity', dataset, 'PNKDIF',
                                            seed, str(e), traceback.format_exc(), datetime.now().isoformat()])
                    logger.error(f"  -> ERROR: {e}")
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

# =============================================================================
# SUMMARY
# =============================================================================

def generate_summary():
    """Generate summary statistics."""
    import pandas as pd

    if not RAW_CSV.exists():
        logger.warning("No raw results found.")
        return

    df = pd.read_csv(RAW_CSV)

    summary_rows = []

    for experiment in df['experiment'].unique():
        exp_df = df[df['experiment'] == experiment]

        for dataset in exp_df['dataset'].unique():
            ds_df = exp_df[exp_df['dataset'] == dataset]

            for method in ds_df['method'].unique():
                m_df = ds_df[ds_df['method'] == method]

                # Group by param if exists
                if 'param_value' in m_df.columns:
                    for param_val in m_df['param_value'].unique():
                        p_df = m_df[m_df['param_value'] == param_val]
                        param_name = p_df['param_name'].iloc[0]

                        summary_rows.append({
                            'experiment': experiment,
                            'dataset': dataset,
                            'method': method,
                            'param_name': param_name,
                            'param_value': param_val,
                            'n_runs': len(p_df),
                            'auroc_mean': p_df['auroc'].mean(),
                            'auroc_std': p_df['auroc'].std(),
                            'auprc_mean': p_df['auprc'].mean(),
                            'auprc_std': p_df['auprc'].std(),
                            'runtime_mean': p_df['runtime_sec'].mean(),
                        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    logger.info(f"Summary saved to {SUMMARY_CSV}")

    # Print key results
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY")
    print("=" * 80)

    # syn_nonlinear results
    nonlinear = summary_df[summary_df['experiment'] == 'syn_nonlinear_baseline']
    if len(nonlinear) > 0:
        print("\nSYN_NONLINEAR (non-linear anomaly boundaries)")
        print("-" * 60)
        nonlinear = nonlinear.sort_values('auroc_mean', ascending=False)
        for _, row in nonlinear.iterrows():
            print(f"  {row['method']:15s} | AUROC={row['auroc_mean']:.4f}±{row['auroc_std']:.4f}")

# =============================================================================
# MAIN
# =============================================================================

def run_phase2():
    logger.info("=" * 60)
    logger.info("PHASE 2: Ablation Studies")
    logger.info("=" * 60)

    state = load_state()
    completed = set(state['completed'])

    init_csv(RAW_CSV, RAW_HEADERS)
    init_csv(ERRORS_CSV, ERROR_HEADERS)

    # Run experiments
    run_syn_nonlinear_experiment(state, completed)
    run_k_sensitivity(state, completed)
    run_m_sensitivity(state, completed)
    run_hidden_dim_sensitivity(state, completed)

    logger.info("=" * 60)
    logger.info("Generating summary...")
    generate_summary()
    logger.info("Phase 2 complete!")

if __name__ == '__main__':
    try:
        run_phase2()
    except KeyboardInterrupt:
        logger.info("Interrupted. State saved.")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
