"""
Phase 3 Runner: Scaling Studies

Test computational scaling as N increases.
Verify O(N log N) complexity claim.
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
from models.pnkdif import PNKDIF, PNKDIFConfig
from evaluation.metrics import compute_metrics

# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456]  # 3 seeds for timing
N_VALUES = [1000, 2000, 5000, 10000, 20000, 50000]
ANOMALY_RATE = 0.05

# Paths
RESULTS_DIR = ROOT / 'results'
LOGS_DIR = ROOT / 'logs'
RAW_CSV = RESULTS_DIR / 'phase3_raw.csv'
SUMMARY_CSV = RESULTS_DIR / 'phase3_summary.csv'
STATE_FILE = RESULTS_DIR / 'phase3_state.json'
LOG_FILE = LOGS_DIR / 'phase3.log'

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

RAW_HEADERS = ['method', 'n_samples', 'seed', 'runtime_sec', 'auroc', 'timestamp']

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
# SCALING EXPERIMENT
# =============================================================================

def run_scaling():
    logger.info("=" * 60)
    logger.info("PHASE 3: Scaling Studies")
    logger.info("=" * 60)

    state = load_state()
    completed = set(state['completed'])

    init_csv(RAW_CSV, RAW_HEADERS)

    methods = {
        'IF': lambda s: get_method('IF', random_state=s),
        'DIF': lambda s: get_method('DIF', random_state=s),
        'PNKDIF': lambda s: PNKDIF(PNKDIFConfig(n_neighbors=min(100, s//20), random_state=s)),
        'QCAD': lambda s: get_method('QCAD', random_state=s),
        'ROCOD': lambda s: get_method('ROCOD', random_state=s),
    }

    for N in N_VALUES:
        logger.info(f"\nN = {N}")
        logger.info("-" * 40)

        for seed in SEEDS:
            # Generate data once per (N, seed)
            context, behavior, labels = get_dataset(
                'syn_linear', n_samples=N, anomaly_rate=ANOMALY_RATE, random_state=seed
            )

            for method_name, method_fn in methods.items():
                key = run_key(method_name, N, seed)
                if key in completed:
                    continue

                try:
                    # Adjust K for PNKDIF based on N
                    if method_name == 'PNKDIF':
                        K = min(100, N // 20)
                        model = PNKDIF(PNKDIFConfig(n_neighbors=K, random_state=seed))
                    else:
                        model = method_fn(seed)

                    # Time the run
                    start = time.time()
                    scores = model.fit_predict(context, behavior)
                    runtime = time.time() - start

                    # Quick AUROC check
                    from sklearn.metrics import roc_auc_score
                    auroc = roc_auc_score(labels, scores)

                    row = [method_name, N, seed, f"{runtime:.4f}", f"{auroc:.4f}",
                           datetime.now().isoformat()]
                    append_csv(RAW_CSV, row)

                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

                    logger.info(f"  {method_name:10s} | N={N:6d} | Time={runtime:.3f}s | AUROC={auroc:.4f}")

                except Exception as e:
                    logger.error(f"  {method_name} | N={N} | ERROR: {e}")
                    completed.add(key)
                    state['completed'] = list(completed)
                    save_state(state)

    generate_summary()

def generate_summary():
    """Generate scaling summary."""
    import pandas as pd

    if not RAW_CSV.exists():
        return

    df = pd.read_csv(RAW_CSV)

    summary_rows = []
    for method in df['method'].unique():
        m_df = df[df['method'] == method]
        for N in m_df['n_samples'].unique():
            n_df = m_df[m_df['n_samples'] == N]
            summary_rows.append({
                'method': method,
                'n_samples': N,
                'runtime_mean': n_df['runtime_sec'].mean(),
                'runtime_std': n_df['runtime_sec'].std(),
                'auroc_mean': n_df['auroc'].mean(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # Print scaling table
    print("\n" + "=" * 80)
    print("SCALING SUMMARY (runtime in seconds)")
    print("=" * 80)

    pivot = summary_df.pivot(index='n_samples', columns='method', values='runtime_mean')
    print(pivot.to_string())

    logger.info(f"Summary saved to {SUMMARY_CSV}")
    logger.info("Phase 3 complete!")

if __name__ == '__main__':
    try:
        run_scaling()
    except KeyboardInterrupt:
        logger.info("Interrupted. State saved.")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
