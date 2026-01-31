"""
Smoke test: Verify all methods work correctly before full experiment.

Checks:
1. All methods produce scores with correct direction
2. Anomalies rank high on Syn-Linear (where contextual methods should excel)
3. No runtime errors
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from data.synthetic import get_dataset
from models.baselines import get_method, METHODS
from models.pnkdif import PNKDIF, PNKDIFConfig, PNKDIFUniform, PNKDIFNoMLP
from evaluation.metrics import compute_metrics, check_score_direction


def run_smoke_test():
    print("=" * 60)
    print("SMOKE TEST: Baseline Correctness Check")
    print("=" * 60)

    SEED = 42
    N_SAMPLES = 2000  # Smaller for quick test

    # All methods to test
    methods = {
        'IF': lambda: get_method('IF', random_state=SEED),
        'IF_concat': lambda: get_method('IF_concat', random_state=SEED),
        'DIF': lambda: get_method('DIF', random_state=SEED),
        'DIF_concat': lambda: get_method('DIF_concat', random_state=SEED),
        'LOF': lambda: get_method('LOF'),
        'QCAD': lambda: get_method('QCAD', random_state=SEED),
        'ROCOD': lambda: get_method('ROCOD', random_state=SEED),
        'PNKDIF': lambda: PNKDIF(PNKDIFConfig(n_neighbors=50, random_state=SEED)),
        'PNKDIF_uniform': lambda: PNKDIFUniform(PNKDIFConfig(n_neighbors=50, random_state=SEED)),
        'PNKDIF_noMLP': lambda: PNKDIFNoMLP(PNKDIFConfig(n_neighbors=50, random_state=SEED)),
    }

    datasets = ['syn_linear', 'syn_scale', 'syn_multimodal']

    all_passed = True
    results = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print("=" * 60)

        context, behavior, labels = get_dataset(
            ds_name, n_samples=N_SAMPLES, random_state=SEED
        )

        print(f"  Samples: {len(labels)}, Anomalies: {labels.sum():.0f} ({labels.mean()*100:.1f}%)")
        print(f"  Context shape: {context.shape}, Behavior shape: {behavior.shape}")
        print()

        for method_name, method_fn in methods.items():
            try:
                model = method_fn()
                start = time.time()
                scores = model.fit_predict(context, behavior)
                elapsed = time.time() - start

                # Check score direction
                direction_ok = check_score_direction(labels, scores)

                # Compute metrics
                metrics = compute_metrics(labels, scores, k=100)

                status = "PASS" if direction_ok else "FAIL"
                if not direction_ok:
                    all_passed = False

                print(f"  {method_name:15s} | {status:4s} | AUROC={metrics['auroc']:.3f} | "
                      f"AUPRC={metrics['auprc']:.3f} | P@100={metrics['p_at_k']:.3f} | "
                      f"Time={elapsed:.2f}s")

                results.append({
                    'dataset': ds_name,
                    'method': method_name,
                    'direction_ok': direction_ok,
                    'auroc': metrics['auroc'],
                    'auprc': metrics['auprc'],
                    'p_at_k': metrics['p_at_k'],
                    'time': elapsed
                })

            except Exception as e:
                print(f"  {method_name:15s} | ERROR: {e}")
                all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_passed:
        print("\nAll methods passed direction check!")
    else:
        print("\nSome methods FAILED direction check - investigate before proceeding!")

    # Show Syn-Linear rankings (where contextual methods should excel)
    print("\nSyn-Linear AUROC Rankings (contextual methods should be top):")
    syn_linear_results = [r for r in results if r['dataset'] == 'syn_linear']
    syn_linear_results.sort(key=lambda x: x['auroc'], reverse=True)
    for i, r in enumerate(syn_linear_results, 1):
        marker = "*" if 'PNKDIF' in r['method'] or r['method'] in ['QCAD', 'ROCOD'] else " "
        print(f"  {i}. {r['method']:15s} AUROC={r['auroc']:.3f} {marker}")

    # Check if PNKDIF beats non-contextual methods on syn_linear
    pnkdif_auroc = next(r['auroc'] for r in syn_linear_results if r['method'] == 'PNKDIF')
    if_auroc = next(r['auroc'] for r in syn_linear_results if r['method'] == 'IF')
    dif_auroc = next(r['auroc'] for r in syn_linear_results if r['method'] == 'DIF')

    print(f"\nContextual advantage check (Syn-Linear):")
    print(f"  PNKDIF ({pnkdif_auroc:.3f}) vs IF ({if_auroc:.3f}): "
          f"{'BETTER' if pnkdif_auroc > if_auroc else 'WORSE'}")
    print(f"  PNKDIF ({pnkdif_auroc:.3f}) vs DIF ({dif_auroc:.3f}): "
          f"{'BETTER' if pnkdif_auroc > dif_auroc else 'WORSE'}")

    return all_passed, results


if __name__ == '__main__':
    passed, results = run_smoke_test()
    sys.exit(0 if passed else 1)
