"""
Generate publication-quality plots for the PNKDIF paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'paper' / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Style settings for IEEE conference papers
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.5),  # Single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Color scheme
COLORS = {
    'IF': '#1f77b4',
    'IF_concat': '#aec7e8',
    'DIF': '#ff7f0e',
    'DIF_concat': '#ffbb78',
    'LOF': '#2ca02c',
    'QCAD': '#d62728',
    'ROCOD': '#9467bd',
    'PNKDIF': '#8c564b',
    'PNKDIF_uniform': '#e377c2',
    'PNKDIF_noMLP': '#17becf',
}

METHOD_ORDER = ['IF', 'IF_concat', 'DIF', 'DIF_concat', 'LOF', 'QCAD', 'ROCOD', 'PNKDIF', 'PNKDIF_uniform', 'PNKDIF_noMLP']


def plot_synthetic_comparison():
    """Bar plot comparing methods on synthetic datasets."""
    df = pd.read_csv(RESULTS_DIR / 'phase1_summary.csv')

    datasets = ['syn_linear', 'syn_scale', 'syn_multimodal']
    methods = ['IF', 'DIF', 'LOF', 'QCAD', 'ROCOD', 'PNKDIF']

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

    titles = ['(a) Linear Shift', '(b) Scale', '(c) Multimodal']

    for ax, dataset, title in zip(axes, datasets, titles):
        ds_df = df[df['dataset'] == dataset]

        aurocs = []
        stds = []
        colors = []
        for method in methods:
            row = ds_df[ds_df['method'] == method]
            if len(row) > 0:
                aurocs.append(row['auroc_mean'].values[0])
                stds.append(row['auroc_std'].values[0])
                colors.append(COLORS[method])
            else:
                aurocs.append(0)
                stds.append(0)
                colors.append('gray')

        x = np.arange(len(methods))
        bars = ax.bar(x, aurocs, yerr=stds, color=colors, capsize=2, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        # Highlight best
        best_idx = np.argmax(aurocs)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

    axes[0].set_ylabel('AUROC')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'synthetic_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'synthetic_comparison.png')
    plt.close()
    print(f"Saved: synthetic_comparison.pdf")


def plot_real_data_comparison():
    """Bar plot comparing methods on real datasets."""
    df = pd.read_csv(RESULTS_DIR / 'phase4_summary.csv')

    datasets = ['adult_shift', 'bank_shift', 'cardio']
    methods = ['IF', 'DIF', 'LOF', 'QCAD', 'ROCOD', 'PNKDIF', 'PNKDIF_noMLP']

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

    titles = ['(a) Adult (Shift)', '(b) Bank (Shift)', '(c) Cardio']

    for ax, dataset, title in zip(axes, datasets, titles):
        ds_df = df[df['dataset'] == dataset]

        aurocs = []
        stds = []
        colors = []
        for method in methods:
            row = ds_df[ds_df['method'] == method]
            if len(row) > 0:
                aurocs.append(row['auroc_mean'].values[0])
                stds.append(row['auroc_std'].values[0])
                colors.append(COLORS[method])
            else:
                aurocs.append(0)
                stds.append(0)
                colors.append('gray')

        x = np.arange(len(methods))
        bars = ax.bar(x, aurocs, yerr=stds, color=colors, capsize=2, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylim(0.5, 1.05)

        # Highlight best
        best_idx = np.argmax(aurocs)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

    axes[0].set_ylabel('AUROC')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'real_data_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'real_data_comparison.png')
    plt.close()
    print(f"Saved: real_data_comparison.pdf")


def plot_scalability():
    """Line plot showing runtime vs dataset size."""
    df = pd.read_csv(RESULTS_DIR / 'phase3_summary.csv')

    methods = ['IF', 'DIF', 'QCAD', 'ROCOD', 'PNKDIF']

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for method in methods:
        m_df = df[df['method'] == method].sort_values('n_samples')
        ax.plot(m_df['n_samples'], m_df['runtime_mean'],
                marker='o', markersize=4, label=method, color=COLORS[method])

    ax.set_xlabel('Dataset Size (N)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'scalability.pdf')
    plt.savefig(FIGURES_DIR / 'scalability.png')
    plt.close()
    print(f"Saved: scalability.pdf")


def plot_hyperparameter_sensitivity():
    """Plot hyperparameter sensitivity (K, M, d_h)."""
    df = pd.read_csv(RESULTS_DIR / 'phase2_summary.csv')

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

    # K sensitivity
    ax = axes[0]
    k_df = df[(df['dataset'] == 'syn_linear') & (df['method'].str.startswith('PNKDIF_K'))]
    if len(k_df) > 0:
        k_vals = [int(m.split('K')[1]) for m in k_df['method']]
        aurocs = k_df['auroc_mean'].values
        stds = k_df['auroc_std'].values

        sorted_idx = np.argsort(k_vals)
        k_vals = np.array(k_vals)[sorted_idx]
        aurocs = aurocs[sorted_idx]
        stds = stds[sorted_idx]

        ax.errorbar(k_vals, aurocs, yerr=stds, marker='o', capsize=3, color=COLORS['PNKDIF'])
        ax.set_xlabel('K (neighbors)')
        ax.set_ylabel('AUROC')
        ax.set_title('(a) K Sensitivity')
        ax.set_xscale('log')
        ax.set_ylim(0.97, 1.005)
        ax.grid(True, alpha=0.3)

    # M sensitivity
    ax = axes[1]
    m_df = df[(df['dataset'] == 'syn_linear') & (df['method'].str.startswith('PNKDIF_M'))]
    if len(m_df) > 0:
        m_vals = [int(m.split('M')[1]) for m in m_df['method']]
        aurocs = m_df['auroc_mean'].values
        stds = m_df['auroc_std'].values

        sorted_idx = np.argsort(m_vals)
        m_vals = np.array(m_vals)[sorted_idx]
        aurocs = aurocs[sorted_idx]
        stds = stds[sorted_idx]

        ax.errorbar(m_vals, aurocs, yerr=stds, marker='s', capsize=3, color=COLORS['PNKDIF'])
        ax.set_xlabel('M (projections)')
        ax.set_title('(b) M Sensitivity')
        ax.set_ylim(0.97, 1.005)
        ax.grid(True, alpha=0.3)

    # d_h sensitivity
    ax = axes[2]
    dh_df = df[(df['dataset'] == 'syn_linear') & (df['method'].str.startswith('PNKDIF_dh'))]
    if len(dh_df) > 0:
        dh_vals = [int(m.split('dh')[1]) for m in dh_df['method']]
        aurocs = dh_df['auroc_mean'].values
        stds = dh_df['auroc_std'].values

        sorted_idx = np.argsort(dh_vals)
        dh_vals = np.array(dh_vals)[sorted_idx]
        aurocs = aurocs[sorted_idx]
        stds = stds[sorted_idx]

        ax.errorbar(dh_vals, aurocs, yerr=stds, marker='^', capsize=3, color=COLORS['PNKDIF'])
        ax.set_xlabel('$d_h$ (hidden dim)')
        ax.set_title('(c) $d_h$ Sensitivity')
        ax.set_xscale('log')
        ax.set_ylim(0.97, 1.005)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hyperparameter_sensitivity.pdf')
    plt.savefig(FIGURES_DIR / 'hyperparameter_sensitivity.png')
    plt.close()
    print(f"Saved: hyperparameter_sensitivity.pdf")


def plot_ablation():
    """Bar plot for ablation study."""
    df = pd.read_csv(RESULTS_DIR / 'phase2_summary.csv')

    # Get syn_nonlinear results for ablation
    ds_df = df[df['dataset'] == 'syn_nonlinear']

    variants = ['PNKDIF', 'PNKDIF_uniform', 'PNKDIF_noMLP', 'DIF']
    labels = ['PNKDIF\n(full)', 'w/o kernel\nweighting', 'w/o MLP\nprojection', 'w/o peer\nnorm (DIF)']

    aurocs = []
    stds = []
    for var in variants:
        row = ds_df[ds_df['method'] == var]
        if len(row) > 0:
            aurocs.append(row['auroc_mean'].values[0])
            stds.append(row['auroc_std'].values[0])
        else:
            aurocs.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(variants))
    colors = [COLORS.get(v, 'gray') for v in variants]
    bars = ax.bar(x, aurocs, yerr=stds, color=colors, capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('AUROC')
    ax.set_title('Ablation Study (Syn-Nonlinear)')
    ax.set_ylim(0.8, 1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ablation.pdf')
    plt.savefig(FIGURES_DIR / 'ablation.png')
    plt.close()
    print(f"Saved: ablation.pdf")


def plot_swap_comparison():
    """Show difficulty of swap injection anomalies."""
    df = pd.read_csv(RESULTS_DIR / 'phase4_summary.csv')

    datasets = ['adult_swap', 'bank_swap']
    methods = ['IF', 'DIF', 'LOF', 'QCAD', 'ROCOD', 'PNKDIF']

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)

    titles = ['(a) Adult (Swap)', '(b) Bank (Swap)']

    for ax, dataset, title in zip(axes, datasets, titles):
        ds_df = df[df['dataset'] == dataset]

        aurocs = []
        stds = []
        colors = []
        for method in methods:
            row = ds_df[ds_df['method'] == method]
            if len(row) > 0:
                aurocs.append(row['auroc_mean'].values[0])
                stds.append(row['auroc_std'].values[0])
                colors.append(COLORS[method])
            else:
                aurocs.append(0)
                stds.append(0)
                colors.append('gray')

        x = np.arange(len(methods))
        ax.bar(x, aurocs, yerr=stds, color=colors, capsize=2, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylim(0.4, 0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')

    axes[0].set_ylabel('AUROC')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'swap_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'swap_comparison.png')
    plt.close()
    print(f"Saved: swap_comparison.pdf")


if __name__ == '__main__':
    print("Generating plots...")
    plot_synthetic_comparison()
    plot_real_data_comparison()
    plot_scalability()
    plot_hyperparameter_sensitivity()
    plot_ablation()
    plot_swap_comparison()
    print("\nAll plots generated!")
