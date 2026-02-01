"""
Aggregate all experiment results and generate summary tables.
Filters to only the 9 models: IF, IF_concat, DIF, DIF_concat, LOF, QCAD, ROCOD, PNKIF, PNKDIF
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'

# Map old model names to new names
MODEL_RENAME = {
    'PNKDIF_noMLP': 'PNKIF',
}

# Only these 9 models in final results
FINAL_MODELS = ['IF', 'IF_concat', 'DIF', 'DIF_concat', 'LOF', 'QCAD', 'ROCOD', 'PNKIF', 'PNKDIF']

# Display order for models
MODEL_ORDER = {m: i for i, m in enumerate(FINAL_MODELS)}

def load_all_results():
    """Load all phase4_* result CSVs and combine."""
    all_data = []

    # Find all phase4 result files
    csv_files = list(RESULTS_DIR.glob('phase4_*_raw.csv'))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            all_data.append(df)
            print(f"Loaded {csv_path.name}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_path.name}: {e}")

    if not all_data:
        print("No data found!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    return combined


def filter_and_rename(df):
    """Filter to 9 models and rename PNKDIF_noMLP -> PNKIF."""
    # Rename old model names
    df['method'] = df['method'].replace(MODEL_RENAME)

    # Filter to only final models
    df = df[df['method'].isin(FINAL_MODELS)]

    # Remove duplicates (keep latest by timestamp if duplicate)
    df = df.sort_values('timestamp').drop_duplicates(
        subset=['dataset', 'method', 'seed'], keep='last'
    )

    return df


def compute_summary(df):
    """Compute mean and std for each dataset/method pair."""
    summary = df.groupby(['dataset', 'method']).agg({
        'auroc': ['mean', 'std', 'count'],
        'auprc': ['mean', 'std'],
        'p_at_k': ['mean', 'std'],
        'n_samples': 'first',
        'n_context': 'first',
        'n_behavior': 'first',
        'anomaly_rate': 'first',
    }).reset_index()

    # Flatten column names
    summary.columns = ['dataset', 'method',
                       'auroc_mean', 'auroc_std', 'n_seeds',
                       'auprc_mean', 'auprc_std',
                       'p_at_k_mean', 'p_at_k_std',
                       'n_samples', 'n_context', 'n_behavior', 'anomaly_rate']

    # Add model order for sorting
    summary['model_order'] = summary['method'].map(MODEL_ORDER)
    summary = summary.sort_values(['dataset', 'model_order'])

    return summary


def create_pivot_table(summary, metric='auroc_mean'):
    """Create a pivot table with datasets as rows and methods as columns."""
    pivot = summary.pivot(index='dataset', columns='method', values=metric)

    # Reorder columns
    pivot = pivot[[m for m in FINAL_MODELS if m in pivot.columns]]

    return pivot


def highlight_best(pivot):
    """Return the same table with best values marked."""
    result = pivot.copy()
    for idx in result.index:
        row = result.loc[idx]
        best_val = row.max()
        for col in result.columns:
            if row[col] == best_val:
                result.loc[idx, col] = f"**{row[col]:.4f}**"
            else:
                result.loc[idx, col] = f"{row[col]:.4f}"
    return result


def main():
    print("=" * 60)
    print("CDIF Results Aggregation")
    print("=" * 60)

    # Load all data
    df = load_all_results()
    if df.empty:
        return

    # Filter and rename
    df = filter_and_rename(df)
    print(f"\nAfter filtering to 9 models: {len(df)} rows")

    # Show dataset coverage
    print("\nDataset coverage (methods x seeds):")
    coverage = df.groupby('dataset')['method'].value_counts().unstack(fill_value=0)
    print(coverage)

    # Compute summary
    summary = compute_summary(df)

    # Create AUROC pivot table
    auroc_pivot = create_pivot_table(summary, 'auroc_mean')

    print("\n" + "=" * 60)
    print("AUROC Mean (across 10 seeds)")
    print("=" * 60)
    print(auroc_pivot.round(4).to_string())

    # Save to markdown
    output_md = RESULTS_DIR / 'summary_9models.md'
    with open(output_md, 'w') as f:
        f.write("# CDIF Experiment Results Summary\n\n")
        f.write("**Models:** IF, IF_concat, DIF, DIF_concat, LOF, QCAD, ROCOD, PNKIF, PNKDIF\n\n")

        f.write("## AUROC (Mean across seeds)\n\n")
        auroc_md = highlight_best(auroc_pivot)
        f.write(auroc_md.to_markdown())
        f.write("\n\n")

        f.write("## AUPRC (Mean across seeds)\n\n")
        auprc_pivot = create_pivot_table(summary, 'auprc_mean')
        auprc_md = highlight_best(auprc_pivot)
        f.write(auprc_md.to_markdown())
        f.write("\n\n")

        f.write("## Dataset Details\n\n")
        dataset_info = summary[['dataset', 'n_samples', 'n_context', 'n_behavior', 'anomaly_rate', 'n_seeds']].drop_duplicates('dataset')
        f.write(dataset_info.to_markdown(index=False))
        f.write("\n")

    print(f"\nSummary saved to: {output_md}")

    # Also save raw summary as CSV
    summary.to_csv(RESULTS_DIR / 'summary_9models.csv', index=False)
    print(f"CSV saved to: {RESULTS_DIR / 'summary_9models.csv'}")


if __name__ == '__main__':
    main()
