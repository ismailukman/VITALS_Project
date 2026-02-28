"""
Visualization script for Gastric-Motion Synchrony Analysis (Version 3)

Creates comprehensive visualizations of PLV results with mismatch null distribution
and multiple comparison corrections.

Date: 2025-12-03
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# Setup paths
PARENT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = PARENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Input files
PLV_FILE = PROJECT_ROOT / "dataframes" / "plvs_egg_w_motion_v3.csv"
MOTION_FILE = PROJECT_ROOT / "dataframes" / "motion_summary_v3.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "plots" / "gastric_motion_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Motion parameter labels
MOTION_LABELS = {
    'trans_x': 'Translation X\n(Left-Right)',
    'trans_y': 'Translation Y\n(Anterior-Posterior)',
    'trans_z': 'Translation Z\n(Superior-Inferior)',
    'rot_x': 'Rotation X\n(Pitch)',
    'rot_y': 'Rotation Y\n(Roll)',
    'rot_z': 'Rotation Z\n(Yaw)'
}


def load_data():
    """Load PLV and motion summary data."""
    print("Loading data...")
    plv_data = pd.read_csv(PLV_FILE)
    motion_data = pd.read_csv(MOTION_FILE)

    print(f"  PLV data: {len(plv_data)} rows")
    print(f"  Motion data: {len(motion_data)} rows")

    return plv_data, motion_data


def plot_plv_distributions(plv_data, save=True):
    """
    Plot 1: PLV distributions - Empirical vs Null
    Shows overall synchrony strength compared to chance expectation
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Overlapping histograms
    axes[0].hist(plv_data['plv_empirical'], bins=40, alpha=0.6,
                 label=f'Empirical (μ={plv_data["plv_empirical"].mean():.3f})',
                 color='darkblue', edgecolor='black')
    axes[0].hist(plv_data['plv_null_mean'], bins=40, alpha=0.6,
                 label=f'Null (μ={plv_data["plv_null_mean"].mean():.3f})',
                 color='gray', edgecolor='black')
    axes[0].axvline(plv_data['plv_empirical'].mean(), color='darkblue',
                    linestyle='--', linewidth=2, label='Empirical Mean')
    axes[0].axvline(plv_data['plv_null_mean'].mean(), color='gray',
                    linestyle='--', linewidth=2, label='Null Mean')
    axes[0].set_xlabel('Phase Locking Value (PLV)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('A. PLV Distribution: Empirical vs Null',
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Panel B: Violin plot by motion parameter
    plv_melted = plv_data.melt(
        id_vars=['motion_param'],
        value_vars=['plv_empirical', 'plv_null_mean'],
        var_name='Type',
        value_name='PLV'
    )
    plv_melted['Type'] = plv_melted['Type'].map({
        'plv_empirical': 'Empirical',
        'plv_null_mean': 'Null'
    })

    sns.violinplot(data=plv_melted, x='motion_param', y='PLV', hue='Type',
                   split=False, ax=axes[1], inner='quartile')
    axes[1].set_xlabel('Motion Parameter', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('PLV', fontsize=12, fontweight='bold')
    axes[1].set_title('B. PLV by Motion Parameter', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels([MOTION_LABELS[x.get_text()] for x in axes[1].get_xticklabels()],
                            fontsize=9)
    axes[1].legend(fontsize=10, title='Distribution Type')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "01_plv_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def plot_significance_summary(plv_data, save=True):
    """
    Plot 2: Significance summary across correction methods
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    correction_methods = [
        ('sig_uncorrected', 'Uncorrected\n(p < 0.05)', 'lightcoral'),
        ('sig_fdr', 'FDR Corrected\n(q < 0.05)', 'orange'),
        ('sig_bonferroni', 'Bonferroni\n(p < 0.05)', 'darkred')
    ]

    for idx, (col, label, color) in enumerate(correction_methods):
        # Count significant findings per motion parameter
        sig_counts = plv_data[plv_data[col]].groupby('motion_param').size()
        total_counts = plv_data.groupby('motion_param').size()
        sig_percent = (sig_counts / total_counts * 100).fillna(0)

        # Bar plot
        ax = axes[idx]
        bars = ax.bar(range(len(sig_percent)), sig_percent.values,
                      color=color, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, sig_percent.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.0f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(sig_percent)))
        ax.set_xticklabels([MOTION_LABELS[x] for x in sig_percent.index],
                           fontsize=9)
        ax.set_ylabel('% Significant', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.axhline(5, color='black', linestyle='--', linewidth=1,
                   label='Expected by chance (5%)')
        ax.grid(alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Significant Findings by Correction Method',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "02_significance_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def plot_subject_heatmap(plv_data, save=True):
    """
    Plot 3: Heatmap of PLV across subjects and motion parameters
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: Empirical PLV
    pivot_empirical = plv_data.pivot_table(
        index='subject',
        columns='motion_param',
        values='plv_empirical',
        aggfunc='mean'
    )

    sns.heatmap(pivot_empirical, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[0], cbar_kws={'label': 'PLV'}, linewidths=0.5)
    axes[0].set_title('A. Empirical PLV by Subject', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Motion Parameter', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Subject', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels([MOTION_LABELS[x.get_text()] for x in axes[0].get_xticklabels()],
                            rotation=45, ha='right')

    # Panel B: Significance (FDR-corrected)
    pivot_sig = plv_data.pivot_table(
        index='subject',
        columns='motion_param',
        values='sig_fdr',
        aggfunc='sum'
    )

    sns.heatmap(pivot_sig, annot=True, fmt='.0f', cmap='RdYlGn',
                ax=axes[1], cbar_kws={'label': 'Number of Significant Runs'},
                linewidths=0.5, vmin=0, vmax=pivot_sig.max().max())
    axes[1].set_title('B. Significant Findings (FDR) by Subject',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Motion Parameter', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Subject', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels([MOTION_LABELS[x.get_text()] for x in axes[1].get_xticklabels()],
                            rotation=45, ha='right')

    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "03_subject_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def plot_motion_amplitude_vs_plv(plv_data, motion_data, save=True):
    """
    Plot 4: Relationship between motion amplitude and PLV
    """
    # Merge datasets
    merged = plv_data.merge(motion_data, on=['subject', 'run'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    motion_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    for idx, param in enumerate(motion_params):
        ax = axes[idx]

        # Filter for this motion parameter
        data_subset = merged[merged['motion_param'] == param]

        # Get motion amplitude column
        motion_col = f'{param}_mean_abs'

        if motion_col not in data_subset.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(MOTION_LABELS[param])
            continue

        x = data_subset[motion_col]
        y = data_subset['plv_empirical']

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50,
                   c=data_subset['sig_fdr'].map({True: 'red', False: 'gray'}),
                   edgecolors='black', linewidth=0.5)

        # Regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Correlation
            r, p_val = stats.pearsonr(x, y)
            ax.text(0.05, 0.95, f'r={r:.3f}\np={p_val:.3f}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'{param.replace("_", " ").title()} Amplitude',
                      fontsize=10, fontweight='bold')
        ax.set_ylabel('PLV', fontsize=10, fontweight='bold')
        ax.set_title(MOTION_LABELS[param], fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Significant (FDR)'),
        Patch(facecolor='gray', edgecolor='black', label='Not Significant')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Motion Amplitude vs PLV by Parameter',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "04_motion_amplitude_vs_plv.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def plot_pvalue_distributions(plv_data, save=True):
    """
    Plot 5: P-value distributions and QQ plots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Panel A: P-value histogram
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(plv_data['p_value'], bins=50, edgecolor='black',
             color='steelblue', alpha=0.7)
    ax1.axhline(len(plv_data) / 50, color='red', linestyle='--',
                linewidth=2, label='Uniform expectation')
    ax1.set_xlabel('P-value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('A. P-value Distribution (should be uniform under null)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Panel B: Uncorrected p-values by motion parameter
    ax2 = fig.add_subplot(gs[1, 0])
    p_by_motion = [plv_data[plv_data['motion_param'] == param]['p_value'].values
                   for param in motion_params]
    bp = ax2.boxplot(p_by_motion, labels=[MOTION_LABELS[x] for x in motion_params],
                     patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax2.set_ylabel('P-value', fontsize=11, fontweight='bold')
    ax2.set_title('B. P-values by Motion Parameter', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([MOTION_LABELS[x] for x in motion_params],
                        rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')

    # Panel C: QQ plot (uncorrected)
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_p = np.sort(plv_data['p_value'])
    expected_p = np.linspace(0, 1, len(sorted_p))
    ax3.scatter(expected_p, sorted_p, alpha=0.5, s=20)
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=2)
    ax3.set_xlabel('Expected P-value', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Observed P-value', fontsize=11, fontweight='bold')
    ax3.set_title('C. QQ Plot (Uncorrected)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)

    # Panel D: Effect sizes (PLV - null)
    ax4 = fig.add_subplot(gs[1, 2])
    effect_size = plv_data['plv_empirical'] - plv_data['plv_null_mean']
    ax4.hist(effect_size, bins=40, edgecolor='black',
             color='coral', alpha=0.7)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='No effect')
    ax4.axvline(effect_size.mean(), color='darkred', linestyle='-',
                linewidth=2, label=f'Mean = {effect_size.mean():.3f}')
    ax4.set_xlabel('Effect Size (PLV - Null)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('D. Effect Size Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "05_pvalue_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def plot_top_findings(plv_data, save=True):
    """
    Plot 6: Top significant findings
    """
    # Get top 20 most significant findings
    top_findings = plv_data.nsmallest(20, 'p_fdr')

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create labels
    labels = [f"{row['subject']}_run{row['run']}\n{row['motion_param']}"
              for _, row in top_findings.iterrows()]

    y_pos = np.arange(len(labels))

    # Plot bars
    bars = ax.barh(y_pos, top_findings['plv_empirical'],
                   color='steelblue', alpha=0.7, label='Empirical PLV')
    ax.barh(y_pos, top_findings['plv_null_median'],
            color='gray', alpha=0.5, label='Null Median')

    # Add p-value annotations
    for i, (_, row) in enumerate(top_findings.iterrows()):
        ax.text(row['plv_empirical'] + 0.01, i,
                f"p={row['p_fdr']:.4f}",
                va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('PLV', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Significant Gastric-Motion Couplings (FDR-corrected)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    if save:
        output_path = OUTPUT_DIR / "06_top_findings.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig


def generate_summary_stats(plv_data, motion_data):
    """Generate and save summary statistics"""

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Overall stats
    print(f"\nTotal tests: {len(plv_data)}")
    print(f"  Subjects: {plv_data['subject'].nunique()}")
    print(f"  Subject-runs: {len(plv_data.groupby(['subject', 'run']))}")
    print(f"  Motion parameters: {plv_data['motion_param'].nunique()}")

    # PLV stats
    print(f"\nPLV Statistics:")
    print(f"  Empirical PLV: {plv_data['plv_empirical'].mean():.4f} ± {plv_data['plv_empirical'].std():.4f}")
    print(f"  Null PLV: {plv_data['plv_null_mean'].mean():.4f} ± {plv_data['plv_null_mean'].std():.4f}")
    print(f"  Effect size: {(plv_data['plv_empirical'] - plv_data['plv_null_mean']).mean():.4f}")

    # Significance
    print(f"\nSignificant findings:")
    print(f"  Uncorrected (p<0.05): {plv_data['sig_uncorrected'].sum()} ({100*plv_data['sig_uncorrected'].mean():.1f}%)")
    print(f"  FDR-corrected (q<0.05): {plv_data['sig_fdr'].sum()} ({100*plv_data['sig_fdr'].mean():.1f}%)")
    print(f"  Bonferroni (p<0.05): {plv_data['sig_bonferroni'].sum()} ({100*plv_data['sig_bonferroni'].mean():.1f}%)")

    # By motion parameter
    print(f"\nPLV by motion parameter (mean ± std):")
    for param in motion_params:
        subset = plv_data[plv_data['motion_param'] == param]
        print(f"  {MOTION_LABELS[param]:30s}: {subset['plv_empirical'].mean():.4f} ± {subset['plv_empirical'].std():.4f}")

    # Motion amplitude
    print(f"\nMotion amplitude (mean across subjects/runs):")
    for param in motion_params:
        col = f'{param}_mean_abs'
        if col in motion_data.columns:
            print(f"  {param:10s}: {motion_data[col].mean():.6f}")

    # Save to file
    summary_file = OUTPUT_DIR / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("GASTRIC-MOTION SYNCHRONY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total tests: {len(plv_data)}\n")
        f.write(f"Empirical PLV: {plv_data['plv_empirical'].mean():.4f} ± {plv_data['plv_empirical'].std():.4f}\n")
        f.write(f"Null PLV: {plv_data['plv_null_mean'].mean():.4f} ± {plv_data['plv_null_mean'].std():.4f}\n\n")
        f.write(f"Significant (FDR): {plv_data['sig_fdr'].sum()} / {len(plv_data)} ({100*plv_data['sig_fdr'].mean():.1f}%)\n\n")

        f.write("Top 10 findings:\n")
        top10 = plv_data.nsmallest(10, 'p_fdr')[['subject', 'run', 'motion_param', 'plv_empirical', 'p_fdr']]
        f.write(top10.to_string(index=False))

    print(f"\n✓ Summary statistics saved: {summary_file}")


# Motion parameters list
motion_params = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


def main():
    """Main visualization pipeline"""

    print("="*70)
    print("GASTRIC-MOTION SYNCHRONY VISUALIZATION (v3)")
    print("="*70)

    # Check if data files exist
    if not PLV_FILE.exists():
        print(f"\n✗ Error: PLV data file not found: {PLV_FILE}")
        print("  Please run egg_confounds_synchrony_v3.py first")
        return

    if not MOTION_FILE.exists():
        print(f"\n✗ Error: Motion summary file not found: {MOTION_FILE}")
        print("  Please run egg_confounds_synchrony_v3.py first")
        return

    # Load data
    plv_data, motion_data = load_data()

    print(f"\nGenerating visualizations in: {OUTPUT_DIR}")
    print("")

    # Generate all plots
    plot_plv_distributions(plv_data)
    plot_significance_summary(plv_data)
    plot_subject_heatmap(plv_data)
    plot_motion_amplitude_vs_plv(plv_data, motion_data)
    plot_pvalue_distributions(plv_data)
    plot_top_findings(plv_data)

    # Generate summary statistics
    generate_summary_stats(plv_data, motion_data)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll plots saved in: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  01_plv_distributions.png       - PLV empirical vs null")
    print("  02_significance_summary.png    - Significant findings by correction")
    print("  03_subject_heatmap.png         - Subject-wise PLV patterns")
    print("  04_motion_amplitude_vs_plv.png - Motion amplitude relationships")
    print("  05_pvalue_distributions.png    - P-value diagnostics")
    print("  06_top_findings.png            - Top significant couplings")
    print("  summary_statistics.txt         - Numerical summary")
    print("\nDone!")


if __name__ == "__main__":
    main()
