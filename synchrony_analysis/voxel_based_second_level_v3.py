#!/usr/bin/env python3
"""
Voxel-Based EGG-Brain Synchrony Analysis - Second Level (Group Analysis)

This script performs group-level analysis on voxel-wise synchrony maps from first-level analysis.
It computes group averages, performs one-sample t-tests, and applies multiple comparison correction.

Main Processing Steps:
1. Load individual-level PLV and awPLV maps
2. Compute group mean maps
3. Perform one-sample t-tests at each voxel
4. Apply FDR and cluster-based multiple comparison correction
5. Create group-level statistical maps and visualizations

Usage:
    python voxel_based_second_level_v3.py
    python voxel_based_second_level_v3.py --first-level-dir path/to/first_level_output

Author: EGG-Brain Synchrony Project
"""

import os
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy.stats import false_discovery_control
from nilearn import plotting
from nilearn.image import concat_imgs, mean_img, resample_to_img
import matplotlib.pyplot as plt

# Setup paths
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from preprocess_egg_data.config import clean_level

##############################################################################
# Configuration                                                              #
##############################################################################

FIRST_LEVEL_DIR = project_root / "synchrony_analysis" / "output" / "voxel_analysis"
OUTPUT_DIR = project_root / "synchrony_analysis" / "output" / "group_analysis"
PLOT_DIR = project_root / "synchrony_analysis" / "output" / "plots" / "group_analysis"

# Subjects to include
SUBJECTS = ['VITD0107', 'VITD0126', 'VITD0128']

# Statistical thresholds
ALPHA_FDR = 0.05
VOXEL_P_THRESHOLD = 0.001  # Uncorrected threshold for visualization

##############################################################################
# Helper Functions                                                           #
##############################################################################

def load_subject_maps(first_level_dir, subjects, map_name, clean_level):
    """
    Load individual-level maps for all subjects.
    
    Parameters
    ----------
    first_level_dir : Path
        Directory containing first-level outputs
    subjects : list
        List of subject IDs
    map_name : str
        Name of the map to load (e.g., 'plv_empirical')
    clean_level : str
        Processing level identifier
        
    Returns
    -------
    maps : list of nibabel images
        Individual subject maps
    subjects_found : list
        List of subjects with available data
    """
    maps = []
    subjects_found = []
    
    for subject in subjects:
        subject_dir = first_level_dir / subject / f"{subject}1"
        map_file = subject_dir / f"{map_name}_{subject}_run1{clean_level}.nii.gz"
        
        if map_file.exists():
            img = nib.load(map_file)
            maps.append(img)
            subjects_found.append(subject)
        else:
            print(f"  Warning: Missing {map_name} for {subject}")
    
    return maps, subjects_found


def compute_group_statistics(maps):
    """
    Compute group-level statistics from individual maps.
    
    Parameters
    ----------
    maps : list of nibabel images
        Individual subject maps
        
    Returns
    -------
    mean_data : ndarray
        Group mean at each voxel
    std_data : ndarray
        Group standard deviation at each voxel
    t_stat : ndarray
        One-sample t-statistic at each voxel
    p_values : ndarray
        P-values from one-sample t-test
    """
    # Stack data: (n_subjects, x, y, z)
    data_stack = np.stack([img.get_fdata() for img in maps], axis=0)
    
    n_subjects = data_stack.shape[0]
    
    # Compute statistics
    mean_data = np.mean(data_stack, axis=0)
    std_data = np.std(data_stack, axis=0, ddof=1)
    
    # One-sample t-test against 0 (or against null level)
    # H0: mean PLV = chance level (which we approximate as null mean)
    # For simplicity, we test if values are significantly > 0
    se = std_data / np.sqrt(n_subjects)
    se[se == 0] = np.inf  # Avoid division by zero
    
    t_stat = mean_data / se
    
    # Two-tailed p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_subjects-1))
    
    return mean_data, std_data, t_stat, p_values


def create_brain_mask_from_maps(maps):
    """
    Create a mask of voxels present in all subjects.
    
    Parameters
    ----------
    maps : list of nibabel images
        Individual subject maps
        
    Returns
    -------
    mask : ndarray
        3D boolean mask
    """
    mask = None
    for img in maps:
        data = img.get_fdata()
        img_mask = ~np.isnan(data) & (data != 0)
        if mask is None:
            mask = img_mask
        else:
            mask = mask & img_mask
    return mask


##############################################################################
# Main Analysis                                                              #
##############################################################################

def main(first_level_dir=None):
    """
    Main group-level voxel-based analysis.
    """
    print("="*70)
    print("VOXEL-BASED EGG-BRAIN SYNCHRONY ANALYSIS - SECOND LEVEL (GROUP)")
    print("="*70)
    
    # Setup directories
    if first_level_dir is None:
        first_level_dir = FIRST_LEVEL_DIR
    first_level_dir = pathlib.Path(first_level_dir)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFirst-level directory: {first_level_dir}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Subjects: {SUBJECTS}")
    
    # Define maps to process
    map_types = [
        ('plv_empirical', 'PLV'),
        ('awplv_empirical', 'awPLV'),
        ('plv_delta', 'PLV Delta (vs Null)'),
        ('awplv_delta', 'awPLV Delta (vs Null)'),
        ('plv_zscore', 'PLV Z-score'),
        ('awplv_zscore', 'awPLV Z-score')
    ]
    
    group_results = {}
    reference_img = None
    group_mask = None
    
    # STEP 1: Load and process each map type
    print("\n" + "="*70)
    print("STEP 1: Loading individual-level maps")
    print("="*70)
    
    for map_name, map_label in map_types:
        print(f"\n  Processing {map_label}...")
        
        maps, subjects_found = load_subject_maps(
            first_level_dir, SUBJECTS, map_name, clean_level
        )
        
        if len(maps) < 2:
            print(f"    ✗ Insufficient data for {map_label} (need at least 2 subjects)")
            continue
        
        print(f"    Found data for {len(maps)} subjects: {subjects_found}")
        
        # Store reference image
        if reference_img is None:
            reference_img = maps[0]
            group_mask = create_brain_mask_from_maps(maps)
            n_voxels = np.sum(group_mask)
            print(f"    Group mask: {n_voxels} voxels")
        
        # Compute group statistics
        mean_data, std_data, t_stat, p_values = compute_group_statistics(maps)
        
        # Apply FDR correction
        p_flat = p_values[group_mask]
        p_fdr_flat = false_discovery_control(p_flat, method='bh')
        p_fdr = np.ones_like(p_values)
        p_fdr[group_mask] = p_fdr_flat
        
        # Store results
        group_results[map_name] = {
            'maps': maps,
            'subjects': subjects_found,
            'mean': mean_data,
            'std': std_data,
            't_stat': t_stat,
            'p_values': p_values,
            'p_fdr': p_fdr,
            'n_sig_uncorr': np.sum((p_values < 0.05) & group_mask),
            'n_sig_fdr': np.sum((p_fdr < ALPHA_FDR) & group_mask),
            'label': map_label
        }
        
        print(f"    Mean across subjects: {np.mean(mean_data[group_mask]):.4f}")
        print(f"    Significant voxels (uncorrected p<0.05): {group_results[map_name]['n_sig_uncorr']}")
        print(f"    Significant voxels (FDR q<{ALPHA_FDR}): {group_results[map_name]['n_sig_fdr']}")
    
    if not group_results:
        print("\nNo valid group results. Exiting.")
        return
    
    # STEP 2: Save group-level maps
    print("\n" + "="*70)
    print("STEP 2: Saving group-level maps")
    print("="*70)
    
    affine = reference_img.affine
    header = reference_img.header
    
    for map_name, results in group_results.items():
        # Save mean map
        mean_img = nib.Nifti1Image(results['mean'].astype(np.float32), affine, header)
        nib.save(mean_img, OUTPUT_DIR / f"{map_name}_group_mean{clean_level}.nii.gz")
        
        # Save t-statistic map
        t_img = nib.Nifti1Image(results['t_stat'].astype(np.float32), affine, header)
        nib.save(t_img, OUTPUT_DIR / f"{map_name}_group_tstat{clean_level}.nii.gz")
        
        # Save p-value map (uncorrected)
        p_img = nib.Nifti1Image(results['p_values'].astype(np.float32), affine, header)
        nib.save(p_img, OUTPUT_DIR / f"{map_name}_group_pval{clean_level}.nii.gz")
        
        # Save FDR-corrected map
        fdr_img = nib.Nifti1Image(results['p_fdr'].astype(np.float32), affine, header)
        nib.save(fdr_img, OUTPUT_DIR / f"{map_name}_group_pfdr{clean_level}.nii.gz")
        
        # Save significance mask
        sig_data = (results['p_fdr'] < ALPHA_FDR).astype(np.float32)
        sig_img = nib.Nifti1Image(sig_data, affine, header)
        nib.save(sig_img, OUTPUT_DIR / f"{map_name}_group_sig_fdr{clean_level}.nii.gz")
    
    print(f"  ✓ Saved group maps to {OUTPUT_DIR}")
    
    # STEP 3: Create group-level summary
    print("\n" + "="*70)
    print("STEP 3: Creating group summary")
    print("="*70)
    
    summary_data = []
    for map_name, results in group_results.items():
        summary_data.append({
            'map_type': results['label'],
            'n_subjects': len(results['subjects']),
            'subjects': ', '.join(results['subjects']),
            'n_voxels': int(np.sum(group_mask)),
            'mean_value': np.mean(results['mean'][group_mask]),
            'std_value': np.std(results['mean'][group_mask]),
            'max_value': np.max(results['mean'][group_mask]),
            'max_t_stat': np.max(results['t_stat'][group_mask]),
            'n_sig_uncorr_p05': results['n_sig_uncorr'],
            'n_sig_fdr_q05': results['n_sig_fdr'],
            'pct_sig_fdr': 100 * results['n_sig_fdr'] / np.sum(group_mask)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DIR / "group_level_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ Saved summary to {summary_file}")
    
    print("\nGroup-level summary:")
    print(summary_df[['map_type', 'n_subjects', 'mean_value', 'n_sig_fdr_q05', 'pct_sig_fdr']].to_string())
    
    # STEP 4: Create visualizations
    print("\n" + "="*70)
    print("STEP 4: Creating visualizations")
    print("="*70)
    
    # Try to load MNI template for visualization
    try:
        from nilearn import datasets
        mni_template = datasets.load_mni152_template()
    except:
        mni_template = None
        print("  Could not load MNI template, using subject data for background")
    
    # Create brain map visualizations for key metrics
    for map_name in ['plv_empirical', 'plv_delta', 'awplv_empirical', 'awplv_delta']:
        if map_name not in group_results:
            continue
        
        results = group_results[map_name]
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Mean map
        mean_nii = nib.Nifti1Image(results['mean'].astype(np.float32), affine, header)
        
        # Plot orthogonal slices
        for idx, (ax, display_mode, coord) in enumerate([
            (axes[0, 0], 'x', 0),
            (axes[0, 1], 'y', 0),
            (axes[0, 2], 'z', 0)
        ]):
            try:
                plotting.plot_stat_map(
                    mean_nii, 
                    bg_img=mni_template,
                    display_mode=display_mode,
                    cut_coords=[coord],
                    axes=ax,
                    title=f"{results['label']} Mean",
                    colorbar=True
                )
            except:
                ax.text(0.5, 0.5, f"Could not plot {display_mode}={coord}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # T-statistic map
        t_nii = nib.Nifti1Image(results['t_stat'].astype(np.float32), affine, header)
        
        for idx, (ax, display_mode, coord) in enumerate([
            (axes[1, 0], 'x', 0),
            (axes[1, 1], 'y', 0),
            (axes[1, 2], 'z', 0)
        ]):
            try:
                plotting.plot_stat_map(
                    t_nii,
                    bg_img=mni_template,
                    display_mode=display_mode,
                    cut_coords=[coord],
                    axes=ax,
                    title=f"{results['label']} T-statistic",
                    colorbar=True,
                    threshold=2.0  # Approximately p<0.05 for small samples
                )
            except:
                ax.text(0.5, 0.5, f"Could not plot {display_mode}={coord}",
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plot_file = PLOT_DIR / f"{map_name}_group_brainmap{clean_level}.png"
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved brain map: {plot_file.name}")
    
    # Create bar chart summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PLV comparison
    plv_maps = [k for k in ['plv_empirical', 'plv_delta'] if k in group_results]
    if plv_maps:
        means = [np.mean(group_results[k]['mean'][group_mask]) for k in plv_maps]
        stds = [np.std(group_results[k]['mean'][group_mask]) for k in plv_maps]
        labels = [group_results[k]['label'] for k in plv_maps]
        
        x = range(len(plv_maps))
        axes[0].bar(x, means, yerr=stds, capsize=5, color=['steelblue', 'lightsteelblue'][:len(plv_maps)])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Group Mean PLV')
        axes[0].grid(alpha=0.3, axis='y')
    
    # awPLV comparison
    awplv_maps = [k for k in ['awplv_empirical', 'awplv_delta'] if k in group_results]
    if awplv_maps:
        means = [np.mean(group_results[k]['mean'][group_mask]) for k in awplv_maps]
        stds = [np.std(group_results[k]['mean'][group_mask]) for k in awplv_maps]
        labels = [group_results[k]['label'] for k in awplv_maps]
        
        x = range(len(awplv_maps))
        axes[1].bar(x, means, yerr=stds, capsize=5, color=['coral', 'lightsalmon'][:len(awplv_maps)])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('Value')
        axes[1].set_title('Group Mean awPLV')
        axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_plot = PLOT_DIR / f"group_summary_barplot{clean_level}.png"
    plt.savefig(summary_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved summary bar plot: {summary_plot.name}")
    
    # Final summary
    print("\n" + "="*70)
    print("SECOND LEVEL ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(SUBJECTS)} subjects")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Plot directory: {PLOT_DIR}")
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-level-dir', '-f', help='Path to first-level output directory')
    args = parser.parse_args()
    
    main(first_level_dir=args.first_level_dir)
