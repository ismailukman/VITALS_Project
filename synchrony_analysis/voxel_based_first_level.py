#!/usr/bin/env python3
"""
Voxel-Based EGG-Brain Synchrony Analysis - First Level

This script performs voxel-wise synchrony analysis between EGG (gastric) and brain BOLD signals.
It computes phase locking value (PLV) and amplitude-weighted PLV (awPLV) at each voxel.

Main Processing Steps:
1. Load preprocessed EGG data for each subject
2. Load preprocessed fMRI data (4D NIfTI)
3. Filter brain signals at subject-specific gastric frequency
4. Compute voxel-wise PLV and awPLV
5. Generate null distribution using mismatch approach
6. Perform statistical testing with FDR correction
7. Save individual-level NIfTI maps and statistics

Usage:
    python voxel_based_first_level.py
    python voxel_based_first_level.py --metadata path/to/metadata.csv

Author: EGG-Brain Synchrony Project
"""

import os
import sys
import pathlib
import argparse
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import false_discovery_control
from scipy.interpolate import interp1d
from mne.filter import filter_data
import matplotlib.pyplot as plt

# Setup paths
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from preprocess_egg_data.config import (
    sample_rate_fmri, intermediate_sample_rate, clean_level,
    main_project_path, brain_data_path,
    bandpass_lim, filter_order, transition_width
)

##############################################################################
# Configuration                                                              #
##############################################################################

# Paths
METADATA_PATH = project_root / "preprocess_egg_data" / "egg_brain_metadata.csv"
OUTPUT_DIR = project_root / "synchrony_analysis" / "output" / "voxel_analysis"
PLOT_DIR = project_root / "synchrony_analysis" / "output" / "plots" / "voxel_analysis"

# Subjects with both EGG and brain data
SUBJECTS_WITH_BRAIN_DATA = ['VITD0107', 'VITD0126', 'VITD0128']

# Processing parameters
MIN_VOXEL_INTENSITY = 100  # Minimum mean intensity to include voxel

##############################################################################
# Helper Functions                                                           #
##############################################################################

def create_brain_mask(fmri_data, min_intensity=MIN_VOXEL_INTENSITY):
    """
    Create a brain mask from 4D fMRI data.
    
    Parameters
    ----------
    fmri_data : ndarray
        4D fMRI data (x, y, z, time)
    min_intensity : float
        Minimum mean intensity to include voxel
        
    Returns
    -------
    mask : ndarray
        3D boolean mask
    """
    mean_data = np.mean(fmri_data, axis=3)
    mask = mean_data > min_intensity
    return mask


def bandpass_filter_voxels(voxel_data, sample_rate, center_freq, bandpass_lim=0.015):
    """
    Apply bandpass filter to voxel time series around gastric frequency.
    
    Parameters
    ----------
    voxel_data : ndarray
        2D array (n_voxels, n_timepoints)
    sample_rate : float
        Sampling rate in Hz
    center_freq : float
        Center frequency for bandpass
    bandpass_lim : float
        Bandwidth (+/- this value from center)
        
    Returns
    -------
    filtered : ndarray
        Filtered voxel data
    """
    l_freq = center_freq - bandpass_lim
    h_freq = center_freq + bandpass_lim
    
    filter_length = int(filter_order * np.floor(sample_rate / l_freq))
    
    filtered = filter_data(
        voxel_data,
        sfreq=sample_rate,
        l_freq=l_freq,
        h_freq=h_freq,
        filter_length=filter_length,
        l_trans_bandwidth=transition_width * l_freq,
        h_trans_bandwidth=transition_width * h_freq,
        n_jobs=1,
        method='fir',
        phase='zero-double',
        fir_window='hamming',
        fir_design='firwin2',
        verbose=False
    )
    
    return filtered


def calc_plv_voxelwise(brain_phase, gastric_phase):
    """
    Compute voxel-wise PLV between brain and gastric signals.
    
    Parameters
    ----------
    brain_phase : ndarray
        Brain phase (n_voxels, n_timepoints)
    gastric_phase : ndarray
        Gastric phase (n_timepoints,)
        
    Returns
    -------
    plv : ndarray
        PLV for each voxel (n_voxels,)
    """
    phase_diff = brain_phase - gastric_phase[np.newaxis, :]
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
    return plv


def calc_awplv_voxelwise(brain_signal, gastric_signal):
    """
    Compute voxel-wise amplitude-weighted PLV.
    
    Parameters
    ----------
    brain_signal : ndarray
        Brain signal (n_voxels, n_timepoints)
    gastric_signal : ndarray
        Gastric signal (n_timepoints,)
        
    Returns
    -------
    awplv : ndarray
        awPLV for each voxel (n_voxels,)
    """
    # Get analytic signals
    brain_analytic = hilbert(brain_signal, axis=1)
    gastric_analytic = hilbert(gastric_signal)
    
    # Extract phase and amplitude
    brain_phase = np.angle(brain_analytic)
    gastric_phase = np.angle(gastric_analytic)
    brain_amplitude = np.abs(brain_analytic)
    gastric_amplitude = np.abs(gastric_analytic)
    
    # Phase difference
    phase_diff = brain_phase - gastric_phase[np.newaxis, :]
    phase_diff_complex = np.exp(1j * phase_diff)
    
    # Amplitude weights
    amplitude_product = brain_amplitude * gastric_amplitude[np.newaxis, :]
    sum_amplitude = np.sum(amplitude_product, axis=1, keepdims=True)
    sum_amplitude[sum_amplitude == 0] = 1.0  # Avoid division by zero
    weights = amplitude_product / sum_amplitude
    
    # Compute awPLV
    awplv = np.abs(np.sum(weights * phase_diff_complex, axis=1))
    
    return awplv


def resample_gastric_to_fmri(gastric_signal, gastric_sr, fmri_sr, n_fmri_timepoints):
    """
    Resample gastric signal to match fMRI sampling rate.
    """
    gastric_time = np.arange(len(gastric_signal)) / gastric_sr
    fmri_time = np.arange(n_fmri_timepoints) / fmri_sr
    
    interp_func = interp1d(gastric_time, gastric_signal, kind='cubic', fill_value='extrapolate')
    resampled = interp_func(fmri_time)
    
    return resampled


def save_nifti_map(data, mask, affine, header, output_path):
    """
    Save voxel data as a NIfTI file.
    
    Parameters
    ----------
    data : ndarray
        1D array of voxel values (n_voxels,)
    mask : ndarray
        3D boolean mask
    affine : ndarray
        Affine transformation matrix
    header : nibabel header
        NIfTI header
    output_path : str or Path
        Output file path
    """
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = data
    img = nib.Nifti1Image(vol, affine=affine, header=header)
    nib.save(img, str(output_path))


##############################################################################
# Main Analysis                                                              #
##############################################################################

def main(metadata_path=None):
    """
    Main voxel-based EGG-brain synchrony analysis (first level).
    """
    print("="*70)
    print("VOXEL-BASED EGG-BRAIN SYNCHRONY ANALYSIS - FIRST LEVEL")
    print("="*70)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    if metadata_path is None:
        metadata_path = METADATA_PATH
    
    print(f"\nLoading metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    
    # Filter to subjects with brain data
    metadata_df = metadata_df[metadata_df['subject'].isin(SUBJECTS_WITH_BRAIN_DATA)]
    print(f"Subjects with both EGG and brain data: {list(metadata_df['subject'].unique())}")
    
    # Storage for all subjects' data
    all_subjects_data = {}
    
    # STEP 1: Load data for each subject
    print("\n" + "="*70)
    print("STEP 1: Loading data for all subjects")
    print("="*70)
    
    for _, row in metadata_df.iterrows():
        subject = row['subject']
        run = int(row['run'])
        
        print(f"\n  Processing {subject} run {run}...")
        
        # Load EGG data
        egg_derivatives_path = project_root / "preprocess_egg_data" / "output" / "derivatives"
        gastric_file = egg_derivatives_path / subject / f"{subject}{run}" / f"gast_data_{subject}_run{run}{clean_level}.npy"
        freq_file = egg_derivatives_path / subject / f"{subject}{run}" / f"max_freq{subject}_run{run}{clean_level}.npy"
        
        if not gastric_file.exists():
            print(f"    ✗ Missing EGG data: {gastric_file}")
            print(f"    → Need to run EGG preprocessing for {subject}")
            continue
        
        gastric_signal = np.load(gastric_file)
        gastric_freq = float(np.load(freq_file))
        print(f"    Gastric frequency: {gastric_freq:.4f} Hz ({gastric_freq*60:.2f} cpm)")
        
        # Load brain data
        brain_file = pathlib.Path(brain_data_path) / f"dswauRestingState_EGG_{subject}.nii"
        
        if not brain_file.exists():
            print(f"    ✗ Missing brain data: {brain_file}")
            continue
        
        fmri_img = nib.load(brain_file)
        fmri_data = fmri_img.get_fdata()
        n_timepoints = fmri_data.shape[3]
        print(f"    fMRI shape: {fmri_data.shape}")
        
        # Create mask
        mask = create_brain_mask(fmri_data)
        n_voxels = np.sum(mask)
        print(f"    Mask contains {n_voxels} voxels")
        
        # Extract voxel time series
        voxel_data = fmri_data[mask, :].astype(np.float64)
        
        # Filter voxels at gastric frequency
        print(f"    Filtering at {gastric_freq:.4f} Hz...")
        voxel_data_filtered = bandpass_filter_voxels(
            voxel_data, sample_rate_fmri, gastric_freq, bandpass_lim
        )
        
        # Resample gastric signal to fMRI rate
        gastric_resampled = resample_gastric_to_fmri(
            gastric_signal, intermediate_sample_rate, sample_rate_fmri, n_timepoints
        )
        
        # Match lengths
        min_length = min(voxel_data_filtered.shape[1], len(gastric_resampled))
        voxel_ts = voxel_data_filtered[:, :min_length]
        gastric_ts = gastric_resampled[:min_length]
        
        # Store data
        all_subjects_data[(subject, run)] = {
            'subject': subject,
            'run': run,
            'gastric_signal': gastric_ts,
            'gastric_freq': gastric_freq,
            'voxel_data': voxel_ts,
            'mask': mask,
            'affine': fmri_img.affine,
            'header': fmri_img.header,
            'shape': fmri_data.shape[:3],
            'n_voxels': n_voxels,
            'n_timepoints': min_length
        }
        
        print(f"    ✓ Loaded: {n_voxels} voxels, {min_length} timepoints")
    
    if not all_subjects_data:
        print("\nNo valid data found. Please ensure EGG data is preprocessed for all subjects.")
        return
    
    print(f"\nSuccessfully loaded {len(all_subjects_data)} subject-run pairs")
    
    # STEP 2: Compute empirical PLV and awPLV
    print("\n" + "="*70)
    print("STEP 2: Computing empirical PLV and awPLV")
    print("="*70)
    
    empirical_results = {}
    
    for (subject, run), data in all_subjects_data.items():
        print(f"\n  {subject} run {run}:")
        
        voxel_ts = data['voxel_data']
        gastric_ts = data['gastric_signal']
        
        # Get phases
        brain_analytic = hilbert(voxel_ts, axis=1)
        brain_phase = np.angle(brain_analytic)
        gastric_analytic = hilbert(gastric_ts)
        gastric_phase = np.angle(gastric_analytic)
        
        # Compute PLV
        plv = calc_plv_voxelwise(brain_phase, gastric_phase)
        
        # Compute awPLV
        awplv = calc_awplv_voxelwise(voxel_ts, gastric_ts)
        
        empirical_results[(subject, run)] = {
            'plv': plv,
            'awplv': awplv,
            'brain_phase': brain_phase,
            'voxel_data': voxel_ts
        }
        
        print(f"    PLV:   mean={np.mean(plv):.4f}, std={np.std(plv):.4f}")
        print(f"    awPLV: mean={np.mean(awplv):.4f}, std={np.std(awplv):.4f}")
    
    # STEP 3: Compute null distribution using mismatch approach
    print("\n" + "="*70)
    print("STEP 3: Computing null distribution (mismatch approach)")
    print("="*70)
    
    null_results = {}
    individual_stats = []
    
    for (subject, run), data in all_subjects_data.items():
        print(f"\n  {subject} run {run}:")
        
        brain_phase = empirical_results[(subject, run)]['brain_phase']
        voxel_ts = empirical_results[(subject, run)]['voxel_data']
        n_voxels = voxel_ts.shape[0]
        
        null_plvs = []
        null_awplvs = []
        
        # Pair with other subjects' gastric signals
        for (other_subject, other_run), other_data in all_subjects_data.items():
            if other_subject == subject:
                continue
            
            other_gastric = other_data['gastric_signal']
            other_gastric_phase = np.angle(hilbert(other_gastric))
            
            # Match lengths
            min_len = min(brain_phase.shape[1], len(other_gastric_phase))
            
            # Compute null PLV
            null_plv = calc_plv_voxelwise(brain_phase[:, :min_len], other_gastric_phase[:min_len])
            null_plvs.append(null_plv)
            
            # Compute null awPLV
            null_awplv = calc_awplv_voxelwise(voxel_ts[:, :min_len], other_gastric[:min_len])
            null_awplvs.append(null_awplv)
        
        # Stack: (n_null_subjects, n_voxels)
        null_plvs = np.array(null_plvs)
        null_awplvs = np.array(null_awplvs)
        
        # Get empirical values
        plv_empirical = empirical_results[(subject, run)]['plv']
        awplv_empirical = empirical_results[(subject, run)]['awplv']
        
        # Compute p-values
        p_values_plv = np.mean(null_plvs >= plv_empirical[np.newaxis, :], axis=0)
        p_values_awplv = np.mean(null_awplvs >= awplv_empirical[np.newaxis, :], axis=0)
        
        # Handle edge cases
        p_values_plv[p_values_plv == 0] = 1.0 / (null_plvs.shape[0] + 1)
        p_values_awplv[p_values_awplv == 0] = 1.0 / (null_awplvs.shape[0] + 1)
        
        # FDR correction
        p_fdr_plv = false_discovery_control(p_values_plv, method='bh')
        p_fdr_awplv = false_discovery_control(p_values_awplv, method='bh')
        
        # Null statistics
        plv_null_mean = np.mean(null_plvs, axis=0)
        plv_null_std = np.std(null_plvs, axis=0)
        awplv_null_mean = np.mean(null_awplvs, axis=0)
        awplv_null_std = np.std(null_awplvs, axis=0)
        
        null_results[(subject, run)] = {
            'p_values_plv': p_values_plv,
            'p_values_awplv': p_values_awplv,
            'p_fdr_plv': p_fdr_plv,
            'p_fdr_awplv': p_fdr_awplv,
            'plv_null_mean': plv_null_mean,
            'plv_null_std': plv_null_std,
            'awplv_null_mean': awplv_null_mean,
            'awplv_null_std': awplv_null_std,
            'n_null': null_plvs.shape[0]
        }
        
        n_sig_plv_uncorr = np.sum(p_values_plv < 0.05)
        n_sig_plv_fdr = np.sum(p_fdr_plv < 0.05)
        n_sig_awplv_uncorr = np.sum(p_values_awplv < 0.05)
        n_sig_awplv_fdr = np.sum(p_fdr_awplv < 0.05)
        
        print(f"    PLV:   {n_sig_plv_uncorr} uncorrected, {n_sig_plv_fdr} FDR significant")
        print(f"    awPLV: {n_sig_awplv_uncorr} uncorrected, {n_sig_awplv_fdr} FDR significant")
        
        # Store individual stats
        individual_stats.append({
            'subject': subject,
            'run': run,
            'n_voxels': n_voxels,
            'n_timepoints': data['n_timepoints'],
            'gastric_freq': data['gastric_freq'],
            'plv_mean': np.mean(plv_empirical),
            'plv_std': np.std(plv_empirical),
            'plv_max': np.max(plv_empirical),
            'plv_null_mean': np.mean(plv_null_mean),
            'n_sig_plv_uncorr': n_sig_plv_uncorr,
            'n_sig_plv_fdr': n_sig_plv_fdr,
            'pct_sig_plv_fdr': 100 * n_sig_plv_fdr / n_voxels,
            'awplv_mean': np.mean(awplv_empirical),
            'awplv_std': np.std(awplv_empirical),
            'awplv_max': np.max(awplv_empirical),
            'awplv_null_mean': np.mean(awplv_null_mean),
            'n_sig_awplv_uncorr': n_sig_awplv_uncorr,
            'n_sig_awplv_fdr': n_sig_awplv_fdr,
            'pct_sig_awplv_fdr': 100 * n_sig_awplv_fdr / n_voxels
        })
    
    # Save individual statistics
    stats_df = pd.DataFrame(individual_stats)
    stats_file = OUTPUT_DIR / "individual_level_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n  ✓ Saved statistics to: {stats_file}")
    
    # STEP 4: Save subject-level NIfTI maps
    print("\n" + "="*70)
    print("STEP 4: Saving subject-level NIfTI maps")
    print("="*70)
    
    for (subject, run), data in all_subjects_data.items():
        subject_dir = OUTPUT_DIR / subject / f"{subject}{run}"
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        mask = data['mask']
        affine = data['affine']
        header = data['header']
        
        emp = empirical_results[(subject, run)]
        null = null_results[(subject, run)]
        
        # Maps to save
        maps = {
            'plv_empirical': emp['plv'],
            'awplv_empirical': emp['awplv'],
            'plv_delta': emp['plv'] - null['plv_null_mean'],
            'awplv_delta': emp['awplv'] - null['awplv_null_mean'],
            'plv_zscore': (emp['plv'] - null['plv_null_mean']) / (null['plv_null_std'] + 1e-10),
            'awplv_zscore': (emp['awplv'] - null['awplv_null_mean']) / (null['awplv_null_std'] + 1e-10),
            'plv_p_uncorr': null['p_values_plv'],
            'awplv_p_uncorr': null['p_values_awplv'],
            'plv_p_fdr': null['p_fdr_plv'],
            'awplv_p_fdr': null['p_fdr_awplv'],
            'plv_sig_fdr': (null['p_fdr_plv'] < 0.05).astype(np.float32),
            'awplv_sig_fdr': (null['p_fdr_awplv'] < 0.05).astype(np.float32)
        }
        
        for map_name, map_data in maps.items():
            output_path = subject_dir / f"{map_name}_{subject}_run{run}{clean_level}.nii.gz"
            save_nifti_map(map_data, mask, affine, header, output_path)
        
        print(f"  ✓ Saved {len(maps)} maps for {subject} run {run}")
    
    # STEP 5: Create summary visualizations
    print("\n" + "="*70)
    print("STEP 5: Creating summary visualizations")
    print("="*70)
    
    # Plot individual-level statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    subjects = stats_df['subject'].tolist()
    x_pos = range(len(subjects))
    
    # PLV mean
    axes[0, 0].bar(x_pos, stats_df['plv_mean'], color='steelblue', alpha=0.8)
    axes[0, 0].errorbar(x_pos, stats_df['plv_mean'], yerr=stats_df['plv_std'], 
                        fmt='none', color='black', capsize=3)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(subjects)
    axes[0, 0].set_ylabel('Mean PLV')
    axes[0, 0].set_title('PLV per Subject')
    axes[0, 0].grid(alpha=0.3)
    
    # awPLV mean
    axes[0, 1].bar(x_pos, stats_df['awplv_mean'], color='coral', alpha=0.8)
    axes[0, 1].errorbar(x_pos, stats_df['awplv_mean'], yerr=stats_df['awplv_std'],
                        fmt='none', color='black', capsize=3)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(subjects)
    axes[0, 1].set_ylabel('Mean awPLV')
    axes[0, 1].set_title('awPLV per Subject')
    axes[0, 1].grid(alpha=0.3)
    
    # Significant voxels PLV
    axes[1, 0].bar(x_pos, stats_df['pct_sig_plv_fdr'], color='steelblue', alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(subjects)
    axes[1, 0].set_ylabel('% Significant Voxels (FDR < 0.05)')
    axes[1, 0].set_title('PLV Significant Voxels')
    axes[1, 0].grid(alpha=0.3)
    
    # Significant voxels awPLV
    axes[1, 1].bar(x_pos, stats_df['pct_sig_awplv_fdr'], color='coral', alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(subjects)
    axes[1, 1].set_ylabel('% Significant Voxels (FDR < 0.05)')
    axes[1, 1].set_title('awPLV Significant Voxels')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    summary_plot = PLOT_DIR / "first_level_summary.png"
    plt.savefig(summary_plot, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved summary plot: {summary_plot}")
    
    # Final summary
    print("\n" + "="*70)
    print("FIRST LEVEL ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(all_subjects_data)} subjects")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nIndividual-level summary:")
    print(stats_df[['subject', 'plv_mean', 'pct_sig_plv_fdr', 'awplv_mean', 'pct_sig_awplv_fdr']].to_string())
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', '-m', help='Path to metadata CSV file')
    args = parser.parse_args()
    
    main(metadata_path=args.metadata)
