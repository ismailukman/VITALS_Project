#!/usr/bin/env python3
"""
ROI-Based EGG-Brain Synchrony Analysis

This script performs ROI-level synchrony analysis between EGG (gastric) and brain signals.
It uses an atlas to parcellate the brain into regions and computes phase locking value (PLV)
and amplitude-weighted PLV (awPLV) between the gastric signal and each ROI's mean time series.

Main Processing Steps:
1. Load preprocessed EGG data for each subject
2. Load preprocessed fMRI data (4D NIfTI)
3. Apply atlas to extract ROI time series
4. Filter brain signals at subject-specific gastric frequency
5. Compute PLV and awPLV between gastric and each ROI
6. Generate null distribution using mismatch approach
7. Perform statistical testing with FDR correction
8. Output results and visualizations

Usage:
    python roi_based_analysis.py
    python roi_based_analysis.py --metadata path/to/metadata.csv

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
import seaborn as sns

# Setup paths
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

from preprocess_egg_data.config import (
    sample_rate_fmri, intermediate_sample_rate, clean_level,
    main_project_path, brain_data_path, atlas_path,
    bandpass_lim, filter_order, transition_width
)

##############################################################################
# Configuration                                                              #
##############################################################################

# Paths
METADATA_PATH = project_root / "preprocess_egg_data" / "egg_brain_metadata.csv"
OUTPUT_DIR = project_root / "synchrony_analysis" / "output" / "roi_analysis"
PLOT_DIR = project_root / "synchrony_analysis" / "output" / "plots" / "roi_analysis"

# Atlas files
ATLAS_NII = pathlib.Path(atlas_path) / "atlas.nii"
ATLAS_LABELS = pathlib.Path(atlas_path) / "atlas.txt"

# Subjects with both EGG and brain data
SUBJECTS_WITH_BRAIN_DATA = ['VITD0107', 'VITD0126', 'VITD0128']

##############################################################################
# Helper Functions                                                           #
##############################################################################

def load_atlas(atlas_nii_path, atlas_labels_path):
    """
    Load atlas NIfTI and region labels.
    
    Parameters
    ----------
    atlas_nii_path : str or Path
        Path to atlas NIfTI file
    atlas_labels_path : str or Path
        Path to atlas labels text file
        
    Returns
    -------
    atlas_data : ndarray
        3D array with region indices
    atlas_affine : ndarray
        Affine transformation matrix
    roi_labels : list
        List of ROI names
    """
    atlas_img = nib.load(atlas_nii_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    atlas_affine = atlas_img.affine
    
    # Load labels
    with open(atlas_labels_path, 'r') as f:
        roi_labels = [line.strip() for line in f.readlines() if line.strip()]
    
    return atlas_data, atlas_affine, roi_labels


def extract_roi_timeseries(fmri_data, atlas_data):
    """
    Extract mean time series for each ROI from 4D fMRI data.
    
    Parameters
    ----------
    fmri_data : ndarray
        4D fMRI data (x, y, z, time)
    atlas_data : ndarray
        3D atlas parcellation
        
    Returns
    -------
    roi_timeseries : ndarray
        2D array (n_rois, n_timepoints)
    valid_rois : list
        List of valid ROI indices (with sufficient voxels)
    """
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois > 0]  # Exclude background
    
    n_timepoints = fmri_data.shape[3]
    roi_timeseries = []
    valid_rois = []
    
    for roi_idx in unique_rois:
        mask = atlas_data == roi_idx
        n_voxels = np.sum(mask)
        
        if n_voxels >= 10:  # Minimum voxels for reliable estimate
            # Extract voxels and compute mean
            roi_data = fmri_data[mask, :]
            mean_ts = np.mean(roi_data, axis=0)
            roi_timeseries.append(mean_ts)
            valid_rois.append(int(roi_idx))
    
    return np.array(roi_timeseries), valid_rois


def bandpass_filter_signal(signal_data, sample_rate, center_freq, bandpass_lim=0.015):
    """
    Apply bandpass filter around gastric frequency.
    
    Parameters
    ----------
    signal_data : ndarray
        1D or 2D signal array (n_signals, n_timepoints) or (n_timepoints,)
    sample_rate : float
        Sampling rate in Hz
    center_freq : float
        Center frequency for bandpass
    bandpass_lim : float
        Bandwidth (+/- this value from center)
        
    Returns
    -------
    filtered : ndarray
        Filtered signal(s)
    """
    l_freq = center_freq - bandpass_lim
    h_freq = center_freq + bandpass_lim
    
    # Ensure proper dimensions
    if signal_data.ndim == 1:
        signal_data = signal_data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    
    filter_length = int(filter_order * np.floor(sample_rate / l_freq))
    
    filtered = filter_data(
        signal_data,
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
    
    if squeeze:
        filtered = filtered.squeeze()
    
    return filtered


def calc_plv(phase_a, phase_b):
    """
    Compute Phase Locking Value between two phase time series.
    
    Parameters
    ----------
    phase_a : ndarray
        Phase of signal A (n_timepoints,) or (n_signals, n_timepoints)
    phase_b : ndarray
        Phase of signal B (n_timepoints,)
        
    Returns
    -------
    plv : float or ndarray
        Phase locking value(s)
    """
    if phase_a.ndim == 1:
        phase_diff = phase_a - phase_b
    else:
        phase_diff = phase_a - phase_b[np.newaxis, :]
    
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    return plv


def calc_awplv(signal_a, signal_b):
    """
    Compute Amplitude-Weighted Phase Locking Value.
    
    Parameters
    ----------
    signal_a : ndarray
        Signal A (n_timepoints,) or (n_signals, n_timepoints)
    signal_b : ndarray
        Signal B (n_timepoints,)
        
    Returns
    -------
    awplv : float or ndarray
        Amplitude-weighted PLV
    """
    # Get analytic signals
    if signal_a.ndim == 1:
        signal_a = signal_a[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False
    
    analytic_a = hilbert(signal_a, axis=-1)
    analytic_b = hilbert(signal_b)
    
    # Extract phase and amplitude
    phase_a = np.angle(analytic_a)
    phase_b = np.angle(analytic_b)
    amp_a = np.abs(analytic_a)
    amp_b = np.abs(analytic_b)
    
    # Phase difference
    phase_diff = phase_a - phase_b[np.newaxis, :]
    phase_diff_complex = np.exp(1j * phase_diff)
    
    # Amplitude weights
    amp_product = amp_a * amp_b[np.newaxis, :]
    sum_amp = np.sum(amp_product, axis=-1, keepdims=True)
    sum_amp[sum_amp == 0] = 1.0  # Avoid division by zero
    weights = amp_product / sum_amp
    
    # Compute awPLV
    awplv = np.abs(np.sum(weights * phase_diff_complex, axis=-1))
    
    if squeeze:
        awplv = awplv.squeeze()
    
    return awplv


def resample_gastric_to_fmri(gastric_signal, gastric_sr, fmri_sr, n_fmri_timepoints):
    """
    Resample gastric signal to match fMRI sampling rate.
    
    Parameters
    ----------
    gastric_signal : ndarray
        Gastric signal at intermediate sample rate
    gastric_sr : float
        Gastric signal sample rate (Hz)
    fmri_sr : float
        fMRI sample rate (Hz)
    n_fmri_timepoints : int
        Number of fMRI timepoints to match
        
    Returns
    -------
    resampled : ndarray
        Resampled gastric signal
    """
    gastric_time = np.arange(len(gastric_signal)) / gastric_sr
    fmri_time = np.arange(n_fmri_timepoints) / fmri_sr
    
    # Only interpolate within original time range
    valid_idx = fmri_time <= gastric_time[-1]
    
    interp_func = interp1d(gastric_time, gastric_signal, kind='cubic', fill_value='extrapolate')
    resampled = interp_func(fmri_time)
    
    return resampled, valid_idx


##############################################################################
# Main Analysis                                                              #
##############################################################################

def main(metadata_path=None):
    """
    Main ROI-based EGG-brain synchrony analysis.
    """
    print("="*70)
    print("ROI-BASED EGG-BRAIN SYNCHRONY ANALYSIS")
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
    
    # Load atlas
    print(f"\nLoading atlas from: {ATLAS_NII}")
    atlas_data, atlas_affine, roi_labels = load_atlas(ATLAS_NII, ATLAS_LABELS)
    n_atlas_rois = len(roi_labels)
    print(f"Atlas contains {n_atlas_rois} ROIs")
    
    # Storage for all subjects' data
    all_subjects_data = {}
    
    # STEP 1: Load and process data for each subject
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
        
        # Resample atlas to fMRI space if needed
        if atlas_data.shape != fmri_data.shape[:3]:
            from nilearn.image import resample_to_img
            atlas_img = nib.Nifti1Image(atlas_data.astype(np.float32), atlas_affine)
            atlas_resampled = resample_to_img(atlas_img, fmri_img, interpolation='nearest')
            atlas_data_subj = atlas_resampled.get_fdata().astype(int)
        else:
            atlas_data_subj = atlas_data
        
        # Extract ROI time series
        roi_timeseries, valid_rois = extract_roi_timeseries(fmri_data, atlas_data_subj)
        print(f"    Extracted {len(valid_rois)} valid ROIs")
        
        # Filter ROI time series at gastric frequency
        roi_timeseries_filtered = bandpass_filter_signal(
            roi_timeseries, sample_rate_fmri, gastric_freq, bandpass_lim
        )
        
        # Resample gastric signal to fMRI rate
        gastric_resampled, valid_idx = resample_gastric_to_fmri(
            gastric_signal, intermediate_sample_rate, sample_rate_fmri, n_timepoints
        )
        
        # Match lengths
        min_length = min(roi_timeseries_filtered.shape[1], len(gastric_resampled))
        roi_ts = roi_timeseries_filtered[:, :min_length]
        gastric_ts = gastric_resampled[:min_length]
        
        # Store data
        all_subjects_data[(subject, run)] = {
            'subject': subject,
            'run': run,
            'gastric_signal': gastric_ts,
            'gastric_freq': gastric_freq,
            'roi_timeseries': roi_ts,
            'valid_rois': valid_rois,
            'n_timepoints': min_length
        }
        
        print(f"    ✓ Loaded successfully ({min_length} timepoints)")
    
    if not all_subjects_data:
        print("\nNo valid data found. Please ensure EGG data is preprocessed for all subjects.")
        return
    
    print(f"\nSuccessfully loaded {len(all_subjects_data)} subject-run pairs")
    
    # STEP 2: Compute empirical PLV and awPLV for each subject
    print("\n" + "="*70)
    print("STEP 2: Computing empirical PLV and awPLV")
    print("="*70)
    
    empirical_results = {}
    
    for (subject, run), data in all_subjects_data.items():
        print(f"\n  {subject} run {run}:")
        
        roi_ts = data['roi_timeseries']
        gastric_ts = data['gastric_signal']
        
        # Get phases
        roi_analytic = hilbert(roi_ts, axis=-1)
        roi_phase = np.angle(roi_analytic)
        gastric_analytic = hilbert(gastric_ts)
        gastric_phase = np.angle(gastric_analytic)
        
        # Compute PLV for each ROI
        plv = calc_plv(roi_phase, gastric_phase)
        
        # Compute awPLV for each ROI
        awplv = calc_awplv(roi_ts, gastric_ts)
        
        empirical_results[(subject, run)] = {
            'plv': plv,
            'awplv': awplv,
            'roi_phase': roi_phase,
            'roi_timeseries': roi_ts,
            'gastric_phase': gastric_phase,
            'gastric_signal': gastric_ts
        }
        
        print(f"    PLV:   mean={np.mean(plv):.4f}, max={np.max(plv):.4f}")
        print(f"    awPLV: mean={np.mean(awplv):.4f}, max={np.max(awplv):.4f}")
    
    # STEP 3: Compute null distribution using mismatch approach
    print("\n" + "="*70)
    print("STEP 3: Computing null distribution (mismatch approach)")
    print("="*70)
    
    null_results = {}
    
    for (subject, run), data in all_subjects_data.items():
        print(f"\n  {subject} run {run}:")
        
        roi_phase = empirical_results[(subject, run)]['roi_phase']
        roi_ts = empirical_results[(subject, run)]['roi_timeseries']
        n_rois = roi_phase.shape[0]
        
        null_plvs = []
        null_awplvs = []
        
        # Pair with other subjects' gastric signals
        for (other_subject, other_run), other_data in all_subjects_data.items():
            if other_subject == subject:
                continue
            
            other_gastric = other_data['gastric_signal']
            other_gastric_phase = np.angle(hilbert(other_gastric))
            
            # Match lengths
            min_len = min(roi_phase.shape[1], len(other_gastric_phase))
            
            # Compute null PLV
            null_plv = calc_plv(roi_phase[:, :min_len], other_gastric_phase[:min_len])
            null_plvs.append(null_plv)
            
            # Compute null awPLV
            null_awplv = calc_awplv(roi_ts[:, :min_len], other_gastric[:min_len])
            null_awplvs.append(null_awplv)
        
        # Stack null values: (n_null_subjects, n_rois)
        null_plvs = np.array(null_plvs)
        null_awplvs = np.array(null_awplvs)
        
        # Compute p-values (proportion of null >= empirical)
        plv_empirical = empirical_results[(subject, run)]['plv']
        awplv_empirical = empirical_results[(subject, run)]['awplv']
        
        p_values_plv = np.mean(null_plvs >= plv_empirical[np.newaxis, :], axis=0)
        p_values_awplv = np.mean(null_awplvs >= awplv_empirical[np.newaxis, :], axis=0)
        
        # FDR correction across ROIs
        p_fdr_plv = false_discovery_control(p_values_plv, method='bh')
        p_fdr_awplv = false_discovery_control(p_values_awplv, method='bh')
        
        null_results[(subject, run)] = {
            'null_plvs': null_plvs,
            'null_awplvs': null_awplvs,
            'p_values_plv': p_values_plv,
            'p_values_awplv': p_values_awplv,
            'p_fdr_plv': p_fdr_plv,
            'p_fdr_awplv': p_fdr_awplv,
            'plv_null_mean': np.mean(null_plvs, axis=0),
            'awplv_null_mean': np.mean(null_awplvs, axis=0)
        }
        
        n_sig_plv = np.sum(p_fdr_plv < 0.05)
        n_sig_awplv = np.sum(p_fdr_awplv < 0.05)
        print(f"    Significant ROIs (FDR < 0.05): PLV={n_sig_plv}, awPLV={n_sig_awplv}")
    
    # STEP 4: Compile results
    print("\n" + "="*70)
    print("STEP 4: Compiling results")
    print("="*70)
    
    # Get common valid ROIs
    first_key = list(all_subjects_data.keys())[0]
    valid_rois = all_subjects_data[first_key]['valid_rois']
    n_rois = len(valid_rois)
    
    # Create results dataframe
    results_list = []
    
    for (subject, run), data in all_subjects_data.items():
        subj_valid_rois = data['valid_rois']
        emp = empirical_results[(subject, run)]
        null = null_results[(subject, run)]
        
        for i, roi_idx in enumerate(subj_valid_rois):
            roi_name = roi_labels[roi_idx - 1] if roi_idx <= len(roi_labels) else f"ROI_{roi_idx}"
            
            results_list.append({
                'subject': subject,
                'run': run,
                'roi_index': roi_idx,
                'roi_name': roi_name,
                'plv_empirical': emp['plv'][i],
                'plv_null_mean': null['plv_null_mean'][i],
                'plv_delta': emp['plv'][i] - null['plv_null_mean'][i],
                'p_value_plv': null['p_values_plv'][i],
                'p_fdr_plv': null['p_fdr_plv'][i],
                'sig_fdr_plv': null['p_fdr_plv'][i] < 0.05,
                'awplv_empirical': emp['awplv'][i],
                'awplv_null_mean': null['awplv_null_mean'][i],
                'awplv_delta': emp['awplv'][i] - null['awplv_null_mean'][i],
                'p_value_awplv': null['p_values_awplv'][i],
                'p_fdr_awplv': null['p_fdr_awplv'][i],
                'sig_fdr_awplv': null['p_fdr_awplv'][i] < 0.05,
                'gastric_freq': data['gastric_freq']
            })
    
    results_df = pd.DataFrame(results_list)
    
    # Save results
    output_csv = OUTPUT_DIR / "roi_synchrony_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n  ✓ Saved results to: {output_csv}")
    
    # STEP 5: Create summary statistics
    print("\n" + "="*70)
    print("STEP 5: Summary Statistics")
    print("="*70)
    
    # Group-level statistics per ROI
    roi_summary = results_df.groupby('roi_name').agg({
        'plv_empirical': ['mean', 'std'],
        'plv_delta': 'mean',
        'sig_fdr_plv': 'sum',
        'awplv_empirical': ['mean', 'std'],
        'awplv_delta': 'mean',
        'sig_fdr_awplv': 'sum'
    }).round(4)
    
    roi_summary.columns = ['_'.join(col).strip('_') for col in roi_summary.columns]
    roi_summary = roi_summary.reset_index()
    roi_summary = roi_summary.sort_values('plv_delta_mean', ascending=False)
    
    summary_csv = OUTPUT_DIR / "roi_summary_statistics.csv"
    roi_summary.to_csv(summary_csv, index=False)
    print(f"  ✓ Saved ROI summary to: {summary_csv}")
    
    # Print top ROIs
    print("\n  Top 10 ROIs by PLV effect (empirical - null):")
    print(roi_summary[['roi_name', 'plv_empirical_mean', 'plv_delta_mean', 'sig_fdr_plv_sum']].head(10).to_string())
    
    # STEP 6: Create visualizations
    print("\n" + "="*70)
    print("STEP 6: Creating visualizations")
    print("="*70)
    
    # Plot 1: ROI-level PLV heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # PLV heatmap
    pivot_plv = results_df.pivot_table(
        index='roi_name', columns='subject', values='plv_empirical', aggfunc='mean'
    )
    sns.heatmap(pivot_plv, ax=axes[0], cmap='viridis', center=None, 
                cbar_kws={'label': 'PLV'})
    axes[0].set_title('PLV by ROI and Subject', fontsize=14)
    axes[0].set_xlabel('Subject')
    axes[0].set_ylabel('ROI')
    
    # awPLV heatmap
    pivot_awplv = results_df.pivot_table(
        index='roi_name', columns='subject', values='awplv_empirical', aggfunc='mean'
    )
    sns.heatmap(pivot_awplv, ax=axes[1], cmap='viridis', center=None,
                cbar_kws={'label': 'awPLV'})
    axes[1].set_title('awPLV by ROI and Subject', fontsize=14)
    axes[1].set_xlabel('Subject')
    axes[1].set_ylabel('ROI')
    
    plt.tight_layout()
    heatmap_file = PLOT_DIR / "roi_synchrony_heatmap.png"
    plt.savefig(heatmap_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved heatmap to: {heatmap_file}")
    
    # Plot 2: Bar plot of significant ROIs
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLV significant count
    sig_counts_plv = results_df.groupby('roi_name')['sig_fdr_plv'].sum().sort_values(ascending=False)
    sig_counts_plv_top = sig_counts_plv.head(20)
    axes[0].barh(range(len(sig_counts_plv_top)), sig_counts_plv_top.values)
    axes[0].set_yticks(range(len(sig_counts_plv_top)))
    axes[0].set_yticklabels(sig_counts_plv_top.index)
    axes[0].set_xlabel('Number of Subjects with Significant PLV (FDR < 0.05)')
    axes[0].set_title('Top 20 ROIs by PLV Significance')
    axes[0].invert_yaxis()
    
    # awPLV significant count
    sig_counts_awplv = results_df.groupby('roi_name')['sig_fdr_awplv'].sum().sort_values(ascending=False)
    sig_counts_awplv_top = sig_counts_awplv.head(20)
    axes[1].barh(range(len(sig_counts_awplv_top)), sig_counts_awplv_top.values, color='coral')
    axes[1].set_yticks(range(len(sig_counts_awplv_top)))
    axes[1].set_yticklabels(sig_counts_awplv_top.index)
    axes[1].set_xlabel('Number of Subjects with Significant awPLV (FDR < 0.05)')
    axes[1].set_title('Top 20 ROIs by awPLV Significance')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    barplot_file = PLOT_DIR / "roi_significance_barplot.png"
    plt.savefig(barplot_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved bar plot to: {barplot_file}")
    
    # Plot 3: Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(results_df['plv_empirical'], bins=30, alpha=0.7, label='Empirical', color='blue')
    axes[0].axvline(results_df['plv_empirical'].mean(), color='blue', linestyle='--', label=f"Mean: {results_df['plv_empirical'].mean():.3f}")
    axes[0].set_xlabel('PLV')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of ROI-level PLV')
    axes[0].legend()
    
    axes[1].hist(results_df['awplv_empirical'], bins=30, alpha=0.7, label='Empirical', color='coral')
    axes[1].axvline(results_df['awplv_empirical'].mean(), color='coral', linestyle='--', label=f"Mean: {results_df['awplv_empirical'].mean():.3f}")
    axes[1].set_xlabel('awPLV')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of ROI-level awPLV')
    axes[1].legend()
    
    plt.tight_layout()
    dist_file = PLOT_DIR / "roi_distribution.png"
    plt.savefig(dist_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved distribution plot to: {dist_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(all_subjects_data)} subject-run pairs")
    print(f"Analyzed {n_rois} ROIs from atlas")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Plot directory: {PLOT_DIR}")
    print("\nFiles created:")
    print(f"  - {output_csv.name}")
    print(f"  - {summary_csv.name}")
    print(f"  - {heatmap_file.name}")
    print(f"  - {barplot_file.name}")
    print(f"  - {dist_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', '-m', help='Path to metadata CSV file')
    args = parser.parse_args()
    
    main(metadata_path=args.metadata)
