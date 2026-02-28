#!/usr/bin/env python3
"""
Network-Level EGG-Brain Synchrony Analysis

This script computes synchrony between the gastric (EGG) rhythm and brain signals
at the NETWORK level, using two complementary approaches:

  Approach 1 — CONN ICA-based networks (networks.nii, 32 ROIs across 8 networks)
    Uses the 4D networks NIfTI where each volume is a probabilistic ROI map
    derived from ICA analysis of the HCP dataset (497 subjects).
    Networks: DefaultMode, SensoriMotor, Visual, Salience, DorsalAttention,
              FrontoParietal, Language, Cerebellar

  Approach 2 — Atlas-derived network groupings (atlas.groups.info, 22 networks)
    Uses the ROI-level results from roi_based_analysis.py and groups them into
    the 22 anatomical/functional networks defined in atlas.groups.info.
    This leverages the already-computed per-ROI PLV/awPLV values.

Both produce:
  - Network-level PLV and awPLV (mean of constituent ROIs/voxels)
  - Empirical vs null (mismatch) comparison
  - FDR-corrected significance
  - v5-style density plots per network
  - Summary bar charts and tables

Usage:
    python network_synchrony.py

Author: EGG-Brain Synchrony Project
"""

import os
import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import hilbert
from scipy.stats import false_discovery_control, mannwhitneyu
from scipy.interpolate import interp1d
from mne.filter import filter_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

METADATA_PATH = project_root / "preprocess_egg_data" / "egg_brain_metadata.csv"
OUTPUT_DIR = project_root / "network_analysis" / "output"
PLOT_DIR = project_root / "network_analysis" / "output" / "plots"

# Atlas files
ATLAS_NII = pathlib.Path(atlas_path) / "atlas.nii"
ATLAS_LABELS = pathlib.Path(atlas_path) / "atlas.txt"
ATLAS_GROUPS = pathlib.Path(atlas_path) / "atlas.groups.info"

# Network files (CONN ICA-based)
NETWORKS_NII = pathlib.Path(atlas_path) / "networks.nii"
NETWORKS_LABELS = pathlib.Path(atlas_path) / "networks.txt"

# Subjects with both EGG and brain data
SUBJECTS_WITH_BRAIN_DATA = ['VITD0107', 'VITD0126', 'VITD0128']

# Network name mapping for the CONN networks (8 networks, 32 ROIs total)
CONN_NETWORK_NAMES = [
    'Default Mode', 'Default Mode', 'Default Mode', 'Default Mode',
    'SensoriMotor', 'SensoriMotor', 'SensoriMotor',
    'Visual', 'Visual', 'Visual', 'Visual',
    'Salience', 'Salience', 'Salience', 'Salience', 'Salience', 'Salience', 'Salience',
    'Dorsal Attention', 'Dorsal Attention', 'Dorsal Attention', 'Dorsal Attention',
    'FrontoParietal', 'FrontoParietal', 'FrontoParietal', 'FrontoParietal',
    'Language', 'Language', 'Language', 'Language',
    'Cerebellar', 'Cerebellar'
]


##############################################################################
# Shared Helper Functions                                                    #
##############################################################################

def bandpass_filter_signal(signal_data, sample_rate, center_freq, bw=None):
    """Bandpass filter around gastric frequency."""
    if bw is None:
        bw = bandpass_lim
    l_freq = center_freq - bw
    h_freq = center_freq + bw

    if signal_data.ndim == 1:
        signal_data = signal_data[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    fl = int(filter_order * np.floor(sample_rate / l_freq))

    filtered = filter_data(
        signal_data, sfreq=sample_rate,
        l_freq=l_freq, h_freq=h_freq,
        filter_length=fl,
        l_trans_bandwidth=transition_width * l_freq,
        h_trans_bandwidth=transition_width * h_freq,
        n_jobs=1, method='fir', phase='zero-double',
        fir_window='hamming', fir_design='firwin2', verbose=False
    )
    return filtered.squeeze() if squeeze else filtered


def calc_plv(phase_a, phase_b):
    """Phase Locking Value. phase_a: (n, T) or (T,), phase_b: (T,)."""
    if phase_a.ndim == 1:
        diff = phase_a - phase_b
    else:
        diff = phase_a - phase_b[np.newaxis, :]
    return np.abs(np.mean(np.exp(1j * diff), axis=-1))


def calc_awplv(signal_a, signal_b):
    """Amplitude-weighted PLV. signal_a: (n, T) or (T,), signal_b: (T,)."""
    if signal_a.ndim == 1:
        signal_a = signal_a[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    analytic_a = hilbert(signal_a, axis=-1)
    analytic_b = hilbert(signal_b)
    pa, pb = np.angle(analytic_a), np.angle(analytic_b)
    aa, ab = np.abs(analytic_a), np.abs(analytic_b)

    diff = np.exp(1j * (pa - pb[np.newaxis, :]))
    amp = aa * ab[np.newaxis, :]
    s = np.sum(amp, axis=-1, keepdims=True)
    s[s == 0] = 1.0
    w = amp / s
    awplv = np.abs(np.sum(w * diff, axis=-1))
    return awplv.squeeze() if squeeze else awplv


def resample_gastric_to_fmri(gastric_signal, gastric_sr, fmri_sr, n_fmri_tp):
    """Resample gastric signal to fMRI sampling rate."""
    gt = np.arange(len(gastric_signal)) / gastric_sr
    ft = np.arange(n_fmri_tp) / fmri_sr
    f = interp1d(gt, gastric_signal, kind='cubic', fill_value='extrapolate')
    return f(ft)


def _sig_marker(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


##############################################################################
# Parse atlas.groups.info → ROI-to-network mapping                          #
##############################################################################

def parse_atlas_groups(groups_file, labels_file):
    """
    Parse atlas.groups.info to create a mapping from ROI name → network name.
    Also loads atlas labels from atlas.txt.
    
    Returns
    -------
    roi_labels : list of str
        ROI label names (132 entries)
    roi_to_network : dict
        Mapping from ROI label → network name
    network_rois : dict
        Mapping from network name → list of ROI labels
    """
    # Load ROI labels
    with open(labels_file, 'r') as f:
        roi_labels = [line.strip() for line in f if line.strip()]

    # Parse groups file
    roi_to_network = {}
    network_rois = {}
    current_network = None

    with open(groups_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip header lines
            if line_stripped.startswith('ROI order') or line_stripped.startswith('These clusters'):
                continue
            if line_stripped.startswith('hierarchical') or line_stripped.startswith('et al'):
                continue
            if line_stripped.startswith('applied to') or line_stripped.startswith('dataset'):
                continue
            if 'networks / ' in line_stripped and 'ROIs' in line_stripped:
                continue

            # Separator line
            if line_stripped.startswith('---'):
                continue

            # Check if this is a network header (doesn't start with 'atlas.')
            if not line_stripped.startswith('atlas.'):
                current_network = line_stripped
                if current_network not in network_rois:
                    network_rois[current_network] = []
                continue

            # ROI entry: extract the label after 'atlas.'
            if current_network is not None:
                # Strip 'atlas.' prefix and any trailing whitespace
                roi_entry = line_stripped[6:].strip()  # Remove 'atlas.'

                # Match against the roi_labels list
                matched = False
                for label in roi_labels:
                    if roi_entry == label or label.startswith(roi_entry.split(' (')[0]):
                        roi_to_network[label] = current_network
                        network_rois[current_network].append(label)
                        matched = True
                        break

                if not matched:
                    # Try partial match
                    for label in roi_labels:
                        # Compare first word
                        if roi_entry.split()[0] == label.split()[0]:
                            if roi_entry.split()[-1] == label.split()[-1]:
                                roi_to_network[label] = current_network
                                network_rois[current_network].append(label)
                                matched = True
                                break

    return roi_labels, roi_to_network, network_rois


##############################################################################
# APPROACH 1: CONN ICA-based Network Analysis (networks.nii)                #
##############################################################################

def run_conn_network_analysis():
    """
    Analyse EGG-brain synchrony using the CONN ICA-based network ROIs.
    networks.nii is 4D (91×109×91×32) — each volume is a probabilistic map
    for one of 32 ROIs belonging to 8 canonical networks.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: CONN ICA-Based Network Analysis")
    print("=" * 70)

    # Load network labels and NIfTI
    with open(NETWORKS_LABELS, 'r') as f:
        net_roi_labels = [line.strip() for line in f if line.strip()]
    print(f"  Network ROIs: {len(net_roi_labels)}")
    print(f"  Networks: {sorted(set(CONN_NETWORK_NAMES))}")

    net_img = nib.load(NETWORKS_NII)
    net_data = net_img.get_fdata()  # (91,109,91,32)
    net_affine = net_img.affine
    print(f"  Networks NIfTI shape: {net_data.shape}")

    # Load metadata
    metadata_df = pd.read_csv(METADATA_PATH)
    metadata_df = metadata_df[metadata_df['subject'].isin(SUBJECTS_WITH_BRAIN_DATA)]

    # Load data for all subjects
    all_data = {}
    for _, row in metadata_df.iterrows():
        subject = row['subject']
        run = int(row['run'])

        # EGG data
        egg_dir = project_root / "preprocess_egg_data" / "output" / "derivatives"
        gastric_file = egg_dir / subject / f"{subject}{run}" / f"gast_data_{subject}_run{run}{clean_level}.npy"
        freq_file = egg_dir / subject / f"{subject}{run}" / f"max_freq{subject}_run{run}{clean_level}.npy"

        if not gastric_file.exists():
            print(f"  ✗ Missing EGG for {subject}")
            continue

        gastric_signal = np.load(gastric_file)
        gastric_freq = float(np.load(freq_file))

        # Brain data
        brain_file = pathlib.Path(brain_data_path) / f"dswauRestingState_EGG_{subject}.nii"
        if not brain_file.exists():
            print(f"  ✗ Missing brain data for {subject}")
            continue

        fmri_img = nib.load(brain_file)
        fmri_data = fmri_img.get_fdata()
        n_tp = fmri_data.shape[3]
        print(f"\n  Loading {subject} run {run}... fMRI shape={fmri_data.shape}, "
              f"gastric freq={gastric_freq:.4f} Hz")

        # Resample network maps to fMRI space if needed
        if net_data.shape[:3] != fmri_data.shape[:3]:
            from nilearn.image import resample_to_img
            net_resampled = resample_to_img(net_img, fmri_img, interpolation='continuous')
            net_data_subj = net_resampled.get_fdata()
        else:
            net_data_subj = net_data

        # Extract mean time series for each network ROI (weighted by probability)
        net_roi_ts = []
        for roi_idx in range(net_data_subj.shape[3]):
            prob_map = net_data_subj[:, :, :, roi_idx]
            mask = prob_map > 0.25  # Threshold
            if np.sum(mask) < 10:
                mask = prob_map > 0.1
            if np.sum(mask) < 10:
                net_roi_ts.append(np.zeros(n_tp))
                continue
            weights = prob_map[mask]
            voxel_ts = fmri_data[mask, :]
            weighted_mean = np.average(voxel_ts, weights=weights, axis=0)
            net_roi_ts.append(weighted_mean)

        net_roi_ts = np.array(net_roi_ts)  # (32, n_tp)

        # Filter at gastric frequency
        net_roi_ts_filt = bandpass_filter_signal(net_roi_ts, sample_rate_fmri, gastric_freq)

        # Resample gastric signal
        gastric_resampled = resample_gastric_to_fmri(
            gastric_signal, intermediate_sample_rate, sample_rate_fmri, n_tp
        )

        min_len = min(net_roi_ts_filt.shape[1], len(gastric_resampled))
        all_data[(subject, run)] = {
            'subject': subject,
            'roi_ts_filt': net_roi_ts_filt[:, :min_len],
            'roi_ts_raw': net_roi_ts[:, :min_len],
            'gastric': gastric_resampled[:min_len],
            'gastric_freq': gastric_freq,
            'n_tp': min_len
        }
        print(f"    ✓ Loaded ({min_len} timepoints)")

    if not all_data:
        print("  No data loaded. Exiting Approach 1.")
        return None

    # Compute empirical PLV/awPLV per network ROI
    print(f"\n  Computing empirical PLV and awPLV for {len(net_roi_labels)} ROIs...")
    empirical = {}
    for (subj, run), d in all_data.items():
        roi_phase = np.angle(hilbert(d['roi_ts_filt'], axis=-1))
        gastric_phase = np.angle(hilbert(d['gastric']))
        plv = calc_plv(roi_phase, gastric_phase)
        awplv = calc_awplv(d['roi_ts_filt'], d['gastric'])
        empirical[(subj, run)] = {'plv': plv, 'awplv': awplv}
        print(f"    {subj}: PLV mean={np.mean(plv):.4f}, awPLV mean={np.mean(awplv):.4f}")

    # Null distribution (mismatch)
    print(f"\n  Computing null distribution (mismatch)...")
    null = {}
    for (subj, run), d in all_data.items():
        roi_phase = np.angle(hilbert(d['roi_ts_filt'], axis=-1))
        null_plvs, null_awplvs = [], []
        for (other_subj, _), od in all_data.items():
            if other_subj == subj:
                continue
            other_g = od['gastric']
            other_gp = np.angle(hilbert(other_g))
            ml = min(roi_phase.shape[1], len(other_gp))
            null_plvs.append(calc_plv(roi_phase[:, :ml], other_gp[:ml]))
            null_awplvs.append(calc_awplv(d['roi_ts_filt'][:, :ml], other_g[:ml]))

        null_plvs = np.array(null_plvs)
        null_awplvs = np.array(null_awplvs)

        p_plv = np.mean(null_plvs >= empirical[(subj, run)]['plv'][np.newaxis, :], axis=0)
        p_awplv = np.mean(null_awplvs >= empirical[(subj, run)]['awplv'][np.newaxis, :], axis=0)

        # FDR correction
        p_fdr_plv = false_discovery_control(p_plv, method='bh')
        p_fdr_awplv = false_discovery_control(p_awplv, method='bh')

        null[(subj, run)] = {
            'null_plvs': null_plvs, 'null_awplvs': null_awplvs,
            'p_plv': p_plv, 'p_awplv': p_awplv,
            'p_fdr_plv': p_fdr_plv, 'p_fdr_awplv': p_fdr_awplv,
            'null_plv_mean': np.mean(null_plvs, axis=0),
            'null_awplv_mean': np.mean(null_awplvs, axis=0)
        }
        n_sig = np.sum(p_fdr_plv < 0.05)
        print(f"    {subj}: {n_sig}/{len(p_fdr_plv)} ROIs significant (PLV FDR<0.05)")

    # Compile into DataFrame
    results = []
    for (subj, run), d in all_data.items():
        emp = empirical[(subj, run)]
        nl = null[(subj, run)]
        for i, label in enumerate(net_roi_labels):
            network = CONN_NETWORK_NAMES[i] if i < len(CONN_NETWORK_NAMES) else 'Unknown'
            results.append({
                'subject': subj, 'run': run,
                'roi_label': label, 'network': network,
                'plv_empirical': emp['plv'][i],
                'plv_null_mean': nl['null_plv_mean'][i],
                'plv_delta': emp['plv'][i] - nl['null_plv_mean'][i],
                'p_value_plv': nl['p_plv'][i],
                'p_fdr_plv': nl['p_fdr_plv'][i],
                'sig_fdr_plv': nl['p_fdr_plv'][i] < 0.05,
                'awplv_empirical': emp['awplv'][i],
                'awplv_null_mean': nl['null_awplv_mean'][i],
                'awplv_delta': emp['awplv'][i] - nl['null_awplv_mean'][i],
                'p_value_awplv': nl['p_awplv'][i],
                'p_fdr_awplv': nl['p_fdr_awplv'][i],
                'sig_fdr_awplv': nl['p_fdr_awplv'][i] < 0.05,
                'gastric_freq': d['gastric_freq']
            })

    roi_df = pd.DataFrame(results)

    # Aggregate to NETWORK level
    net_df = roi_df.groupby(['subject', 'run', 'network']).agg({
        'plv_empirical': 'mean',
        'plv_null_mean': 'mean',
        'plv_delta': 'mean',
        'awplv_empirical': 'mean',
        'awplv_null_mean': 'mean',
        'awplv_delta': 'mean',
    }).reset_index()

    # Network-level summary across subjects
    net_summary = net_df.groupby('network').agg({
        'plv_empirical': ['mean', 'std'],
        'plv_delta': 'mean',
        'awplv_empirical': ['mean', 'std'],
        'awplv_delta': 'mean'
    }).round(4)
    net_summary.columns = ['_'.join(c).strip('_') for c in net_summary.columns]
    net_summary = net_summary.reset_index().sort_values('plv_delta_mean', ascending=False)

    return roi_df, net_df, net_summary, null, empirical, all_data


##############################################################################
# APPROACH 2: Atlas-grouping Network Analysis (atlas.groups.info)            #
##############################################################################

def run_atlas_network_analysis():
    """
    Use pre-computed ROI-level results from roi_based_analysis.py and aggregate
    them into the 22 networks defined in atlas.groups.info.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Atlas-Grouped Network Analysis (22 networks)")
    print("=" * 70)

    roi_results_file = project_root / "roi_analysis" / "output" / "roi_synchrony_results.csv"
    if not roi_results_file.exists():
        print(f"  ✗ ROI results not found: {roi_results_file}")
        print("  → Run roi_based_analysis.py first.")
        return None

    roi_df = pd.read_csv(roi_results_file)
    print(f"  Loaded {len(roi_df)} ROI-level results")

    # Parse network groupings
    roi_labels, roi_to_network, network_rois = parse_atlas_groups(ATLAS_GROUPS, ATLAS_LABELS)
    print(f"  Parsed {len(network_rois)} networks from atlas.groups.info")

    # Map each ROI result row to its network
    def match_network(roi_name):
        """Find network for a given ROI name."""
        if roi_name in roi_to_network:
            return roi_to_network[roi_name]
        # Try partial match
        for label, net in roi_to_network.items():
            if roi_name.split('(')[0].strip() in label or label.split('(')[0].strip() in roi_name:
                return net
        return 'Unassigned'

    roi_df['network'] = roi_df['roi_name'].apply(match_network)

    assigned = roi_df[roi_df['network'] != 'Unassigned']
    unassigned = roi_df[roi_df['network'] == 'Unassigned']
    print(f"  Assigned: {len(assigned)} rows, Unassigned: {len(unassigned)} rows")

    if len(unassigned) > 0:
        unique_unassigned = unassigned['roi_name'].unique()
        print(f"  Unassigned ROIs ({len(unique_unassigned)}): {list(unique_unassigned[:5])}...")

    # Aggregate to network level per subject
    net_df = assigned.groupby(['subject', 'run', 'network']).agg({
        'plv_empirical': 'mean',
        'plv_null_mean': 'mean',
        'plv_delta': 'mean',
        'p_fdr_plv': lambda x: np.min(x),  # Most significant ROI
        'sig_fdr_plv': 'sum',
        'awplv_empirical': 'mean',
        'awplv_null_mean': 'mean',
        'awplv_delta': 'mean',
        'p_fdr_awplv': lambda x: np.min(x),
        'sig_fdr_awplv': 'sum',
    }).reset_index()

    net_df.rename(columns={
        'sig_fdr_plv': 'n_sig_rois_plv',
        'sig_fdr_awplv': 'n_sig_rois_awplv',
        'p_fdr_plv': 'min_p_fdr_plv',
        'p_fdr_awplv': 'min_p_fdr_awplv'
    }, inplace=True)

    # Add number of ROIs per network
    roi_counts = assigned.groupby('network')['roi_name'].nunique().reset_index()
    roi_counts.columns = ['network', 'n_rois_in_network']
    net_df = net_df.merge(roi_counts, on='network', how='left')

    # Network summary across subjects
    net_summary = net_df.groupby('network').agg({
        'plv_empirical': ['mean', 'std'],
        'plv_delta': 'mean',
        'n_sig_rois_plv': 'mean',
        'awplv_empirical': ['mean', 'std'],
        'awplv_delta': 'mean',
        'n_sig_rois_awplv': 'mean',
        'n_rois_in_network': 'first'
    }).round(4)
    net_summary.columns = ['_'.join(c).strip('_') for c in net_summary.columns]
    net_summary = net_summary.reset_index().sort_values('plv_delta_mean', ascending=False)

    return roi_df, net_df, net_summary, network_rois


##############################################################################
# Plotting Functions                                                         #
##############################################################################

def plot_network_density_grid(net_df, network_names, plot_dir, approach_label):
    """
    V5-style density grid: paired PLV (blue) + awPLV (red) side-by-side
    per network, matching the ROI density grid style.
    """
    networks = sorted(network_names)
    n = len(networks)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(figsize=(24, 5 * n_rows + 2))
    outer_grid = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                   wspace=0.35, hspace=0.55,
                                   top=0.93, bottom=0.08, left=0.06, right=0.96)

    fig.suptitle(f'{approach_label}: Network-Level EGG-Brain Synchrony (PLV & awPLV)\n'
                 'Empirical (filled) vs Null/Mismatch (dashed)',
                 fontsize=22, weight='bold', y=0.98)

    for idx, network in enumerate(networks):
        r = idx // n_cols
        c = idx % n_cols
        ndata = net_df[net_df['network'] == network]
        emp_plv = ndata['plv_empirical'].values
        null_plv = ndata['plv_null_mean'].values
        emp_awplv = ndata['awplv_empirical'].values
        null_awplv = ndata['awplv_null_mean'].values

        # Mann-Whitney U tests
        if len(emp_plv) >= 2 and len(null_plv) >= 2:
            _, p_mw_plv = mannwhitneyu(emp_plv, null_plv, alternative='greater')
        else:
            p_mw_plv = 1.0
        if len(emp_awplv) >= 2 and len(null_awplv) >= 2:
            _, p_mw_awplv = mannwhitneyu(emp_awplv, null_awplv, alternative='greater')
        else:
            p_mw_awplv = 1.0

        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[r, c], wspace=0.3)

        # --- PLV panel ---
        ax_plv = fig.add_subplot(inner[0])
        if len(null_plv) >= 2:
            sns.kdeplot(null_plv, ax=ax_plv, color='blue', linestyle='--',
                       label='Null', warn_singular=False)
        if len(emp_plv) >= 2:
            sns.kdeplot(emp_plv, ax=ax_plv, color='blue', fill=True, alpha=0.3,
                       label='Empirical', warn_singular=False)
        ax_plv.scatter(emp_plv, np.zeros_like(emp_plv) + 0.1, color='blue',
                      marker='o', s=40, zorder=5, alpha=0.8)
        ax_plv.scatter(null_plv, np.zeros_like(null_plv) - 0.1, color='blue',
                      marker='x', s=30, zorder=5, alpha=0.6)
        ax_plv.axvline(np.mean(emp_plv), color='blue', linewidth=1.5, alpha=0.7)
        if len(null_plv) > 0:
            ax_plv.axvline(np.mean(null_plv), color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

        sig = _sig_marker(p_mw_plv)
        delta_plv = np.mean(emp_plv) - np.mean(null_plv)
        ax_plv.set_title(f"PLV Δ={delta_plv:.4f}\np={p_mw_plv:.3f} {sig}", fontsize=10)
        ax_plv.set_xlabel('PLV', fontsize=10)
        ax_plv.set_ylabel('Density', fontsize=10)
        ax_plv.tick_params(labelsize=8)
        if idx == 0:
            ax_plv.legend(fontsize=8, loc='upper right')

        # --- awPLV panel ---
        ax_awplv = fig.add_subplot(inner[1])
        if len(null_awplv) >= 2:
            sns.kdeplot(null_awplv, ax=ax_awplv, color='red', linestyle='--',
                       label='Null', warn_singular=False)
        if len(emp_awplv) >= 2:
            sns.kdeplot(emp_awplv, ax=ax_awplv, color='red', fill=True, alpha=0.3,
                       label='Empirical', warn_singular=False)
        ax_awplv.scatter(emp_awplv, np.zeros_like(emp_awplv) + 0.1, color='red',
                        marker='o', s=40, zorder=5, alpha=0.8)
        ax_awplv.scatter(null_awplv, np.zeros_like(null_awplv) - 0.1, color='red',
                        marker='x', s=30, zorder=5, alpha=0.6)
        ax_awplv.axvline(np.mean(emp_awplv), color='red', linewidth=1.5, alpha=0.7)
        if len(null_awplv) > 0:
            ax_awplv.axvline(np.mean(null_awplv), color='red', linestyle=':', linewidth=1.5, alpha=0.5)

        sig = _sig_marker(p_mw_awplv)
        delta_awplv = np.mean(emp_awplv) - np.mean(null_awplv)
        ax_awplv.set_title(f"awPLV Δ={delta_awplv:.4f}\np={p_mw_awplv:.3f} {sig}", fontsize=10)
        ax_awplv.set_xlabel('awPLV', fontsize=10)
        ax_awplv.set_ylabel('')
        ax_awplv.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_awplv.tick_params(labelsize=8)
        if idx == 0:
            ax_awplv.legend(fontsize=8, loc='upper right')

        # Network name label above the paired panels
        pos_plv = ax_plv.get_position()
        pos_awplv = ax_awplv.get_position()
        mid_x = (pos_plv.x0 + pos_awplv.x1) / 2
        mid_y = max(pos_plv.y1, pos_awplv.y1) + 0.01
        fig.text(mid_x, mid_y, network, ha='center', va='bottom', fontsize=11,
                weight='bold', style='italic')

    fig.text(0.5, 0.02,
             r"$\bf{Figure}$: KDE density of EGG-brain synchrony per network. "
             "Empirical (filled/circles) vs Null-mismatch (dashed/crosses).\n"
             "PLV in blue; awPLV in red. "
             "Stars: * p<0.05, ** p<0.01, *** p<0.001 (Mann-Whitney U).",
             ha='center', fontsize=13)

    fname = plot_dir / f"network_density_grid_{approach_label.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fname.name}")


def plot_network_pooled_density(net_df, plot_dir, approach_label):
    """
    Pooled PLV and awPLV density: empirical vs null distributions
    aggregated across all networks and subjects. Side-by-side panels.
    """
    plv_emp = net_df['plv_empirical'].values
    plv_null = net_df['plv_null_mean'].values
    awplv_emp = net_df['awplv_empirical'].values
    awplv_null = net_df['awplv_null_mean'].values

    fig, (ax_plv, ax_awplv) = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLV ---
    sns.kdeplot(plv_emp, ax=ax_plv, color='blue', fill=True, alpha=0.3,
               label=f'Empirical (n={len(plv_emp)})')
    sns.kdeplot(plv_null, ax=ax_plv, color='blue', linestyle='--',
               label=f'Null (n={len(plv_null)})')
    ax_plv.axvline(np.mean(plv_emp), color='blue', linewidth=2, alpha=0.7)
    ax_plv.axvline(np.mean(plv_null), color='blue', linestyle=':', linewidth=2, alpha=0.5)

    stat, p_val = mannwhitneyu(plv_emp, plv_null, alternative='greater')
    sig = _sig_marker(p_val)
    ax_plv.set_title(f'PLV: Empirical vs Null\n'
                     f'Effect={np.mean(plv_emp) - np.mean(plv_null):.4f}, '
                     f'p={p_val:.4f} {sig}', fontsize=16)
    ax_plv.set_xlabel('PLV', fontsize=14)
    ax_plv.set_ylabel('Density', fontsize=14)
    ax_plv.legend(fontsize=12, loc='upper right')
    ax_plv.tick_params(labelsize=12)

    # --- awPLV ---
    sns.kdeplot(awplv_emp, ax=ax_awplv, color='red', fill=True, alpha=0.3,
               label=f'Empirical (n={len(awplv_emp)})')
    sns.kdeplot(awplv_null, ax=ax_awplv, color='red', linestyle='--',
               label=f'Null (n={len(awplv_null)})')
    ax_awplv.axvline(np.mean(awplv_emp), color='red', linewidth=2, alpha=0.7)
    ax_awplv.axvline(np.mean(awplv_null), color='red', linestyle=':', linewidth=2, alpha=0.5)

    stat, p_val = mannwhitneyu(awplv_emp, awplv_null, alternative='greater')
    sig = _sig_marker(p_val)
    ax_awplv.set_title(f'awPLV: Empirical vs Null\n'
                       f'Effect={np.mean(awplv_emp) - np.mean(awplv_null):.4f}, '
                       f'p={p_val:.4f} {sig}', fontsize=16)
    ax_awplv.set_xlabel('awPLV', fontsize=14)
    ax_awplv.set_ylabel('Density', fontsize=14)
    ax_awplv.legend(fontsize=12, loc='upper right')
    ax_awplv.tick_params(labelsize=12)

    fig.suptitle(f'{approach_label}: Pooled Network-Level Synchrony',
                 fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fname = plot_dir / f"network_pooled_density_{approach_label.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fname.name}")


def plot_network_bar_comparison(net_summary, plot_dir, approach_label):
    """
    Side-by-side bar plot of PLV and awPLV delta (empirical - null) per network.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(net_summary) * 0.4)))

    # PLV delta
    sorted_df = net_summary.sort_values('plv_delta_mean', ascending=True)
    colors_plv = ['steelblue' if v > 0 else 'lightcoral' for v in sorted_df['plv_delta_mean']]
    axes[0].barh(sorted_df['network'], sorted_df['plv_delta_mean'], color=colors_plv, edgecolor='gray')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('PLV Delta (Empirical − Null)', fontsize=12)
    axes[0].set_title('PLV Effect by Network', fontsize=14, weight='bold')
    axes[0].tick_params(labelsize=10)

    # awPLV delta
    sorted_df = net_summary.sort_values('awplv_delta_mean', ascending=True)
    colors_awplv = ['coral' if v > 0 else 'lightskyblue' for v in sorted_df['awplv_delta_mean']]
    axes[1].barh(sorted_df['network'], sorted_df['awplv_delta_mean'], color=colors_awplv, edgecolor='gray')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_xlabel('awPLV Delta (Empirical − Null)', fontsize=12)
    axes[1].set_title('awPLV Effect by Network', fontsize=14, weight='bold')
    axes[1].tick_params(labelsize=10)

    fig.suptitle(f'{approach_label}: Network-Level EGG-Brain Synchrony',
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fname = plot_dir / f"network_bar_{approach_label.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fname.name}")


def plot_combined_network_summary(conn_summary, atlas_summary, plot_dir):
    """
    Combined figure comparing both approaches.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Row 1: CONN networks
    if conn_summary is not None:
        s = conn_summary.sort_values('plv_delta_mean', ascending=True)
        c = ['steelblue' if v > 0 else 'lightcoral' for v in s['plv_delta_mean']]
        axes[0, 0].barh(s['network'], s['plv_delta_mean'], color=c, edgecolor='gray')
        axes[0, 0].axvline(0, color='black', linewidth=0.8)
        axes[0, 0].set_xlabel('PLV Delta')
        axes[0, 0].set_title('CONN ICA Networks — PLV', fontsize=13, weight='bold')

        s = conn_summary.sort_values('awplv_delta_mean', ascending=True)
        c = ['coral' if v > 0 else 'lightskyblue' for v in s['awplv_delta_mean']]
        axes[0, 1].barh(s['network'], s['awplv_delta_mean'], color=c, edgecolor='gray')
        axes[0, 1].axvline(0, color='black', linewidth=0.8)
        axes[0, 1].set_xlabel('awPLV Delta')
        axes[0, 1].set_title('CONN ICA Networks — awPLV', fontsize=13, weight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'No CONN data', ha='center', va='center', fontsize=14)
        axes[0, 1].text(0.5, 0.5, 'No CONN data', ha='center', va='center', fontsize=14)

    # Row 2: Atlas-grouped networks
    if atlas_summary is not None:
        s = atlas_summary.sort_values('plv_delta_mean', ascending=True)
        c = ['steelblue' if v > 0 else 'lightcoral' for v in s['plv_delta_mean']]
        axes[1, 0].barh(s['network'], s['plv_delta_mean'], color=c, edgecolor='gray')
        axes[1, 0].axvline(0, color='black', linewidth=0.8)
        axes[1, 0].set_xlabel('PLV Delta')
        axes[1, 0].set_title('Atlas-Grouped Networks (22) — PLV', fontsize=13, weight='bold')

        s = atlas_summary.sort_values('awplv_delta_mean', ascending=True)
        c = ['coral' if v > 0 else 'lightskyblue' for v in s['awplv_delta_mean']]
        axes[1, 1].barh(s['network'], s['awplv_delta_mean'], color=c, edgecolor='gray')
        axes[1, 1].axvline(0, color='black', linewidth=0.8)
        axes[1, 1].set_xlabel('awPLV Delta')
        axes[1, 1].set_title('Atlas-Grouped Networks (22) — awPLV', fontsize=13, weight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'No atlas data', ha='center', va='center', fontsize=14)
        axes[1, 1].text(0.5, 0.5, 'No atlas data', ha='center', va='center', fontsize=14)

    fig.suptitle('EGG-Brain Synchrony: Network-Level Comparison\n'
                 '(Blue = positive effect, Red = negative)',
                 fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fname = plot_dir / "network_combined_comparison.png"
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fname.name}")


##############################################################################
# Main                                                                       #
##############################################################################

def main():
    print("=" * 70)
    print("NETWORK-LEVEL EGG-BRAIN SYNCHRONY ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── APPROACH 1: CONN ICA networks ──
    conn_result = run_conn_network_analysis()
    conn_summary = None
    if conn_result is not None:
        conn_roi_df, conn_net_df, conn_summary, conn_null, conn_emp, conn_data = conn_result

        # Save CSVs
        conn_roi_df.to_csv(OUTPUT_DIR / "conn_network_roi_results.csv", index=False)
        conn_net_df.to_csv(OUTPUT_DIR / "conn_network_level_results.csv", index=False)
        conn_summary.to_csv(OUTPUT_DIR / "conn_network_summary.csv", index=False)
        print(f"\n  ✓ Saved CONN results to {OUTPUT_DIR}")

        # Print summary
        print("\n  CONN Network Summary (sorted by PLV delta):")
        print(conn_summary.to_string(index=False))

        # Plots
        print("\n  Creating CONN network plots...")
        plot_network_density_grid(conn_net_df, conn_net_df['network'].unique(),
                                   PLOT_DIR, 'CONN ICA')
        plot_network_pooled_density(conn_net_df, PLOT_DIR, 'CONN ICA')
        plot_network_bar_comparison(conn_summary, PLOT_DIR, 'CONN ICA')

    # ── APPROACH 2: Atlas-grouped networks ──
    atlas_result = run_atlas_network_analysis()
    atlas_summary = None
    if atlas_result is not None:
        atlas_roi_df, atlas_net_df, atlas_summary, network_rois = atlas_result

        # Save CSVs
        atlas_roi_df.to_csv(OUTPUT_DIR / "atlas_network_roi_results.csv", index=False)
        atlas_net_df.to_csv(OUTPUT_DIR / "atlas_network_level_results.csv", index=False)
        atlas_summary.to_csv(OUTPUT_DIR / "atlas_network_summary.csv", index=False)
        print(f"\n  ✓ Saved atlas-grouped results to {OUTPUT_DIR}")

        # Print summary
        print("\n  Atlas-Grouped Network Summary (sorted by PLV delta):")
        print(atlas_summary.to_string(index=False))

        # Plots
        print("\n  Creating atlas-grouped network plots...")
        plot_network_density_grid(atlas_net_df, atlas_net_df['network'].unique(),
                                   PLOT_DIR, 'Atlas Grouped')
        plot_network_pooled_density(atlas_net_df, PLOT_DIR, 'Atlas Grouped')
        plot_network_bar_comparison(atlas_summary, PLOT_DIR, 'Atlas Grouped')

    # ── Combined comparison ──
    if conn_summary is not None or atlas_summary is not None:
        print("\n  Creating combined comparison plot...")
        plot_combined_network_summary(conn_summary, atlas_summary, PLOT_DIR)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("NETWORK ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Plot directory:   {PLOT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")
    for f in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
