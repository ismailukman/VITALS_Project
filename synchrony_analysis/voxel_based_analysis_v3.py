import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import (sample_rate_fmri, intermediate_sample_rate, clean_level, main_project_path)
import nibabel as nib
from scipy import signal
from scipy.stats import false_discovery_control
from matplotlib import pyplot as plt
import pandas as pd
from utils.gastric_utils import to_phase_resampled, plot_example_synchrony
import numpy as np
from scipy.signal import hilbert

##############################################################################
# Configuration                                                              #
##############################################################################

META_DATAFRAME_PATH = pathlib.Path(__file__).parent.parent / "dataframes" / "egg_brain_meta_data_10subjects.csv"
OUTPUT_DIR = pathlib.Path(main_project_path) / "derivatives" / "brain_gast_voxel_analysis_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = pathlib.Path(main_project_path).parent / "plots" / "brain_gast_voxel_v3"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

##############################################################################
# Helper Functions                                                           #
##############################################################################

def calc_plv_voxelwise(brain_phase, gastric_phase):
    """
    Compute voxel-wise PLV between brain signals and gastric signal.

    Args:
        brain_phase: Array of shape (n_voxels, n_timepoints) - brain phase
        gastric_phase: Array of shape (n_timepoints,) - gastric phase

    Returns:
        plv: Array of shape (n_voxels,) - PLV for each voxel
    """
    plv = np.abs(np.mean(np.exp(1j * (brain_phase - gastric_phase[np.newaxis, :])), axis=1))
    return plv


def calc_awplv_voxelwise(brain_signal, gastric_signal):
    """
    Compute voxel-wise amplitude-weighted PLV between brain and gastric signals.

    Args:
        brain_signal: Array of shape (n_voxels, n_timepoints) - brain signal
        gastric_signal: Array of shape (n_timepoints,) - gastric signal

    Returns:
        awplv: Array of shape (n_voxels,) - awPLV for each voxel
    """
    # Get analytic signals
    brain_analytic = signal.hilbert(brain_signal, axis=1)
    gastric_analytic = hilbert(gastric_signal)

    # Extract phase and amplitude
    brain_phase = np.angle(brain_analytic)
    gastric_phase = np.angle(gastric_analytic)
    brain_amplitude = np.abs(brain_analytic)
    gastric_amplitude = np.abs(gastric_analytic)

    # Calculate phase difference
    phase_diff = brain_phase - gastric_phase[np.newaxis, :]
    phase_diff_complex = np.exp(1j * phase_diff)

    # Calculate amplitude weights (product of brain and gastric amplitudes)
    # Shape: (n_voxels, n_timepoints)
    amplitude_product = brain_amplitude * gastric_amplitude[np.newaxis, :]

    # Normalize weights for each voxel
    sum_amplitude_product = np.sum(amplitude_product, axis=1, keepdims=True)
    # Avoid division by zero
    sum_amplitude_product[sum_amplitude_product == 0] = 1.0
    weights = amplitude_product / sum_amplitude_product

    # Compute awPLV for each voxel
    awplv = np.abs(np.sum(weights * phase_diff_complex, axis=1))

    return awplv


##############################################################################
# Main Analysis                                                              #
##############################################################################

def main():
    """
    Voxel-based gastric-brain synchrony analysis (Version 3).

    This version computes individual-level gastric-brain synchrony:
    - Computes both PLV and awPLV for each voxel
    - Uses mismatch null distribution (subject's brain vs other subjects' gastric signals)
    - Performs individual-level analysis with FDR correction
    - Saves subject-level maps and CSV results
    - Creates summary plots
    """

    print("="*70)
    print("VOXEL-BASED GASTRIC-BRAIN SYNCHRONY ANALYSIS - VERSION 3")
    print("="*70)

    # Load metadata
    record_meta_pd = pd.read_csv(META_DATAFRAME_PATH)
    subjects_runs = list(zip(record_meta_pd['subject'], record_meta_pd['run']))

    print(f"\nFound {len(subjects_runs)} subject-run pairs in metadata")

    # STEP 1: Load all subjects' data
    print("\n" + "="*70)
    print("STEP 1: Loading data for all subjects")
    print("="*70)

    all_data = {}
    common_mask = None
    reference_affine = None
    reference_header = None
    reference_shape = None

    for subject_name, run in subjects_runs:
        try:
            data_path = f'{main_project_path}/derivatives/brain_gast/{subject_name}/{subject_name}{run}'

            # Debug prints
            print(f"DEBUG: main_project_path: {main_project_path}")
            print(f"DEBUG: clean_level: {clean_level}")
            print(f"DEBUG: data_path: {data_path}")

            # Load gastric signal
            gastric_file = f'{data_path}/gast_data_{subject_name}_run{run}{clean_level}_sliced.npy'
            print(f"DEBUG: Checking for gastric_file: {gastric_file}")
            if not os.path.exists(gastric_file):
                print(f"  ✗ Missing gastric data: {subject_name} run {run}")
                continue
            gastric_signal = np.load(gastric_file)

            # Load brain signal
            brain_file = f'{data_path}/func_filtered_{subject_name}_run{run}{clean_level}_sliced.npz'
            if not os.path.exists(brain_file):
                print(f"  ✗ Missing brain data: {subject_name} run {run}")
                continue
            brain_signal = np.load(brain_file)['brain_signal']

            # Load mask
            mask_file = f'{data_path}/mask_{subject_name}_run{run}{clean_level}.npz'
            if not os.path.exists(mask_file):
                print(f"  ✗ Missing mask: {subject_name} run {run}")
                continue
            mask = np.load(mask_file)['mask']

            # Load original fMRI for spatial reference
            afni_path = f'{data_path}/{subject_name}_task-rest_run-0{run}_space-MNI_desc-preproc_bold_{clean_level}.nii.gz'
            fmriprep_path = f'{main_project_path}/fmriprep/out/{subject_name}/fmriprep/{subject_name}_task-rest_run-0{run}_space-MNI152NLin6Asym_desc-preproc_bold_{clean_level}.nii.gz'

            if os.path.exists(afni_path):
                original_fmri = nib.load(afni_path)
            elif os.path.exists(fmriprep_path):
                original_fmri = nib.load(fmriprep_path)
            else:
                print(f"  ✗ Missing fMRI: {subject_name} run {run}")
                continue

            # Store reference spatial information from first subject
            if reference_affine is None:
                reference_affine = original_fmri.affine
                reference_header = original_fmri.header
                reference_shape = original_fmri.shape[:3]
                common_mask = mask
            else:
                # Intersect masks to ensure all subjects have data at these voxels
                common_mask = common_mask & mask

            all_data[(subject_name, run)] = {
                'subject': subject_name,
                'run': run,
                'gastric': gastric_signal,
                'brain': brain_signal,
                'mask': mask,
                'affine': original_fmri.affine,
                'header': original_fmri.header,
                'shape': original_fmri.shape[:3]
            }

            print(f"  ✓ Loaded {subject_name} run {run}: {brain_signal.shape[0]} voxels, {brain_signal.shape[1]} timepoints")

        except Exception as e:
            print(f"  ✗ Error loading {subject_name} run {run}: {e}")

    if not all_data:
        print("\nNo valid data found. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(all_data)} subject-run pairs")
    print(f"Common mask contains {np.sum(common_mask)} voxels")

    # STEP 2: Compute empirical PLV and awPLV for each subject
    print("\n" + "="*70)
    print("STEP 2: Computing empirical PLV and awPLV")
    print("="*70)

    empirical_results = {}

    for idx, ((subject_name, run), data) in enumerate(all_data.items()):
        print(f"\n  Processing {subject_name} run {run} ({idx+1}/{len(all_data)})...")

        brain_signal = data['brain']
        gastric_signal = data['gastric']

        # Calculate brain phase
        brain_signal_phase = signal.hilbert(brain_signal, axis=1)
        brain_signal_phase = np.angle(brain_signal_phase)

        # Calculate gastric phase
        gastric_signal_phase = to_phase_resampled(gastric_signal, intermediate_sample_rate, sample_rate_fmri)

        # Match lengths for all calculations
        min_length = min(brain_signal.shape[1], len(gastric_signal_phase))
        brain_signal_matched = brain_signal[:, :min_length]
        brain_phase_matched = brain_signal_phase[:, :min_length]
        gastric_phase_matched = gastric_signal_phase[:min_length]

        # For awPLV, we need the raw gastric signal at fMRI sample rate
        # Resample gastric signal to match brain timepoints
        from scipy.interpolate import interp1d
        gastric_time_orig = np.arange(len(gastric_signal)) / intermediate_sample_rate
        gastric_time_new = np.arange(brain_signal.shape[1]) / sample_rate_fmri
        # Only use times within original range
        valid_idx = gastric_time_new <= gastric_time_orig[-1]
        gastric_time_new = gastric_time_new[valid_idx]

        interp_func = interp1d(gastric_time_orig, gastric_signal, kind='cubic', fill_value='extrapolate')
        gastric_signal_resampled = interp_func(gastric_time_new)

        # Match lengths again
        min_length_signal = min(brain_signal.shape[1], len(gastric_signal_resampled))
        gastric_signal_matched = gastric_signal_resampled[:min_length_signal]
        brain_signal_for_awplv = brain_signal[:, :min_length_signal]

        # Calculate PLV
        plv_empirical = calc_plv_voxelwise(brain_phase_matched, gastric_phase_matched)

        # Calculate awPLV
        awplv_empirical = calc_awplv_voxelwise(brain_signal_for_awplv, gastric_signal_matched)

        empirical_results[(subject_name, run)] = {
            'plv': plv_empirical,
            'awplv': awplv_empirical,
            'brain_phase': brain_signal_phase,
            'brain_signal': brain_signal
        }

        print(f"    PLV: mean={np.mean(plv_empirical):.4f}, std={np.std(plv_empirical):.4f}")
        print(f"    awPLV: mean={np.mean(awplv_empirical):.4f}, std={np.std(awplv_empirical):.4f}")

    # STEP 3: Compute null distribution using mismatch approach
    print("\n" + "="*70)
    print("STEP 3: Computing null distribution (mismatch)")
    print("="*70)

    null_results = {}
    individual_stats = []

    for idx, ((subject_name, run), data) in enumerate(all_data.items()):
        print(f"\n  Computing null for {subject_name} run {run} ({idx+1}/{len(all_data)})...")

        brain_phase = empirical_results[(subject_name, run)]['brain_phase']
        brain_signal = empirical_results[(subject_name, run)]['brain_signal']
        n_voxels = brain_signal.shape[0]

        # Collect null PLV and awPLV values by pairing with other subjects' gastric signals
        null_plvs = []
        null_awplvs = []

        for (other_subject, other_run), other_data in all_data.items():
            if other_subject == subject_name:
                continue  # Skip same subject

            other_gastric = other_data['gastric']
            other_gastric_phase = to_phase_resampled(other_gastric, intermediate_sample_rate, sample_rate_fmri)

            # Match lengths for PLV
            min_length = min(brain_phase.shape[1], len(other_gastric_phase))

            # Calculate null PLV
            null_plv = calc_plv_voxelwise(brain_phase[:, :min_length], other_gastric_phase[:min_length])
            null_plvs.append(null_plv)

            # For awPLV, resample gastric signal to brain timepoints
            from scipy.interpolate import interp1d
            gastric_time_orig = np.arange(len(other_gastric)) / intermediate_sample_rate
            gastric_time_new = np.arange(brain_signal.shape[1]) / sample_rate_fmri
            valid_idx = gastric_time_new <= gastric_time_orig[-1]
            gastric_time_new = gastric_time_new[valid_idx]

            interp_func = interp1d(gastric_time_orig, other_gastric, kind='cubic', fill_value='extrapolate')
            other_gastric_resampled = interp_func(gastric_time_new)

            min_length_awplv = min(brain_signal.shape[1], len(other_gastric_resampled))

            # Calculate null awPLV
            null_awplv = calc_awplv_voxelwise(brain_signal[:, :min_length_awplv], other_gastric_resampled[:min_length_awplv])
            null_awplvs.append(null_awplv)

        # Convert to arrays: shape (n_null_subjects, n_voxels)
        null_plvs = np.array(null_plvs)
        null_awplvs = np.array(null_awplvs)

        # Calculate p-values for each voxel
        plv_empirical = empirical_results[(subject_name, run)]['plv']
        awplv_empirical = empirical_results[(subject_name, run)]['awplv']

        p_values_plv = np.mean(null_plvs >= plv_empirical[np.newaxis, :], axis=0)
        p_values_awplv = np.mean(null_awplvs >= awplv_empirical[np.newaxis, :], axis=0)

        # Calculate null statistics
        plv_null_median = np.median(null_plvs, axis=0)
        plv_null_mean = np.mean(null_plvs, axis=0)
        plv_null_std = np.std(null_plvs, axis=0)

        awplv_null_median = np.median(null_awplvs, axis=0)
        awplv_null_mean = np.mean(null_awplvs, axis=0)
        awplv_null_std = np.std(null_awplvs, axis=0)

        # Apply FDR correction across voxels
        p_fdr_plv = false_discovery_control(p_values_plv, method='bh')
        p_fdr_awplv = false_discovery_control(p_values_awplv, method='bh')

        # Apply Bonferroni correction
        bonferroni_alpha = 0.05 / n_voxels
        p_bonferroni_plv = p_values_plv
        p_bonferroni_awplv = p_values_awplv

        null_results[(subject_name, run)] = {
            'p_values_plv': p_values_plv,
            'p_values_awplv': p_values_awplv,
            'p_fdr_plv': p_fdr_plv,
            'p_fdr_awplv': p_fdr_awplv,
            'p_bonferroni_plv': p_bonferroni_plv,
            'p_bonferroni_awplv': p_bonferroni_awplv,
            'plv_null_median': plv_null_median,
            'plv_null_mean': plv_null_mean,
            'plv_null_std': plv_null_std,
            'awplv_null_median': awplv_null_median,
            'awplv_null_mean': awplv_null_mean,
            'awplv_null_std': awplv_null_std,
            'n_null': len(null_plvs)
        }

        # Store individual statistics
        individual_stats.append({
            'subject': subject_name,
            'run': run,
            'plv_empirical': np.mean(plv_empirical),
            'plv_null_median': np.mean(plv_null_median),
            'plv_null_mean': np.mean(plv_null_mean),
            'plv_null_std': np.mean(plv_null_std),
            'plv_median': np.median(plv_empirical),
            'plv_std': np.std(plv_empirical),
            'awplv_empirical': np.mean(awplv_empirical),
            'awplv_null_median': np.mean(awplv_null_median),
            'awplv_null_mean': np.mean(awplv_null_mean),
            'awplv_null_std': np.mean(awplv_null_std),
            'awplv_median': np.median(awplv_empirical),
            'awplv_std': np.std(awplv_empirical),
            'n_sig_voxels_uncorrected_plv': np.sum(p_values_plv < 0.05),
            'n_sig_voxels_uncorrected_awplv': np.sum(p_values_awplv < 0.05),
            'n_sig_voxels_fdr_plv': np.sum(p_fdr_plv < 0.05),
            'n_sig_voxels_fdr_awplv': np.sum(p_fdr_awplv < 0.05),
            'n_sig_voxels_bonferroni_plv': np.sum(p_bonferroni_plv < bonferroni_alpha),
            'n_sig_voxels_bonferroni_awplv': np.sum(p_bonferroni_awplv < bonferroni_alpha),
            'n_voxels': n_voxels,
            'n_null_subjects': len(null_plvs),
            'bonferroni_alpha': bonferroni_alpha
        })

        print(f"    PLV: {np.sum(p_values_plv < 0.05)} uncorrected, {np.sum(p_fdr_plv < 0.05)} FDR, {np.sum(p_bonferroni_plv < bonferroni_alpha)} Bonferroni")
        print(f"    awPLV: {np.sum(p_values_awplv < 0.05)} uncorrected, {np.sum(p_fdr_awplv < 0.05)} FDR, {np.sum(p_bonferroni_awplv < bonferroni_alpha)} Bonferroni")

    # Save individual-level statistics
    individual_stats_df = pd.DataFrame(individual_stats)
    individual_stats_df.to_csv(OUTPUT_DIR / "individual_level_statistics.csv", index=False)
    print(f"\n  ✓ Saved individual statistics to {OUTPUT_DIR / 'individual_level_statistics.csv'}")

    # STEP 4: Save subject-level maps
    print("\n" + "="*70)
    print("STEP 4: Saving subject-level maps")
    print("="*70)

    for (subject_name, run), data in all_data.items():
        subject_dir = OUTPUT_DIR / subject_name / f"{subject_name}{run}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        mask = data['mask']
        affine = data['affine']
        header = data['header']
        shape = data['shape']

        plv_emp = empirical_results[(subject_name, run)]['plv']
        awplv_emp = empirical_results[(subject_name, run)]['awplv']
        null_stats = null_results[(subject_name, run)]

        # Save maps
        maps_to_save = {
            'plv_empirical': plv_emp,
            'awplv_empirical': awplv_emp,
            'plv_p_values': null_stats['p_values_plv'],
            'awplv_p_values': null_stats['p_values_awplv'],
            'plv_p_fdr': null_stats['p_fdr_plv'],
            'awplv_p_fdr': null_stats['p_fdr_awplv'],
            'plv_delta': plv_emp - null_stats['plv_null_median'],
            'awplv_delta': awplv_emp - null_stats['awplv_null_median']
        }

        for map_name, map_data in maps_to_save.items():
            vol = np.zeros(shape)
            vol[mask] = map_data
            img = nib.Nifti1Image(vol, affine=affine, header=header)
            nib.save(img, subject_dir / f"{map_name}_{subject_name}_run{run}{clean_level}.nii.gz")

        print(f"  ✓ Saved maps for {subject_name} run {run}")

    # STEP 5: Create summary plots
    print("\n" + "="*70)
    print("STEP 5: Creating summary plots")
    print("="*70)

    # Plot subject-level statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PLV empirical per subject
    axes[0, 0].bar(range(len(individual_stats_df)), individual_stats_df['plv_empirical'])
    axes[0, 0].set_xlabel('Subject-Run Index', fontsize=12)
    axes[0, 0].set_ylabel('Mean PLV', fontsize=12)
    axes[0, 0].set_title('Mean PLV per Subject-Run', fontsize=14)
    axes[0, 0].grid(alpha=0.3)

    # awPLV empirical per subject
    axes[0, 1].bar(range(len(individual_stats_df)), individual_stats_df['awplv_empirical'], color='red')
    axes[0, 1].set_xlabel('Subject-Run Index', fontsize=12)
    axes[0, 1].set_ylabel('Mean awPLV', fontsize=12)
    axes[0, 1].set_title('Mean awPLV per Subject-Run', fontsize=14)
    axes[0, 1].grid(alpha=0.3)

    # Number of significant voxels PLV
    axes[1, 0].bar(range(len(individual_stats_df)), individual_stats_df['n_sig_voxels_fdr_plv'])
    axes[1, 0].set_xlabel('Subject-Run Index', fontsize=12)
    axes[1, 0].set_ylabel('Number of Significant Voxels', fontsize=12)
    axes[1, 0].set_title('Significant Voxels (PLV, FDR < 0.05)', fontsize=14)
    axes[1, 0].grid(alpha=0.3)

    # Number of significant voxels awPLV
    axes[1, 1].bar(range(len(individual_stats_df)), individual_stats_df['n_sig_voxels_fdr_awplv'], color='red')
    axes[1, 1].set_xlabel('Subject-Run Index', fontsize=12)
    axes[1, 1].set_ylabel('Number of Significant Voxels', fontsize=12)
    axes[1, 1].set_title('Significant Voxels (awPLV, FDR < 0.05)', fontsize=14)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "individual_level_summary.png", dpi=200)
    plt.close()
    print(f"  ✓ Saved individual summary plot to {PLOT_DIR / 'individual_level_summary.png'}")

    # STEP 6: Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nProcessed {len(all_data)} subject-run pairs")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Plot directory: {PLOT_DIR}")
    print("\nFiles saved:")
    print(f"  - Individual statistics: individual_level_statistics.csv")
    print(f"  - Subject-level maps: {len(all_data)} subjects with 8 maps each")
    print(f"  - Plots: 1 summary plot")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()