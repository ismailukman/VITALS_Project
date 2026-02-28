from nilearn import plotting
from nilearn.image import resample_to_img
import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import (sample_rate_fmri, intermediate_sample_rate, clean_level, main_project_path)
import nibabel as nib
from scipy import signal
from matplotlib import pyplot as plt
import os
import pandas as pd
import argparse
from utils.gastric_utils import to_phase_resampled, plot_example_synchrony
import numpy as np

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='subject_name', help='The subject to process')
parser.add_argument('run', metavar='run_number', help='The run number of the subject')
args = parser.parse_args()
subject_name = args.subject
run = str(args.run)

print('Processing subject:', subject_name, 'run:', run)

# Load data - updated paths to use main_project_path
data_path = f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' + subject_name+run
MNI_tamplate_path = os.environ['FSLDIR'] + '/data/standard/MNI152_T1_2mm.nii.gz'
gastric_signal = np.load(data_path + '/gast_data_' + subject_name + '_run' + run + clean_level +
                         '_sliced.npy')
brain_signal = np.load(data_path + '/func_filtered_' + subject_name + '_run' + run + clean_level +
                       '_sliced.npz')['brain_signal']

# Try AFNI path first, then fall back to fMRIPrep path
afni_path = f'{data_path}/{subject_name}_task-rest_run-0{run}_space-MNI_desc-preproc_bold_{clean_level}.nii.gz'
fmriprep_path = f'{main_project_path}/fmriprep/out/{subject_name}/fmriprep/{subject_name}_task-rest_run-0{run}_space-MNI152NLin6Asym_desc-preproc_bold_{clean_level}.nii.gz'

if os.path.exists(afni_path):
    original_fmri = nib.load(afni_path)
    print(f'Using AFNI preprocessed data: {afni_path}')
elif os.path.exists(fmriprep_path):
    original_fmri = nib.load(fmriprep_path)
    print(f'Using fMRIPrep data: {fmriprep_path}')
else:
    raise FileNotFoundError(f"Preprocessed fMRI not found at:\n  {afni_path}\n  or {fmriprep_path}")
record_meta_pd = pd.read_csv('dataframes/egg_brain_meta_data_v2.csv')
record_meta = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name) &
                                    (record_meta_pd['run'] == int(run)),:].to_dict('records')[0]

MNI_tamplate_3mm = resample_to_img(nib.load(MNI_tamplate_path), original_fmri)
mask = np.load(f'{main_project_path}/derivatives/brain_gast/mask_' + subject_name + '_run' + run + clean_level + '.npz')['mask']
plot_path = f'{main_project_path}/plots/brain_gast/' + subject_name + '/' + subject_name+run

# calculate the PLV
# see Time series analysis in neuroscience. Alexander Zhigalov Dept. of CS, University of Helsinki and Dept. of NBE,
# Aalto University

# calc phase
brain_signal_phase = signal.hilbert(brain_signal, axis=1)
brain_signal_phase = np.apply_along_axis(np.angle, 1, brain_signal_phase)
gastric_signal_phase = to_phase_resampled(gastric_signal, intermediate_sample_rate, sample_rate_fmri)

# calc phase-locking value
plvs_empirical = np.abs(np.mean(np.exp(1j * (brain_signal_phase - gastric_signal_phase[np.newaxis,:])), axis=1))

# Calculate amplitude-weighted PLV (awPLV)
# Resample gastric signal to fMRI sampling rate for amplitude calculation
from scipy.interpolate import interp1d
gastric_time_orig = np.arange(len(gastric_signal)) / intermediate_sample_rate
gastric_time_new = np.arange(brain_signal.shape[1]) / sample_rate_fmri
valid_idx = gastric_time_new <= gastric_time_orig[-1]
gastric_time_new = gastric_time_new[valid_idx]
interp_func = interp1d(gastric_time_orig, gastric_signal, kind='cubic', fill_value='extrapolate')
gastric_signal_resampled = interp_func(gastric_time_new)

# Match lengths
min_length = min(brain_signal.shape[1], len(gastric_signal_resampled))
brain_signal_matched = brain_signal[:, :min_length]
gastric_signal_matched = gastric_signal_resampled[:min_length]

# Get analytic signals for amplitude
brain_analytic = signal.hilbert(brain_signal_matched, axis=1)
gastric_analytic = signal.hilbert(gastric_signal_matched)

# Extract amplitudes
brain_amplitude = np.abs(brain_analytic)
gastric_amplitude = np.abs(gastric_analytic)

# Calculate amplitude-weighted phase locking
amplitude_product = brain_amplitude * gastric_amplitude[np.newaxis, :]
sum_amplitude_product = np.sum(amplitude_product, axis=1, keepdims=True)
sum_amplitude_product[sum_amplitude_product == 0] = 1.0
weights = amplitude_product / sum_amplitude_product

# Compute awPLV
phase_diff = np.angle(brain_analytic) - np.angle(gastric_analytic)[np.newaxis, :]
awplvs_empirical = np.abs(np.sum(weights * np.exp(1j * phase_diff), axis=1))

print(f'PLV stats: mean={np.mean(plvs_empirical):.4f}, std={np.std(plvs_empirical):.4f}')
print(f'awPLV stats: mean={np.mean(awplvs_empirical):.4f}, std={np.std(awplvs_empirical):.4f}')

vol_new = np.zeros(original_fmri.shape[:3])  # Use only spatial dimensions (x, y, z)
vol_new[mask] = plvs_empirical
img_plv = nib.Nifti1Image(vol_new, affine = original_fmri.affine, header=original_fmri.header)

# plot an example of high/ low/ random gastric-brain synchrony
plot_example_synchrony(gastric_signal, brain_signal, plvs_empirical, plot_path + 'egg_BOLD_sync_example.png')

# plot the empirical PLV map
plotting.plot_stat_map(img_plv, bg_img = MNI_tamplate_3mm, title="plot_stat_map",colorbar = True, threshold=np.percentile(plvs_empirical,95))
plt.savefig(plot_path + 'empirical_plv_map.png', dpi=200)
plt.close('all')

print('calculating null distribution of PLV and awPLV values')
sample_per_min = int(intermediate_sample_rate * 60)
samples_per_tr = int(intermediate_sample_rate / sample_rate_fmri)
k = int((len(gastric_signal) - (2*sample_per_min)) / samples_per_tr)
plvs_permutation = np.zeros((len(plvs_empirical), k))
awplvs_permutation = np.zeros((len(awplvs_empirical), k))

for inx_permut in np.arange(k):
    # permut the gastric signal
    gastric_signal_permut = np.roll(gastric_signal, sample_per_min + int(inx_permut * samples_per_tr))

    # calc PLV with permuted signal
    gastric_phase_permut = to_phase_resampled(gastric_signal_permut,
                                              intermediate_sample_rate, sample_rate_fmri)
    plvs_permutation[:,inx_permut] = \
        np.abs(np.mean(np.exp(1j * (brain_signal_phase - gastric_phase_permut[np.newaxis, :])), axis=1))

    # calc awPLV with permuted signal
    # Resample permuted gastric signal
    gastric_time_orig_perm = np.arange(len(gastric_signal_permut)) / intermediate_sample_rate
    gastric_time_new_perm = np.arange(brain_signal.shape[1]) / sample_rate_fmri
    valid_idx_perm = gastric_time_new_perm <= gastric_time_orig_perm[-1]
    gastric_time_new_perm = gastric_time_new_perm[valid_idx_perm]
    interp_func_perm = interp1d(gastric_time_orig_perm, gastric_signal_permut, kind='cubic', fill_value='extrapolate')
    gastric_signal_permut_resampled = interp_func_perm(gastric_time_new_perm)

    # Match lengths
    min_length_perm = min(brain_signal.shape[1], len(gastric_signal_permut_resampled))
    brain_signal_perm = brain_signal[:, :min_length_perm]
    gastric_signal_perm = gastric_signal_permut_resampled[:min_length_perm]

    # Get analytic signals
    brain_analytic_perm = signal.hilbert(brain_signal_perm, axis=1)
    gastric_analytic_perm = signal.hilbert(gastric_signal_perm)

    # Extract amplitudes
    brain_amplitude_perm = np.abs(brain_analytic_perm)
    gastric_amplitude_perm = np.abs(gastric_analytic_perm)

    # Calculate amplitude weights
    amplitude_product_perm = brain_amplitude_perm * gastric_amplitude_perm[np.newaxis, :]
    sum_amplitude_product_perm = np.sum(amplitude_product_perm, axis=1, keepdims=True)
    sum_amplitude_product_perm[sum_amplitude_product_perm == 0] = 1.0
    weights_perm = amplitude_product_perm / sum_amplitude_product_perm

    # Compute awPLV
    phase_diff_perm = np.angle(brain_analytic_perm) - np.angle(gastric_analytic_perm)[np.newaxis, :]
    awplvs_permutation[:,inx_permut] = np.abs(np.sum(weights_perm * np.exp(1j * phase_diff_perm), axis=1))

# Compute all the relevant subject-level statistical maps
plv_p_vals = (plvs_permutation < plvs_empirical[:,np.newaxis]).mean(axis=1)
plv_permut_median = np.median(plvs_permutation,axis=1)
plv_delta = plvs_empirical - plv_permut_median

awplv_p_vals = (awplvs_permutation < awplvs_empirical[:,np.newaxis]).mean(axis=1)
awplv_permut_median = np.median(awplvs_permutation,axis=1)
awplv_delta = awplvs_empirical - awplv_permut_median

# Save PLV maps
for measure_name, measure in zip(['plv_p_vals', 'plv_delta', 'plv_permut_median', 'plvs_empirical'],
                                  [plv_p_vals, plv_delta, plv_permut_median, plvs_empirical]):
    vol_new = np.zeros(original_fmri.shape[:3])
    vol_new[mask] = measure
    img = nib.Nifti1Image(vol_new, affine = original_fmri.affine, header=original_fmri.header)
    nib.save(img, data_path + '/' + measure_name + '_' + subject_name + '_run' + run + clean_level + '.nii.gz')

    plotting.plot_stat_map(img, bg_img = MNI_tamplate_3mm, title=measure_name, colorbar = True,
                           threshold=np.percentile(plvs_empirical,95))
    plt.savefig(plot_path + 'thres95_' + measure_name + '_map.png', dpi=200)
    plt.close('all')

# Save awPLV maps
for measure_name, measure in zip(['awplv_p_vals', 'awplv_delta', 'awplv_permut_median', 'awplvs_empirical'],
                                  [awplv_p_vals, awplv_delta, awplv_permut_median, awplvs_empirical]):
    vol_new = np.zeros(original_fmri.shape[:3])
    vol_new[mask] = measure
    img = nib.Nifti1Image(vol_new, affine = original_fmri.affine, header=original_fmri.header)
    nib.save(img, data_path + '/' + measure_name + '_' + subject_name + '_run' + run + clean_level + '.nii.gz')

    plotting.plot_stat_map(img, bg_img = MNI_tamplate_3mm, title=measure_name, colorbar = True,
                           threshold=np.percentile(awplvs_empirical,95))
    plt.savefig(plot_path + 'thres95_' + measure_name + '_map.png', dpi=200)
    plt.close('all')

print(f'Done synchrony analysis for: {subject_name} run {run}')
print(f'Saved {k} permutations for PLV and awPLV null distributions')