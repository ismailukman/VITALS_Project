import numpy as np
from nilearn.image import resample_to_img
import nibabel as nib
from nilearn.image import concat_imgs
import pandas as pd
import nipype.interfaces.fsl as fsl
from fsl.data import vest
import os
import sys
import pathlib
import argparse

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.resolve()))
os.chdir(os.path.dirname(pathlib.Path(__file__).parent.resolve()))
from config import clean_level, main_project_path

# Handle command line arguments
parser = argparse.ArgumentParser(description='Second-level voxel-based analysis with PLV and awPLV')
parser.add_argument('--metadata', default='dataframes/egg_brain_meta_data_10subjects.csv',
                    help='Metadata CSV file (default: egg_brain_meta_data_10subjects.csv)')
parser.add_argument('--include-awplv', action='store_true', default=True,
                    help='Include awPLV measures in analysis (default: True)')
args = parser.parse_args()

# Ensure FSL_DIR is set
if 'FSL_DIR' not in os.environ:
    print("Warning: FSL_DIR not set. Attempting to use /usr/local/fsl")
    os.environ['FSL_DIR'] = '/usr/local/fsl'

MNI_tamplate_path = os.environ['FSL_DIR'] + '/data/standard'
MNI_2mm_mask = nib.load(MNI_tamplate_path + '/MNI152_T1_2mm_brain_mask.nii.gz')

# load and filter the subjects list
print(f'Loading metadata from: {args.metadata}')
record_meta_pd = pd.read_csv(args.metadata)
if clean_level == 'strict_gs_cardiac':
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_exclude'] == False, :]
    record_meta_pd = record_meta_pd.loc[record_meta_pd['ppu_found'] == True, :]
subjects_dict = {}
for subject_name in record_meta_pd['subject'].unique():
    subjects_dict[subject_name] = record_meta_pd.loc[(record_meta_pd['subject'] == subject_name),
                                                     'run'].unique()

# Determine which measures to process
plv_measures = ['plv_delta', 'plv_permut_median', 'plvs_empirical']
awplv_measures = ['awplv_delta', 'awplv_permut_median', 'awplvs_empirical']
measures_to_process = plv_measures + (awplv_measures if args.include_awplv else [])

print(f'Processing {len(measures_to_process)} measures: {measures_to_process}')

# compute simple average for each measure
for measure_name in measures_to_process:
    for subject_index, subject_name in enumerate(subjects_dict.keys()):
        for run_index, run in enumerate(subjects_dict[subject_name]):
            run = str(run)  # Convert numpy.int64 to string
            data_path = f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' + subject_name+run
            img_plv = nib.load(f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' + subject_name + run +
                               '/' + measure_name + '_' + subject_name + '_run' + run + clean_level + '.nii.gz')
            if run_index == 0:
                imgs_plv_runs = np.zeros(np.concatenate([[len(subjects_dict[subject_name])], img_plv.shape]))
                if subject_index == 0:
                    MNI_mask_aligned = resample_to_img(MNI_2mm_mask, img_plv, interpolation='nearest')
                    imgs_plv_subjects = np.zeros(np.concatenate([[len(subjects_dict.keys())], img_plv.shape]))
            imgs_plv_runs[run_index,...] = img_plv.get_fdata()
        imgs_plv_runs = imgs_plv_runs.mean(axis=0)
        imgs_plv_runs[MNI_mask_aligned.get_fdata() == 0] = 0
        img_plv_avg = nib.Nifti1Image(imgs_plv_runs, affine=img_plv.affine, header=img_plv.header)
        nib.save(img_plv_avg, f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' +
                 measure_name + '_' + subject_name + '_mean_runs' + clean_level + '.nii.gz')
        imgs_plv_subjects[subject_index,...] = imgs_plv_runs
    imgs_plv_subjects = imgs_plv_subjects.mean(axis=0)
    img_plv_avg = nib.Nifti1Image(imgs_plv_subjects, affine=img_plv.affine, header=img_plv.header)
    nib.save(img_plv_avg, f'{main_project_path}/derivatives/brain_gast/' + measure_name + '_mean_subjects' + clean_level + '.nii.gz')

## perform Two-Sample Paired T-test - https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise/UserGuide
# Create design matrices once (shared for all measure types)
n_subjecs = subjects_dict.keys().__len__()
design_mat = np.zeros((n_subjecs*2, n_subjecs + 1))
design_mat[:n_subjecs, 0] = 1
design_mat[n_subjecs:, 0] = -1
design_mat[:n_subjecs, 1:] = np.eye(n_subjecs)
design_mat[n_subjecs:, 1:] = np.eye(n_subjecs)
with open(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.mat', 'w') as text_file:
    text_file.write(vest.generateVest(design_mat))

design_con = np.zeros(n_subjecs + 1)
design_con[0] = 1
with open(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.con', 'w') as text_file:
    text_file.write(vest.generateVest(design_con))

design_grp = np.concatenate([np.arange(n_subjecs), np.arange(n_subjecs)]) + 1
with open(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.grp', 'w') as text_file:
    text_file.write(vest.generateVest(design_grp[:, np.newaxis]))

# Process PLV measures
print('\nRunning FSL Randomise for PLV measures...')
image_list = []
for measure_name in ['plvs_empirical', 'plv_permut_median']:
    for subject_index, subject_name in enumerate(subjects_dict.keys()):
        image_list.append(nib.load(f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' +
                                   measure_name + '_' + subject_name + '_mean_runs' + clean_level + '.nii.gz'))

image_4d = concat_imgs(image_list)
nib.save(image_4d, f'{main_project_path}/derivatives/brain_gast/fsl_randomize/4d_plv.nii.gz')
nib.save(MNI_mask_aligned, f'{main_project_path}/derivatives/brain_gast/fsl_randomize/mask.nii.gz')

rand_plv = fsl.Randomise(in_file=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/4d_plv.nii.gz'),
                         mask=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/mask.nii.gz'),
                         tcon=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.con'),
                         design_mat=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.mat'),
                         x_block_labels=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.grp'),
                         base_name=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/result_plv' + clean_level + '_'),
                         vox_p_values=True, c_thresh=2.3, cm_thresh=2.3, tfce=True,
                         num_perm=10000)
print(rand_plv.cmdline)
rand_plv.run()

# Process awPLV measures if requested
if args.include_awplv:
    print('\nRunning FSL Randomise for awPLV measures...')
    image_list = []
    for measure_name in ['awplvs_empirical', 'awplv_permut_median']:
        for subject_index, subject_name in enumerate(subjects_dict.keys()):
            image_list.append(nib.load(f'{main_project_path}/derivatives/brain_gast/' + subject_name + '/' +
                                       measure_name + '_' + subject_name + '_mean_runs' + clean_level + '.nii.gz'))

    image_4d = concat_imgs(image_list)
    nib.save(image_4d, f'{main_project_path}/derivatives/brain_gast/fsl_randomize/4d_awplv.nii.gz')

    rand_awplv = fsl.Randomise(in_file=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/4d_awplv.nii.gz'),
                               mask=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/mask.nii.gz'),
                               tcon=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.con'),
                               design_mat=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.mat'),
                               x_block_labels=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/design.grp'),
                               base_name=os.path.abspath(f'{main_project_path}/derivatives/brain_gast/fsl_randomize/result_awplv' + clean_level + '_'),
                               vox_p_values=True, c_thresh=2.3, cm_thresh=2.3, tfce=True,
                               num_perm=10000)
    print(rand_awplv.cmdline)
    rand_awplv.run()

print('\nDone running the second level analysis code.')