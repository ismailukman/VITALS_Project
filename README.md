# VITALS Gastric-Brain Synchrony Analysis

This repository contains the EGG preprocessing pipeline and EGG-Brain synchrony analysis for the VITALS project, investigating phase coupling between the gastric slow wave and brain BOLD signals.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [EGG Preprocessing](#egg-preprocessing)
  - [Data Organization](#data-organization)
  - [Metadata File Format](#metadata-file-format)
  - [Usage](#usage)
  - [Processing Pipeline](#processing-pipeline)
  - [Output Files](#output-files)
- [ROI-Level Synchrony Analysis](#roi-level-synchrony-analysis)
  - [Atlas](#atlas)
  - [Synchrony Metrics](#synchrony-metrics)
  - [Statistical Testing](#statistical-testing)
  - [Running the Analysis](#running-the-analysis)
  - [ROI Analysis Output](#roi-analysis-output)
- [Increasing the Data for Synchrony Analysis](#increasing-the-data-for-synchrony-analysis)
- [Acknowledgments](#acknowledgments)

---

## Overview

The project has two main components:

1. **EGG Preprocessing** (`preprocess_egg_data/`): Reads raw EGG data from AcqKnowledge/Biopac format, aligns with fMRI acquisition, identifies the dominant gastric frequency, and outputs cleaned filtered signals.

2. **ROI-Level Synchrony Analysis** (`roi_analysis/`): Computes phase locking value (PLV) and amplitude-weighted PLV (awPLV) between the gastric rhythm and brain ROI time series extracted using a 132-ROI composite atlas.

---

## Requirements

### Python Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
mne>=0.24.0
bioread>=3.0.0
scikit-learn>=0.24.0
```

### Installation

```bash
# Using conda (recommended)
conda activate brain_gut

# Or install dependencies manually
pip install numpy scipy pandas matplotlib mne bioread scikit-learn nibabel nilearn seaborn
```

---

## Project Structure

```
VITALS_Gastric_Brain_Synchrony/
├── README.md
├── .gitignore
├── preprocess_egg_data/          # EGG preprocessing pipeline
│   ├── config.py                 # Central configuration
│   ├── preprocess_gastric_data.py
│   ├── egg_brain_metadata.csv    # Subject metadata
│   ├── requirements.txt
│   └── utils/
│       ├── gastric_utils.py
│       └── spect_utils.py
├── roi_analysis/                 # ROI-level EGG-brain synchrony
│   ├── roi_based_analysis.py     # Main analysis script
│   └── output/                   # Results (gitignored)
│       ├── roi_synchrony_results.csv
│       ├── roi_summary_statistics.csv
│       └── plots/
├── egg_data/                     # Raw EGG .acq files (gitignored)
├── atlas/                        # Brain atlas files (gitignored)
│   ├── atlas.nii                 # 132-ROI composite atlas
│   ├── atlas.txt                 # ROI labels
│   └── atlas.groups.info         # 22 network groupings
└── subject_data_nii/             # Preprocessed fMRI NIfTI (gitignored)
```

---

## EGG Preprocessing

### Data Organization

#### Input Data Structure

Place your raw EGG data in the `egg_data/` folder following this structure:

```
egg_data/
├── VITD0107_Acq/
│   └── VITD0107_EGG.acq
├── VITD0126_Acq/
│   └── VITD0126_EGG.acq
├── VITD0128_Acq/
│   └── VITD0128_EGG.acq
└── ...
```

#### Expected .acq File Contents

Each `.acq` file should contain:

- **Channels 0-3**: EGG electrode signals (4 channels)
- **Channel 4**: EDA signal (optional)
- **Channels 5-6**: Digital trigger signals (MRI trigger on channel 6, configurable in `config.py`)

---

### Metadata File Format

Create a metadata CSV file (`egg_metadata.csv`) with the following columns:

| Column                 | Description                   | Example Values                       |
| ---------------------- | ----------------------------- | ------------------------------------ |
| `subject`            | Subject identifier            | `sub-01`, `sub-02`               |
| `run`                | Run number                    | `1`, `2`                         |
| `mri_length`         | fMRI scan duration in seconds | `600`                              |
| `num_channles`       | Number of EGG channels        | `4`                                |
| `trigger_start`      | Trigger detection mode        | `auto` or seconds (e.g., `10.5`) |
| `dominant_channel`   | Which channel to use          | `auto` or channel index (0-3)      |
| `dominant_frequency` | Gastric frequency             | `auto` or Hz (e.g., `0.05`)      |

### Example Metadata File

```csv
subject,run,mri_length,num_channles,trigger_start,dominant_channel,dominant_frequency
VITD0107,1,600,4,0,auto,auto
VITD0126,1,547,4,0,auto,auto
VITD0128,1,600,4,45,auto,auto
```

The metadata file `egg_metadata.csv` is located in the `preprocess_egg_data/` folder.

---

### Usage

#### Single Subject Processing

Process one subject/run at a time:

```bash
cd preprocess_egg_data

# Basic usage
python preprocess_gastric_data.py VITD0107 1

# This will:
# 1. Load EGG data from egg_data/VITD0107_Acq/VITD0107_EGG.acq
# 2. Process the signal
# 3. Save outputs to output/derivatives/VITD0107/VITD01071/
# 4. Save plots to output/plots/VITD0107/VITD01071/
```

**Example Output:**

```
============================================================
Processing subject: VITD0107, run: 1
============================================================
Reading EGG file: egg_data/VITD0107_Acq/VITD0107_EGG.acq
Original sample rate: 2000.0 Hz
MRI duration: 600 seconds
Number of EGG channels: 4
Dominant frequency: 0.0480 Hz
Dominant channel: 1

Output files saved:
  - Data: output/derivatives/VITD0107/VITD01071/gast_data_VITD0107_run1strict.npy
  - Frequency: output/derivatives/VITD0107/VITD01071/max_freqVITD0107_run1strict.npy
  - Plots: output/plots/VITD0107/VITD01071/
```

#### Batch Processing

Process all subjects defined in the metadata file:

```bash
cd preprocess_egg_data

# Process all subjects (uses default metadata file from config)
python preprocess_gastric_data.py --batch

# Specify number of parallel jobs
python preprocess_gastric_data.py --batch --jobs 4

# Sequential processing (1 job)
python preprocess_gastric_data.py --batch --jobs 1
```

**Example Batch Output:**

```
============================================================
BATCH PROCESSING MODE
============================================================
Metadata file: preprocess_egg_data/egg_metadata.csv
Parallel jobs: 8
Total subjects/runs to process: 3
============================================================

[SUCCESS] VITD0107 run 1
[SUCCESS] VITD0126 run 1
[SUCCESS] VITD0128 run 1

============================================================
BATCH PROCESSING COMPLETE
============================================================
Successful: 3/3
Failed: 0/3
```

---

### Processing Pipeline

The preprocessing pipeline consists of the following steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EGG PREPROCESSING PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA LOADING
        ├── Read .acq file (Biopac/AcqKnowledge format)
        ├── Extract EGG channels (typically 4)
        └── Extract trigger channel

Step 2: TRIGGER DETECTION
        ├── Detect MRI trigger onset (auto or manual)
        └── Define recording segment aligned with fMRI

Step 3: SIGNAL SLICING
        ├── Extract EGG segment matching fMRI duration
        └── Generate sliced signal plot

Step 4: RESAMPLING
        ├── Downsample from original rate (e.g., 1000 Hz)
        └── to intermediate rate (10 Hz)

Step 5: SPECTRAL ANALYSIS (Welch PSD)
        ├── Compute power spectral density per channel
        ├── Identify peaks in normogastric range (0.033-0.066 Hz)
        ├── Select dominant channel (highest power peak)
        └── Determine dominant gastric frequency

Step 6: BANDPASS FILTERING
        ├── Apply FIR filter (Hamming window)
        ├── Passband: [dominant_freq ± 0.015 Hz]
        └── Zero-phase filtering (no phase distortion)

Step 7: NORMALIZATION
        └── Z-score standardization

Step 8: OUTPUT
        ├── Save filtered signal (.npy)
        ├── Save dominant frequency (.npy)
        └── Save diagnostic plots (.png)
```

---

### Output Files

#### Directory Structure

After processing, the output folder will contain:

```
preprocess_egg_data/
└── output/
    ├── derivatives/
    │   ├── VITD0107/
    │   │   └── VITD01071/
    │   │       ├── gast_data_VITD0107_run1strict.npy    # Filtered EGG signal
    │   │       └── max_freqVITD0107_run1strict.npy      # Dominant frequency
    │   ├── VITD0126/
    │   │   └── VITD01261/
    │   │       └── ...
    │   └── VITD0128/
    │       └── ...
    └── plots/
        ├── VITD0107/
        │   └── VITD01071/
        │       ├── trigger_cut_VITD0107_1.png              # Trigger detection
        │       ├── sliced_signalVITD0107_1.png             # Raw sliced signal
        │       ├── post_first_resample_VITD0107_1.png      # After resampling
        │       ├── egg_power_spectral_density_VITD0107_1.png  # PSD plot
        │       └── egg_filteredVITD0107_1.png              # Final filtered signal
        └── ...
```

#### Output Data Format

**Filtered Signal** (`gast_data_*.npy`):

- 1D NumPy array
- Sampling rate: 10 Hz (configurable)
- Duration matches fMRI scan
- Z-score normalized (if enabled)

```python
import numpy as np

# Load preprocessed EGG signal
egg_signal = np.load('output/derivatives/VITD0107/VITD01071/gast_data_VITD0107_run1strict.npy')
print(f"Signal shape: {egg_signal.shape}")  # e.g., (6000,) for 600s at 10Hz
print(f"Signal range: [{egg_signal.min():.2f}, {egg_signal.max():.2f}]")
```

**Dominant Frequency** (`max_freq*.npy`):

- Single float value in Hz
- Typically in range 0.033-0.066 Hz

```python
# Load dominant gastric frequency
freq = np.load('output/derivatives/VITD0107/VITD01071/max_freqVITD0107_run1strict.npy')
print(f"Dominant frequency: {freq:.4f} Hz ({freq*60:.2f} cycles/min)")
```

---

---

## ROI-Level Synchrony Analysis

### Atlas

The analysis uses the CONN toolbox default 132-ROI composite atlas (`atlas/atlas.nii`), comprising:

| Source Atlas | ROIs | Description |
|---|---|---|
| Harvard-Oxford Cortical | 91 | Bilateral cortical areas split into L/R |
| Harvard-Oxford Subcortical | 15 | Subcortical structures (excluding WM/CSF) |
| AAL Cerebellar | 26 | Cerebellar parcellation |
| **Total** | **132** | |

### Synchrony Metrics

For each subject, the script:
1. Extracts mean ROI time series from fMRI using the atlas
2. Bandpass-filters both brain and EGG signals around the subject-specific gastric frequency (~0.05 Hz)
3. Computes two synchrony metrics between the gastric rhythm and each ROI:

- **PLV (Phase Locking Value)**: Consistency of phase difference over time. Range [0, 1].
- **awPLV (Amplitude-Weighted PLV)**: PLV weighted by instantaneous signal amplitude. Captures synchrony that co-varies with signal strength.

### Statistical Testing

- **Null distribution**: Mismatch approach — each subject's brain data is paired with other subjects' EGG signals to create null PLV/awPLV values.
- **Per-ROI p-values**: Proportion of null values ≥ empirical value.
- **Multiple comparison correction**: FDR (Benjamini-Hochberg) across all ROIs within each subject.
- **Pooled test**: Mann-Whitney U (one-sided) comparing all empirical vs all null values across subjects.

### Running the Analysis

```bash
conda activate brain_gut
cd /path/to/VITALS_Gastric_Brain_Synchrony

# Ensure EGG preprocessing is done first
python preprocess_egg_data/preprocess_gastric_data.py --batch

# Run ROI synchrony analysis
python roi_analysis/roi_based_analysis.py
```

**Prerequisites**: Raw fMRI NIfTI files in `subject_data_nii/` and atlas files in `atlas/` (both gitignored due to size).

### ROI Analysis Output

```
roi_analysis/output/
├── roi_synchrony_results.csv       # Per-subject, per-ROI PLV/awPLV with p-values
├── roi_summary_statistics.csv      # Mean PLV/awPLV per ROI across subjects
└── plots/
    ├── roi_density_grid_top20.png  # Paired PLV+awPLV density for top 20 ROIs
    ├── roi_pooled_density.png      # Pooled empirical vs null distributions
    ├── roi_synchrony_heatmap.png   # PLV/awPLV heatmap (ROI × subject)
    ├── roi_significance_barplot.png # Top ROIs by significance count
    └── roi_distribution.png        # Histogram of all PLV/awPLV values
```

**CSV columns**: `subject`, `run`, `roi_index`, `roi_name`, `plv_empirical`, `plv_null_mean`, `plv_delta`, `p_value_plv`, `p_fdr_plv`, `sig_fdr_plv`, `awplv_empirical`, `awplv_null_mean`, `awplv_delta`, `p_value_awplv`, `p_fdr_awplv`, `sig_fdr_awplv`, `gastric_freq`

---

## Increasing the Data for Synchrony Analysis

The current analysis uses 3 subjects with concurrent EGG and resting-state fMRI. To improve statistical power and reliability:

### 1. Add More Subjects
- The mismatch null distribution currently has only 2 null samples per subject (N-1 other subjects). With more subjects, the null becomes richer and p-values more granular (currently limited to {0.0, 0.5, 1.0}).
- **Minimum recommended**: 10+ subjects for meaningful FDR-corrected inference; 20+ for robust group-level statistics.
- To add a new subject: place the raw `.acq` in `egg_data/`, the preprocessed fMRI NIfTI in `subject_data_nii/`, add a row to `egg_brain_metadata.csv`, and add the subject ID to `SUBJECTS_WITH_BRAIN_DATA` in the analysis script.

### 2. Include Multiple Runs per Subject
- If subjects have multiple fMRI runs with concurrent EGG, each run can be processed independently. Update the metadata CSV with separate rows per run. This increases total observations and strengthens within-subject reliability.

### 3. Surrogate-Based Null Distribution
- Instead of relying solely on the mismatch null (limited by N-1 pairs), implement **phase-shuffled surrogates**: randomly shift the gastric phase time series by a circular permutation (preserving autocorrelation) and recompute PLV/awPLV. This can generate hundreds or thousands of null samples per subject regardless of sample size, yielding finer-grained p-values.

### 4. Task-Based Paradigms
- The current analysis uses resting-state data. If EGG was also recorded during task fMRI (e.g., MID task, movie watching), those runs can be analyzed separately to examine how gastric-brain coupling changes across conditions.

---

## Acknowledgments

- EGG preprocessing pipeline based on [Levakov et al. 2023](https://github.com/GidLev/brain_gastric_synchronization_2023/tree/master)
- Atlas: CONN toolbox default atlas — Harvard-Oxford (Desikan et al. 2006; Frazier et al. 2005) + AAL Cerebellar (Tzourio-Mazoyer et al. 2002)
