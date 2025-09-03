# Deep Lightweight CNN for EEG Artifact Detection

This repository implements end-to-end EEG artifact detection using an Enhanced Deep Lightweight 1D Convolutional Neural Network (EDL-CNN), together with literature-based rule methods. It targets three artifact categories derived from TUH annotations:

- Eye movements
- Muscle (EMG) artifacts
- Non-physiological artifacts (e.g., electrode noise/pops)

The pipeline includes preprocessing, binary dataset preparation per target, model training, threshold calibration on validation data, final evaluation on held-out test data, optional window-size sweeps, and comparison against rule-based detectors.

## Installation

```bash
pip install -r requirements.txt
```

Data note: raw TUH data and large artifacts are excluded from version control. Download TUH separately and place the required EDF files and CSV annotations under a local folder named `edf/` (create it if needed). Any substructure is acceptable as long as file paths used in your environment are consistent.

## Repository Structure (selected)

```
data_exploration.py                 # Dataset characteristics and validation
eeg_preprocessing_pipeline.py       # Preprocessing and windowing (non-overlapping)
binary_model_data_preparation.py    # Build binary datasets per target
eye_movement_detector.py            # EDL-CNN training/evaluation (eye)
muscle_artifact_detector.py         # EDL-CNN training/evaluation (muscle)
non_physiological_detector.py       # EDL-CNN training/evaluation (non-phys)
rule_based_detectors.py             # Literature-based rule detectors
evaluate_cnn_vs_rules.py            # CNN vs Rules comparison
window_optimization_script.py       # Window-size sweep (optional)
check_channels.py, check_edf.py     # Utilities
checkpoints/                        # Model weights (ignored by Git)
```

## Methodological Summary

- Sampling rate: 250 Hz; standardized channel set
- Windows: non-overlapping; size is configurable (e.g., 1–30 s)
- Split: 60/20/20 at the patient/recording level to prevent leakage
- Normalization: robust scaling
- Threshold calibration (validation set): Youden, fixed specificity, or max TPR at FPR ≤ 0.1
- Metrics (reported on test set): sensitivity, specificity, ROC AUC, prevalence-adjusted PR-AUC, partial ROC AUC at FPR ≤ 0.1
- Rule-based detectors: literature-adapted bandpower, spectral slope, amplitude/variance, and line-noise features with adaptive thresholds

## Typical Workflow

1) Preprocess and window the data (non-overlapping):
```bash
python eeg_preprocessing_pipeline.py --window-seconds 3 --overlap 0.0
```

2) Build binary datasets for each target:
```bash
python binary_model_data_preparation.py
```

3) Train a detector (repeat per target as needed):
```bash
python eye_movement_detector.py
python muscle_artifact_detector.py
python non_physiological_detector.py
```

4) Compare CNN to rule-based methods:
```bash
python evaluate_cnn_vs_rules.py
```

5) Optional: Sweep window sizes and summarize:
```bash
python window_optimization_script.py --target all --force
```

## Notes on Models and Checkpoints

- Trained weights are saved under `checkpoints/<target>/` with unique timestamps.
- Checkpoints are ignored by Git to keep the repository small.

## Metrics and Reporting

Detectors report: accuracy, precision, recall (sensitivity), specificity, F1, ROC AUC, PR AUC, prevalence-adjusted PR AUC, and partial ROC AUC (FPR ≤ 0.1). Thresholds are selected on the validation set and applied to the held-out test set.

Plots saved per run include training history, ROC/PR curves, confusion matrix, and prediction distributions.

## Citation

If this repository is useful in your work, please cite:

```
Nyanney E., Geng Z., Thirumala P. (2025). EEG Artifact Detection with EDL-CNN
and Literature-Based Rules. GitHub repository: https://github.com/ZhaohuiGeng/artifact_identification
```

For data, please acknowledge the Temple University Hospital EEG Corpus (TUH).

## License

MIT License. Ensure compliance with TUH dataset usage terms.