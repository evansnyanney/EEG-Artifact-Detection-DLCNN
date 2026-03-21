"""
artifact_identification - EEG Artifact Detection with DLCNN and Rule-Based Methods
=====================================================================================

This package implements end-to-end EEG artifact detection using a Deep
Lightweight 1D Convolutional Neural Network (DLCNN) together with literature-based
rule-based methods. It targets three artifact categories derived from TUH annotations:

- **Eye movements** (EYE)
- **Muscle artifacts** (EMG, CHEW, SHIV)
- **Non-physiological artifacts** (ELEC, ELPP)

Modules
-------
preprocessing
    EEG preprocessing pipeline (resampling, montage, filtering, windowing).
data_preparation
    Binary dataset preparation for per-target model training.
detectors
    DLCNN detectors and rule-based heuristic detectors.
evaluation
    CNN vs rule-based comparison and standalone rule-based evaluation.
exploration
    Dataset exploration and quality analysis.
utils
    EDF file inspection utilities.
losses
    Shared focal loss function for imbalanced classification.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

from ._version import __version__

from .preprocessing import EEGPreprocessingPipeline
from .data_preparation import BinaryModelDataPreparer
from .losses import focal_loss_with_class_weights
from .exploration import EEGExplorer

__all__ = [
    '__version__',
    'EEGPreprocessingPipeline',
    'BinaryModelDataPreparer',
    'focal_loss_with_class_weights',
    'EEGExplorer',
]
