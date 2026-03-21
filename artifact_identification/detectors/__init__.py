"""
DLCNN artifact detectors for EEG signals.

Each detector implements a lightweight 1D-CNN for binary classification
of a specific artifact type.
"""

from .eye_movement import EyeMovementDetector
from .muscle import MuscleArtifactDetector
from .non_physiological import NonPhysiologicalDetector
from .rule_based import (
    detect_eye_movement_tuh_adapted,
    detect_muscle_artifacts_tuh_adapted,
    detect_non_physiological_tuh_adapted,
    run_rules,
)

__all__ = [
    'EyeMovementDetector',
    'MuscleArtifactDetector',
    'NonPhysiologicalDetector',
    'detect_eye_movement_tuh_adapted',
    'detect_muscle_artifacts_tuh_adapted',
    'detect_non_physiological_tuh_adapted',
    'run_rules',
]
