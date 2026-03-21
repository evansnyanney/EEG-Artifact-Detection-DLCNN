"""Tests for rule-based artifact detectors."""

import numpy as np
import pytest

from artifact_identification.detectors.rule_based import (
    detect_eye_movement_tuh_adapted,
    detect_muscle_artifacts_tuh_adapted,
    detect_non_physiological_tuh_adapted,
    run_rules,
)


def _make_random_eeg(n_windows=50, n_timepoints=250, n_channels=22, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n_windows, n_timepoints, n_channels).astype(np.float32)


class TestRuleBasedDetectors:
    """Smoke tests for rule-based detectors."""

    def test_eye_detector_returns_correct_shape(self):
        X = _make_random_eeg(30)
        labels = detect_eye_movement_tuh_adapted(X, fs=250.0)
        assert labels.shape == (30,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_muscle_detector_returns_correct_shape(self):
        X = _make_random_eeg(30)
        labels = detect_muscle_artifacts_tuh_adapted(X, fs=250.0)
        assert labels.shape == (30,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_nonphys_detector_returns_correct_shape(self):
        X = _make_random_eeg(30)
        labels = detect_non_physiological_tuh_adapted(X, fs=250.0)
        assert labels.shape == (30,)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_run_rules_eye(self):
        X = _make_random_eeg(20)
        labels = run_rules(X, 'eye')
        assert labels.shape == (20,)

    def test_run_rules_muscle(self):
        X = _make_random_eeg(20)
        labels = run_rules(X, 'muscle')
        assert labels.shape == (20,)

    def test_run_rules_nonphys(self):
        X = _make_random_eeg(20)
        labels = run_rules(X, 'nonphys')
        assert labels.shape == (20,)

    def test_run_rules_aliases(self):
        X = _make_random_eeg(10)
        for alias in ['eye_movement', 'muscle_artifacts', 'non_physiological']:
            labels = run_rules(X, alias)
            assert labels.shape == (10,)

    def test_run_rules_invalid_target(self):
        X = _make_random_eeg(10)
        with pytest.raises(ValueError, match="Unknown target"):
            run_rules(X, 'invalid_target')


class TestPreprocessingImport:
    """Verify core modules are importable."""

    def test_import_package(self):
        import artifact_identification
        assert hasattr(artifact_identification, '__version__')

    def test_import_preprocessing(self):
        from artifact_identification import EEGPreprocessingPipeline
        assert EEGPreprocessingPipeline is not None

    def test_import_data_preparation(self):
        from artifact_identification import BinaryModelDataPreparer
        assert BinaryModelDataPreparer is not None

    def test_import_focal_loss(self):
        from artifact_identification import focal_loss_with_class_weights
        assert callable(focal_loss_with_class_weights)

    def test_import_detectors(self):
        from artifact_identification.detectors import (
            EyeMovementDetector,
            MuscleArtifactDetector,
            NonPhysiologicalDetector,
        )
        assert EyeMovementDetector is not None
        assert MuscleArtifactDetector is not None
        assert NonPhysiologicalDetector is not None
