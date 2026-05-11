#!/usr/bin/env python3
"""Train and evaluate the muscle artifact detector."""

import json
import logging
import os

from artifact_identification.detectors.muscle import MuscleArtifactDetector


def _save_thresholds(detector: MuscleArtifactDetector, evaluation: dict) -> None:
    """Persist the detector operating point and input shape for inference use."""
    payload = {
        "model_name": detector.model_name,
        "selected_mode": evaluation["mode"],
        "selected_threshold": float(evaluation["youden_threshold"]),
        "f1_optimal_threshold": float(evaluation["threshold"]),
        "sensitivity": float(evaluation["sensitivity"]),
        "specificity": float(evaluation["specificity"]),
        "input_shape_3d": detector.metadata.get("input_shape_3d"),
        "n_timepoints": int(detector.metadata["n_timepoints"]),
        "n_channels": int(detector.metadata["n_channels"]),
    }
    path = os.path.join(detector.results_dir, f"{detector.model_name}_thresholds.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Initializing Muscle Artifact Detector")

    detector = MuscleArtifactDetector(model_name="edl_cnn_muscle", verbose=True)
    detector.load_data()
    detector.build_model(model_type='lightweight')
    detector.train(epochs=200, batch_size=128)
    evaluation = detector.evaluate()
    _save_thresholds(detector, evaluation)
    detector.plot_training_history()
    detector.save_model()

    logger.info("Muscle artifact detector pipeline completed.")


if __name__ == "__main__":
    main()
