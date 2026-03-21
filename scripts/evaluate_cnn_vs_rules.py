#!/usr/bin/env python3
"""Run CNN vs rule-based comparison for all artifact types."""

import logging

from artifact_identification.evaluation.cnn_vs_rules import evaluate_model


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive CNN vs rule-based comparison")

    evaluations = [
        ('eye_movement', 'results/edl_cnn_eye/edl_cnn_eye', 'Eye Movement Artifacts'),
        ('muscle_artifacts', 'results/edl_cnn_muscle/edl_cnn_muscle', 'Muscle/EMG Artifacts'),
        ('non_physiological', 'results/edl_cnn_non_physiological/edl_cnn_non_physiological', 'Non-Physiological Artifacts'),
    ]

    for model_key, model_prefix, pretty_name in evaluations:
        try:
            evaluate_model(model_key, model_prefix, pretty_name)
        except Exception as e:
            logger.error(f"Evaluation failed for {pretty_name}: {e}")


if __name__ == '__main__':
    main()
