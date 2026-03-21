#!/usr/bin/env python3
"""Evaluate all three rule-based detectors."""

import logging

from artifact_identification.evaluation.rule_based_eval import evaluate_rule_based


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting rule-based evaluation pipeline")

    models = [
        ('eye_movement', 'Eye Movement'),
        ('muscle_artifacts', 'Muscle Artifacts'),
        ('non_physiological', 'Non-Physiological'),
    ]

    for model_key, pretty_name in models:
        evaluate_rule_based(model_key, pretty_name)


if __name__ == "__main__":
    main()
