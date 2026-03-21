#!/usr/bin/env python3
"""Train and evaluate the muscle artifact detector."""

import logging

from artifact_identification.detectors.muscle import MuscleArtifactDetector


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Initializing Muscle Artifact Detector")

    detector = MuscleArtifactDetector(model_name="edl_cnn_muscle", verbose=True)
    detector.load_data()
    detector.build_model(model_type='lightweight')
    detector.train(epochs=200, batch_size=128)
    detector.evaluate()
    detector.plot_training_history()
    detector.save_model()

    logger.info("Muscle artifact detector pipeline completed.")


if __name__ == "__main__":
    main()
