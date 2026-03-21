#!/usr/bin/env python3
"""Train and evaluate the eye movement artifact detector."""

import logging

from artifact_identification.detectors.eye_movement import EyeMovementDetector


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Initializing Eye Movement Detector")

    detector = EyeMovementDetector(model_name="edl_cnn_eye", verbose=True)
    detector.load_data()
    detector.build_model(model_type='lightweight')
    detector.train(epochs=200, batch_size=128)
    detector.evaluate()
    detector.plot_training_history()
    detector.save_model()

    logger.info("Eye movement detection pipeline completed.")


if __name__ == "__main__":
    main()
