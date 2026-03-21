#!/usr/bin/env python3
"""Prepare binary classification datasets for each artifact type."""

import logging

from artifact_identification.data_preparation import BinaryModelDataPreparer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting binary model data preparation")

    preparer = BinaryModelDataPreparer(input_dir='binary_models_data', output_dir='binary_models_data')
    preparer.load_data()
    preparer.prepare_all_models()
    preparer.analyze_data()

    print("\n" + "=" * 70)
    print("BINARY MODEL DATA PREPARATION COMPLETED")
    print("=" * 70)
    print("Three binary models prepared for training:")
    print("  1. Eye Movement Detector")
    print("  2. Muscle Artifact Detector")
    print("  3. Non-Physiological Detector")
    print("=" * 70)


if __name__ == "__main__":
    main()
