#!/usr/bin/env python3
"""Run comprehensive EEG data exploration."""

import logging

from artifact_identification.exploration import EEGExplorer


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive EEG data exploration")

    explorer = EEGExplorer()
    explorer.analyze_edf_files()
    explorer.analyze_csv_annotations()
    explorer.analyze_file_matching()
    explorer.analyze_artifact_patterns()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EEG DATA ANALYSIS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
