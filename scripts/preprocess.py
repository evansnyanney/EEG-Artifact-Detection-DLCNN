#!/usr/bin/env python3
"""Run the EEG preprocessing pipeline."""

import argparse
import logging
import multiprocessing as mp

from artifact_identification.preprocessing import EEGPreprocessingPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='EEG preprocessing pipeline')
    parser.add_argument('--window-seconds', type=float, default=1.0, help='Window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.0, help='Window overlap fraction')
    parser.add_argument('--edf-dir', type=str, default='edf/01_tcp_ar', help='EDF data directory')
    parser.add_argument('--max-files', type=int, default=150, help='Max files to process')
    parser.add_argument('--output', type=str, default='binary_models_data', help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting EEG preprocessing pipeline")

    pipeline = EEGPreprocessingPipeline(
        target_sampling_rate=250,
        target_channels=22,
        window_size=args.window_seconds,
        overlap=args.overlap,
        min_artifact_duration=0.1,
        include_clean_windows=True,
    )

    n_jobs = max(1, mp.cpu_count() - 1)
    results = pipeline.process_dataset(edf_dir=args.edf_dir, max_files=args.max_files, n_jobs=n_jobs)
    data_dict = pipeline.prepare_training_data(results)
    pipeline.save_data(data_dict, results, output_path=args.output)

    logger.info("Preprocessing completed successfully.")
    logger.info(f"Total windows: {results['total_windows']}, Classes: {data_dict['n_classes']}")


if __name__ == "__main__":
    main()
