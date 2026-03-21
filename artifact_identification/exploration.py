#!/usr/bin/env python3
"""
EEG Data Exploration and Analysis

Comprehensive analysis of EEG recordings (EDF) and artifact annotations (CSV)
to support preprocessing, model development, and data quality assessment.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import glob
import logging
from typing import List, Tuple, Dict, Any, DefaultDict
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import mne

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['EEGExplorer']

DEFAULT_EDF_DIR = 'edf/01_tcp_ar'
PROGRESS_INTERVAL = 50
MAX_UNMATCHED_DISPLAY = 10


class EEGExplorer:
    """
    Comprehensive EEG data explorer for EDF recordings and CSV annotations.

    Analyses:
    - EDF file inspection (durations, sampling rates, channels)
    - CSV annotation analysis (artifact types, durations, channels)
    - File matching between EDF and CSV
    - Artifact pattern detection (combinations, timing, co-occurrence)

    Args:
        edf_dir: Path to directory containing EDF and CSV files.
    """

    def __init__(self, edf_dir: str = DEFAULT_EDF_DIR) -> None:
        self.edf_dir = edf_dir
        logger.info(f"Initialized EEGExplorer with directory: {self.edf_dir}")

    def analyze_edf_files(self) -> Dict[str, Any]:
        """Analyze all EDF files to extract recording characteristics."""
        logger.info("Analyzing EDF files (EEG recordings)")
        edf_files = glob.glob(os.path.join(self.edf_dir, "*.edf"))
        logger.info(f"Found {len(edf_files)} EDF files")

        if not edf_files:
            logger.error("No EDF files found.")
            return {}

        durations, sampling_rates, channel_counts = [], [], []
        data_ranges, means, stds = [], [], []
        successful = 0

        for i, fp in enumerate(edf_files):
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(edf_files)} EDF files")
            try:
                raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
                durations.append(raw.times[-1])
                sampling_rates.append(raw.info['sfreq'])
                channel_counts.append(len(raw.ch_names))
                data, _ = raw[:, :]
                data_ranges.append(data.max() - data.min())
                means.append(data.mean())
                stds.append(data.std())
                successful += 1
            except Exception as e:
                logger.warning(f"Failed to load {os.path.basename(fp)}: {e}")

        arrays = {k: np.array(v) for k, v in [
            ('durations', durations), ('sampling_rates', sampling_rates),
            ('channel_counts', channel_counts), ('data_ranges', data_ranges),
            ('means', means), ('stds', stds),
        ]}
        arrays.update({'successful_files': successful, 'total_files': len(edf_files)})

        self._print_edf_summary(**{k: arrays[k] for k in [
            'durations', 'sampling_rates', 'channel_counts', 'data_ranges', 'means', 'stds'
        ]})
        return arrays

    def _print_edf_summary(self, durations, sampling_rates, channel_counts, data_ranges, means, stds):
        print("\n" + "=" * 60)
        print("EDF FILES STATISTICS SUMMARY")
        print("=" * 60)

        for name, arr in [("Duration (s)", durations), ("Amplitude Range", data_ranges),
                          ("Mean", means), ("Std", stds)]:
            print(f"\n{name}:")
            print(f"   Mean: {arr.mean():.2f}, Median: {np.median(arr):.2f}, "
                  f"Std: {arr.std():.2f}, Range: [{arr.min():.2f}, {arr.max():.2f}]")

        print(f"\nSampling Rates (Hz): {np.unique(sampling_rates).tolist()}")
        print(f"Channel Counts: {sorted(np.unique(channel_counts).tolist())}")

    def analyze_csv_annotations(self) -> Dict[str, Any]:
        """Analyze all CSV annotation files for artifact distribution."""
        logger.info("Analyzing CSV annotation files")
        csv_files = glob.glob(os.path.join(self.edf_dir, "*.csv"))

        if not csv_files:
            logger.error("No CSV files found.")
            return {}

        all_labels, all_channels, all_durations = [], [], []
        annotation_counts, all_annotations = [], []
        required_cols = {'label', 'channel', 'start_time', 'stop_time'}

        for fp in csv_files:
            try:
                df = pd.read_csv(fp, comment='#')
                if not required_cols.issubset(df.columns):
                    continue
                all_labels.extend(df['label'].tolist())
                all_channels.extend(df['channel'].tolist())
                all_durations.extend((df['stop_time'] - df['start_time']).tolist())
                annotation_counts.append(len(df))
                all_annotations.append(df)
            except Exception as e:
                logger.warning(f"Failed: {os.path.basename(fp)}: {e}")

        return {
            'annotation_counts': np.array(annotation_counts),
            'durations': np.array(all_durations),
            'labels': all_labels, 'channels': all_channels,
            'all_annotations': all_annotations,
        }

    def analyze_file_matching(self) -> Dict[str, Any]:
        """Analyze correspondence between EDF and CSV files."""
        edf_files = glob.glob(os.path.join(self.edf_dir, "*.edf"))
        matched = sum(1 for e in edf_files if os.path.exists(os.path.splitext(e)[0] + ".csv"))
        return {'matched_pairs': matched, 'total_edf': len(edf_files)}

    def analyze_artifact_patterns(self) -> Dict[str, Any]:
        """Analyze artifact co-occurrence, timing, and channel patterns."""
        csv_files = glob.glob(os.path.join(self.edf_dir, "*.csv"))
        combinations: List[tuple] = []
        channel_patterns: DefaultDict[str, Counter] = defaultdict(Counter)

        for fp in csv_files:
            try:
                df = pd.read_csv(fp, comment='#')
                if {'label', 'channel'}.issubset(df.columns):
                    combinations.append(tuple(sorted(set(df['label'].tolist()))))
                    for _, row in df.iterrows():
                        channel_patterns[row['channel']][row['label']] += 1
            except Exception:
                pass

        return {
            'artifact_combinations': combinations,
            'channel_artifact_patterns': dict(channel_patterns),
        }
