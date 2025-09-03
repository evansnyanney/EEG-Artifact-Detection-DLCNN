#!/usr/bin/env python3
"""
EEG Data Exploration and Analysis (UPMC)

Comprehensive analysis of EEG recordings (EDF) and artifact annotations (CSV) to support:
- Preprocessing pipeline design
- Artifact detection model development
- Data quality assessment
- Dataset characterization

Key Features:
- EDF file analysis: durations, sampling rates, channels, signal stats
- CSV annotation analysis: artifact types, durations, channel patterns
- File matching: EDF-CVS pair integrity
- Artifact pattern analysis: co-occurrence, timing, combinations

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2024
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

import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['EEGExplorer']


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EDF_DIR = 'edf/01_tcp_ar'  # Directory containing EDF and CSV files
PROGRESS_INTERVAL = 50  # Log progress every N files
MAX_UNMATCHED_DISPLAY = 10  # Max unmatched files to display


class EEGExplorer:
    """
    Comprehensive EEG data explorer for EDF recordings and CSV annotations.

    This class analyzes EEG datasets to extract key statistics for preprocessing,
    model design, and data quality assessment. It supports:
    - EDF file inspection (durations, sampling rates, channels)
    - CSV annotation analysis (artifact types, durations, channels)
    - File matching between EDF and CSV
    - Artifact pattern detection (combinations, timing, co-occurrence)

    Attributes:
        edf_dir (str): Path to directory containing EDF and CSV files.
    """

    def __init__(self, edf_dir: str = DEFAULT_EDF_DIR) -> None:
        self.edf_dir = edf_dir
        logger.info(f"Initialized EEGExplorer with directory: {self.edf_dir}")

    def analyze_edf_files(self) -> Dict[str, Any]:
        """
        Analyze all EDF files to extract recording characteristics.

        Returns:
            Dictionary containing durations, sampling rates, channel counts,
            and signal statistics.
        """
        logger.info("Analyzing EDF files (EEG recordings)")
        edf_files = glob.glob(os.path.join(self.edf_dir, "*.edf"))
        logger.info(f"Found {len(edf_files)} EDF files")

        if not edf_files:
            logger.error("No EDF files found.")
            return {}

        durations = []
        sampling_rates = []
        channel_counts = []
        data_ranges = []
        means = []
        stds = []
        successful = 0

        for i, file_path in enumerate(edf_files):
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(edf_files)} EDF files")

            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                durations.append(raw.times[-1])
                sampling_rates.append(raw.info['sfreq'])
                channel_counts.append(len(raw.ch_names))

                data, _ = raw[:, :]
                data_ranges.append(data.max() - data.min())
                means.append(data.mean())
                stds.append(data.std())

                successful += 1
            except Exception as e:
                logger.warning(f"Failed to load {os.path.basename(file_path)}: {e}")

        # Convert to arrays
        durations = np.array(durations)
        sampling_rates = np.array(sampling_rates)
        channel_counts = np.array(channel_counts)
        data_ranges = np.array(data_ranges)
        means = np.array(means)
        stds = np.array(stds)

        logger.info(f"Successfully processed {successful}/{len(edf_files)} EDF files")
        self._print_edf_summary(
            durations, sampling_rates, channel_counts, data_ranges, means, stds
        )

        return {
            'durations': durations,
            'sampling_rates': sampling_rates,
            'channel_counts': channel_counts,
            'data_ranges': data_ranges,
            'means': means,
            'stds': stds,
            'successful_files': successful,
            'total_files': len(edf_files)
        }

    def _print_edf_summary(
        self,
        durations: np.ndarray,
        sampling_rates: np.ndarray,
        channel_counts: np.ndarray,
        data_ranges: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray
    ) -> None:
        """Print summary statistics for EDF file analysis."""
        print("\n" + "=" * 60)
        print("EDF FILES STATISTICS SUMMARY")
        print("=" * 60)

        # Duration
        print(f"\nRecording Duration (seconds):")
        print(f"   Mean:   {durations.mean():.2f}")
        print(f"   Median: {np.median(durations):.2f}")
        print(f"   Std:    {durations.std():.2f}")
        print(f"   Min:    {durations.min():.2f}")
        print(f"   Max:    {durations.max():.2f}")

        # Sampling rates
        unique_sr = np.unique(sampling_rates)
        print(f"\nSampling Rates (Hz): {unique_sr.tolist()}")
        for sr in unique_sr:
            count = np.sum(sampling_rates == sr)
            pct = 100 * count / len(sampling_rates)
            print(f"   {sr:3.0f} Hz: {count:4d} files ({pct:5.1f}%)")

        # Channel counts
        unique_ch = np.unique(channel_counts)
        print(f"\nChannel Counts: {sorted(unique_ch.tolist())}")
        for ch in unique_ch:
            count = np.sum(channel_counts == ch)
            pct = 100 * count / len(channel_counts)
            print(f"   {ch:2d} ch: {count:4d} files ({pct:5.1f}%)")

        # Signal stats
        for name, arr in [("Amplitude Range", data_ranges), ("Mean", means), ("Std", stds)]:
            print(f"\nSignal {name}:")
            print(f"   Mean:   {arr.mean():.2f}")
            print(f"   Median: {np.median(arr):.2f}")
            print(f"   Std:    {arr.std():.2f}")
            print(f"   Min:    {arr.min():.2f}")
            print(f"   Max:    {arr.max():.2f}")

    def analyze_csv_annotations(self) -> Dict[str, Any]:
        """
        Analyze all CSV annotation files for artifact distribution.

        Returns:
            Dictionary containing annotation counts, durations, labels, and channels.
        """
        logger.info("Analyzing CSV annotation files")
        csv_files = glob.glob(os.path.join(self.edf_dir, "*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")

        if not csv_files:
            logger.error("No CSV files found.")
            return {}

        all_labels = []
        all_channels = []
        all_durations = []
        annotation_counts = []
        all_annotations = []
        successful = 0

        required_cols = {'label', 'channel', 'start_time', 'stop_time'}

        for i, file_path in enumerate(csv_files):
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(csv_files)} CSV files")

            try:
                df = pd.read_csv(file_path, comment='#')
                if not required_cols.issubset(df.columns):
                    logger.warning(f"Missing required columns in {os.path.basename(file_path)}")
                    continue

                count = len(df)
                durations = (df['stop_time'] - df['start_time']).tolist()

                all_labels.extend(df['label'].tolist())
                all_channels.extend(df['channel'].tolist())
                all_durations.extend(durations)
                annotation_counts.append(count)
                all_annotations.append(df)

                successful += 1
            except Exception as e:
                logger.warning(f"Failed to load {os.path.basename(file_path)}: {e}")

        all_durations = np.array(all_durations)
        annotation_counts = np.array(annotation_counts)

        logger.info(f"Successfully processed {successful}/{len(csv_files)} CSV files")
        self._print_csv_summary(annotation_counts, all_durations, all_labels, all_channels)

        return {
            'annotation_counts': annotation_counts,
            'durations': all_durations,
            'labels': all_labels,
            'channels': all_channels,
            'all_annotations': all_annotations,
            'successful_files': successful,
            'total_files': len(csv_files)
        }

    def _print_csv_summary(
        self,
        annotation_counts: np.ndarray,
        durations: np.ndarray,
        labels: List[str],
        channels: List[str]
    ) -> None:
        """Print summary statistics for CSV annotation analysis."""
        print("\n" + "=" * 60)
        print("CSV ANNOTATIONS STATISTICS SUMMARY")
        print("=" * 60)

        # Annotation counts
        print(f"\nAnnotations per File:")
        print(f"   Mean:   {annotation_counts.mean():.2f}")
        print(f"   Median: {np.median(annotation_counts):.2f}")
        print(f"   Std:    {annotation_counts.std():.2f}")
        print(f"   Min:    {annotation_counts.min()}")
        print(f"   Max:    {annotation_counts.max()}")

        # Duration
        print(f"\nAnnotation Durations (seconds):")
        print(f"   Mean:   {durations.mean():.2f}")
        print(f"   Median: {np.median(durations):.2f}")
        print(f"   Std:    {durations.std():.2f}")
        print(f"   Min:    {durations.min():.2f}")
        print(f"   Max:    {durations.max():.2f}")

        # Label distribution
        label_counts = Counter(labels)
        print(f"\nArtifact Type Distribution (n={len(labels):,}):")
        for label, count in label_counts.most_common():
            pct = 100 * count / len(labels)
            print(f"   {label:20s}: {count:6d} ({pct:5.1f}%)")

        # Channel distribution
        channel_counts = Counter(channels)
        print(f"\nChannel Distribution (n={len(channels):,}):")
        for channel, count in channel_counts.most_common(10):
            pct = 100 * count / len(channels)
            print(f"   {channel:15s}: {count:6d} ({pct:5.1f}%)")

    def analyze_file_matching(self) -> Dict[str, Any]:
        """
        Analyze correspondence between EDF and CSV files.

        Returns:
            Dictionary with matched pairs, unmatched files, and matching rates.
        """
        logger.info("Analyzing file matching between EDF and CSV")
        edf_files = glob.glob(os.path.join(self.edf_dir, "*.edf"))
        csv_files = glob.glob(os.path.join(self.edf_dir, "*.csv"))

        matched = 0
        unmatched_edf = []
        unmatched_csv = []

        # Match EDF → CSV
        for edf in edf_files:
            base = os.path.splitext(edf)[0]
            if os.path.exists(base + ".csv"):
                matched += 1
            else:
                unmatched_edf.append(os.path.basename(edf))

        # Match CSV → EDF
        for csv in csv_files:
            base = os.path.splitext(csv)[0]
            if not os.path.exists(base + ".edf"):
                unmatched_csv.append(os.path.basename(csv))

        # Summary
        self._print_matching_summary(matched, unmatched_edf, unmatched_csv, len(edf_files), len(csv_files))

        return {
            'matched_pairs': matched,
            'unmatched_edf': unmatched_edf,
            'unmatched_csv': unmatched_csv,
            'total_edf': len(edf_files),
            'total_csv': len(csv_files),
            'edf_matching_rate': 100 * matched / len(edf_files) if edf_files else 0,
            'csv_matching_rate': 100 * matched / len(csv_files) if csv_files else 0
        }

    def _print_matching_summary(
        self,
        matched: int,
        unmatched_edf: List[str],
        unmatched_csv: List[str],
        total_edf: int,
        total_csv: int
    ) -> None:
        """Print file matching results."""
        print("\n" + "=" * 60)
        print("FILE MATCHING ANALYSIS")
        print("=" * 60)

        print(f"\nMatched Pairs:     {matched:4d}")
        print(f"Unmatched EDF:     {len(unmatched_edf):4d}")
        print(f"Unmatched CSV:     {len(unmatched_csv):4d}")

        edf_rate = 100 * matched / total_edf if total_edf else 0
        csv_rate = 100 * matched / total_csv if total_csv else 0
        print(f"\nMatching Rates:")
        print(f"   EDF: {edf_rate:5.1f}%")
        print(f"   CSV: {csv_rate:5.1f}%")

        if unmatched_edf:
            print(f"\nUnmatched EDF Files (first {MAX_UNMATCHED_DISPLAY}):")
            for f in unmatched_edf[:MAX_UNMATCHED_DISPLAY]:
                print(f"   {f}")
            if len(unmatched_edf) > MAX_UNMATCHED_DISPLAY:
                print(f"   ... and {len(unmatched_edf) - MAX_UNMATCHED_DISPLAY} more")

        if unmatched_csv:
            print(f"\nUnmatched CSV Files (first {MAX_UNMATCHED_DISPLAY}):")
            for f in unmatched_csv[:MAX_UNMATCHED_DISPLAY]:
                print(f"   {f}")
            if len(unmatched_csv) > MAX_UNMATCHED_DISPLAY:
                print(f"   ... and {len(unmatched_csv) - MAX_UNMATCHED_DISPLAY} more")

    def analyze_artifact_patterns(self) -> Dict[str, Any]:
        """
        Analyze artifact co-occurrence, timing, and channel patterns.

        Returns:
            Dictionary with artifact combinations, channel patterns, and time distributions.
        """
        logger.info("Analyzing artifact patterns")
        csv_files = glob.glob(os.path.join(self.edf_dir, "*.csv"))

        if not csv_files:
            logger.error("No CSV files found for pattern analysis.")
            return {}

        combinations = []
        channel_patterns: DefaultDict[str, Counter] = defaultdict(Counter)
        time_distributions: DefaultDict[str, List[float]] = defaultdict(list)
        successful = 0

        required_cols = {'label', 'channel', 'start_time'}

        for i, file_path in enumerate(csv_files):
            if (i + 1) % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(csv_files)} files for pattern analysis")

            try:
                df = pd.read_csv(file_path, comment='#')
                if not required_cols.issubset(df.columns):
                    continue

                # Combinations per file
                labels = set(df['label'].tolist())
                combinations.append(tuple(sorted(labels)))

                # Channel-artifact mapping
                for _, row in df.iterrows():
                    channel_patterns[row['channel']][row['label']] += 1
                    time_distributions[row['label']].append(row['start_time'])

                successful += 1
            except Exception as e:
                logger.warning(f"Error processing {os.path.basename(file_path)}: {e}")

        logger.info(f"Pattern analysis completed on {successful}/{len(csv_files)} files")
        self._print_pattern_summary(combinations, channel_patterns, time_distributions)

        return {
            'artifact_combinations': combinations,
            'channel_artifact_patterns': dict(channel_patterns),
            'time_distributions': dict(time_distributions),
            'successful_files': successful,
            'total_files': len(csv_files)
        }

    def _print_pattern_summary(
        self,
        combinations: List[tuple],
        channel_patterns: Dict[str, Counter],
        time_distributions: Dict[str, List[float]]
    ) -> None:
        """Print artifact pattern analysis results."""
        print("\n" + "=" * 60)
        print("ARTIFACT PATTERN ANALYSIS RESULTS")
        print("=" * 60)

        # Combinations
        combo_counts = Counter(combinations)
        print(f"\nMost Common Artifact Combinations:")
        for combo, count in combo_counts.most_common(10):
            combo_str = ", ".join(combo) if combo else "None"
            pct = 100 * count / len(combinations)
            print(f"   {combo_str:40s}: {count:4d} files ({pct:5.1f}%)")

        # Channel patterns
        total_per_channel = {
            ch: sum(counts.values()) for ch, counts in channel_patterns.items()
        }
        print(f"\nTop 10 Most Artifact-Prone Channels:")
        for ch, total in sorted(total_per_channel.items(), key=lambda x: -x[1])[:10]:
            print(f"   {ch:15s}: {total:6d} artifacts")

        # Timing
        print(f"\nArtifact Start Time Statistics:")
        for label, times in time_distributions.items():
            if len(times) < 10:
                continue
            t = np.array(times)
            print(f"   {label:20s}: Mean={t.mean():.2f}s, Std={t.std():.2f}s, "
                  f"Range=[{t.min():.2f}, {t.max():.2f}]s")


def main() -> Dict[str, Any]:
    """
    Run comprehensive EEG data exploration.

    This function executes the following analyses:
    1. EDF file analysis
    2. CSV annotation analysis
    3. File matching
    4. Artifact pattern detection

    Returns:
        Dictionary of all analysis results.
    """
    logger.info("Starting comprehensive EEG data exploration")

    explorer = EEGExplorer()

    results = {}

    try:
        results['edf_analysis'] = explorer.analyze_edf_files()
        results['csv_analysis'] = explorer.analyze_csv_annotations()
        results['matching_analysis'] = explorer.analyze_file_matching()
        results['pattern_analysis'] = explorer.analyze_artifact_patterns()

        logger.info("Data exploration completed successfully")
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EEG DATA ANALYSIS COMPLETED!")
        print("=" * 80)
        print("Use these insights to:")
        print("   - Design preprocessing pipelines")
        print("   - Choose model architectures")
        print("   - Evaluate data quality")
        print("   - Plan artifact detection strategies")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"ERROR: {e}")

    return results


if __name__ == "__main__":
    results = main()