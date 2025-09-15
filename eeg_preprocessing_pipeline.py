#!/usr/bin/env python3
"""
EEG Preprocessing Pipeline for Enhanced Deep Lightweight CNN (EDL-CNN)

A comprehensive pipeline for preprocessing EEG data for deep learning-based artifact detection.
Handles variable sampling rates, channel configurations, and implements standardized steps
for optimal model performance.

Key Features:
- Standardizes sampling rates (250, 256, 400, 512, 1000 Hz)
- Creates bipolar montage and standardizes to 22 canonical channels
- Applies filtering, referencing, and DC offset removal
- Extracts non-overlapping 1s time windows with artifact labeling
- Implements patient-level data splitting to prevent leakage
- Global normalization using RobustScaler for consistent preprocessing

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2025
"""

import os
import glob
import logging
from typing import List, Tuple, Dict, Any, Optional, DefaultDict
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import RobustScaler
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['EEGPreprocessingPipeline']


class EEGPreprocessingPipeline:
    """
    EEG preprocessing pipeline for artifact detection using EDL-CNN.

    This class handles end-to-end preprocessing of EEG data:
    - Bipolar montage creation
    - Channel standardization (22 canonical channels)
    - Resampling, filtering, and referencing
    - Windowing and labeling with artifact annotations
    - Patient-level train/val/test splitting
    - Global normalization for deep learning

    Attributes:
        target_sampling_rate (int): Target sampling rate in Hz.
        target_channels (int): Target number of channels.
        window_size (float): Window duration in seconds.
        overlap (float): Fractional overlap between windows.
        min_artifact_duration (float): Minimum duration for valid artifacts.
        include_clean_windows (bool): Whether to include clean segments.
        clean_label (int): Label assigned to clean windows.
    
    """

    # Configuration
    CONFIG = {
        'canonical_channels': [
            'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
            'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
            'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4',
            'C4-T4', 'T4-A2',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
        ],
        'bipolar_pairs': [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
            ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
            ('C4', 'T4'), ('T4', 'A2'),
            ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
            ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')
        ],
        'artifact_mapping': {
            # Primary artifacts (6 classes)
            'eyem': 0,           # Eye movement
            'musc': 1,           # Muscle artifact
            'elec': 2,           # Electrode artifact
            'chew': 3,           # Chewing
            'shiv': 4,           # Shivering
            'elpp': 5,           # Non-physiological
            # Combined artifacts (mapped to primary)
            'eyem_musc': 0,
            'eyem_elec': 0,
            'musc_elec': 1,
            'eyem_chew': 0,
            'chew_musc': 3,
            'chew_elec': 3,
            'eyem_shiv': 0,
            'shiv_elec': 4,
            # Background and seizures (exclude)
            
            'bckg': -1,
            'cpsz': -1,
            'fnsz': -1,
            'gnsz': -1,
            'tcsz': -1,
        }
    }

    def __init__(
        self,
        target_sampling_rate: int = 250,
        target_channels: int = 22,
        window_size: float = 1.0,
        overlap: float = 0.0,
        min_artifact_duration: float = 0.1,
        include_clean_windows: bool = True,
        clean_label: int = 6,
        verbose: bool = True,
        use_logging: bool = True
    ) -> None:
        self.target_sampling_rate = target_sampling_rate
        self.target_channels = target_channels
        self.window_size = window_size
        self.overlap = overlap
        self.min_artifact_duration = min_artifact_duration
        self.include_clean_windows = include_clean_windows
        self.clean_label = clean_label
        self.verbose = verbose
        self.use_logging = use_logging

        self.window_samples = int(window_size * target_sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

        # Logger setup
        if use_logging:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        else:
            self.logger = None

        # Canonical channels (deduplicated, ordered)
        seen = set()
        self.canonical_channels = [
            ch for ch in self.CONFIG['canonical_channels'] if not (ch in seen or seen.add(ch))
        ]

        # Artifact mapping
        self.artifact_mapping = self.CONFIG['artifact_mapping']

        # Global scaler
        self.scaler = None

        # Statistics
        self.missing_channels_count = 0
        self.zero_window_files = 0

    def create_bipolar_montage(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Create bipolar montage from raw EEG channels.

        Args:
            raw: MNE Raw object with raw channels.

        Returns:
            Raw object with bipolar channels.
        """
        # Clean channel names: 'EEG FP1-REF' -> 'FP1'
        clean_ch_names = {}
        for ch in raw.ch_names:
            if ch.startswith('EEG ') and ch.endswith('-REF'):
                clean_name = ch[4:-4].upper()
                clean_ch_names[ch] = clean_name
            else:
                clean_ch_names[ch] = ch.upper()
        raw.rename_channels(clean_ch_names)

        # Create bipolar channels
        available = set(raw.ch_names)
        bipolar_channels = []
        for anode, cathode in self.CONFIG['bipolar_pairs']:
            if anode in available and cathode in available:
                ch_name = f'{anode}-{cathode}'
                bipolar_channels.append((anode, cathode, ch_name))

        if not bipolar_channels:
            if self.logger:
                self.logger.warning("If No valid bipolar pairs found; using original channels.")
            return raw

        result = mne.set_bipolar_reference(
            raw,
            anode=[p[0] for p in bipolar_channels],
            cathode=[p[1] for p in bipolar_channels],
            ch_name=[p[2] for p in bipolar_channels],
            verbose=False
        )
        return result[0] if isinstance(result, tuple) else result

    def load_and_validate_file(self, edf_path: str, csv_path: str) -> Tuple[Optional[mne.io.Raw], Optional[pd.DataFrame]]:
        """
        Load and validate EDF and CSV files.

        Args:
            edf_path: Path to EDF file.
            csv_path: Path to CSV annotation file.

        Returns:
            Tuple of (raw, annotations) or (None, None) on error.
        """
        if csv_path.endswith('_seiz.csv'):
            if self.logger:
                self.logger.info(f"Skipping seizure file: {csv_path}")
            return None, None

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            annotations = pd.read_csv(csv_path, comment='#')

            required_cols = {'start_time', 'stop_time', 'label'}
            if not required_cols.issubset(annotations.columns):
                raise ValueError(f"Missing columns in {csv_path}")

            annotations['label'] = annotations['label'].str.lower()

            edf_duration = raw.times[-1]
            if annotations['stop_time'].max() > edf_duration + 1.0:
                if self.logger:
                    self.logger.warning(f"Annotations exceed EDF duration in {csv_path}")

            if len(annotations) == 0:
                if self.logger:
                    self.logger.warning(f"No annotations in {csv_path}")
                return None, None

            valid_artifacts = annotations[annotations['label'].isin(self.artifact_mapping)]
            if len(valid_artifacts) == 0:
                if self.logger:
                    self.logger.warning(f"No valid artifacts in {csv_path}")
                return None, None

            return raw, annotations

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading {edf_path}: {e}")
            return None, None

    def resample_eeg(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample EEG to target sampling rate."""
        if raw.info['sfreq'] != self.target_sampling_rate:
            if self.logger:
                self.logger.info(f"Resampling from {raw.info['sfreq']} Hz to {self.target_sampling_rate} Hz")
            raw.resample(self.target_sampling_rate, verbose=False)
        return raw

    def standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Standardize channels to canonical 22-channel bipolar montage.
    
        """
        # Clean and standardize channel names
        raw.rename_channels(lambda x: x.replace('EEG ', '').replace('-REF', '').upper())

        # Try to create bipolar montage if not already present
        has_bipolar = all('-' in ch and not ch.endswith('-REF') for ch in raw.ch_names)
        has_referenced = all(ch.endswith('-REF') for ch in raw.ch_names)

        if not has_bipolar:
            raw = self.create_bipolar_montage(raw)

        # Get current channel names
        current_names = [ch.upper() for ch in raw.ch_names]
        available_channels = []

        for canonical in self.canonical_channels:
            if canonical in current_names:
                idx = current_names.index(canonical)
                available_channels.append(raw.ch_names[idx])

        # Select up to target_channels
        n_pick = min(len(available_channels), self.target_channels)
        if n_pick > 0:
            selected = available_channels[:n_pick]
            raw.pick_channels(selected, ordered=True, verbose=False)
            data, times = raw[:, :]
        else:
            # Fallback: pick first few channels
            raw.pick_channels(raw.ch_names[:min(5, len(raw.ch_names))], verbose=False)
            data, times = raw[:, :]
            selected = raw.ch_names[:len(data)]

        # Pad if needed
        if data.shape[0] < self.target_channels:
            n_pad = self.target_channels - data.shape[0]
            padding = np.zeros((n_pad, data.shape[1]))
            combined_data = np.vstack([data, padding])
            new_ch_names = selected + [f'PAD_{i}' for i in range(n_pad)]
        else:
            combined_data = data
            new_ch_names = selected[:self.target_channels]

        # Ensure consistency
        assert combined_data.shape[0] == len(new_ch_names) == self.target_channels, \
            f"Channel mismatch: {combined_data.shape[0]} vs {len(new_ch_names)} vs {self.target_channels}"

        new_info = mne.create_info(new_ch_names, self.target_sampling_rate, ch_types='eeg')
        return mne.io.RawArray(combined_data, new_info, verbose=False)

    def apply_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply filtering, referencing, and DC offset removal."""
        if self.logger:
            self.logger.info("Applying bandpass (1-40 Hz) and notch (50/60 Hz) filters...")
        raw.filter(1, 40, verbose=False)
        raw.notch_filter([50, 60], verbose=False)

        if self.logger:
            self.logger.info("Setting average reference and removing DC offset...")
        raw.set_eeg_reference('average', verbose=False)
        data, times = raw[:, :]
        data = data - np.mean(data, axis=1, keepdims=True)
        info = mne.create_info(raw.ch_names, self.target_sampling_rate, ch_types='eeg')
        return mne.io.RawArray(data, info, verbose=False)

    def get_window_label(self, annotations: pd.DataFrame, start_time: float, end_time: float) -> Optional[int]:
        """Determine the dominant artifact label in a time window."""
        overlapping = annotations[
            (annotations['start_time'] <= end_time) &
            (annotations['stop_time'] >= start_time)
        ]

        valid_artifacts = []
        for _, row in overlapping.iterrows():
            if row['label'] in self.artifact_mapping:
                duration = row['stop_time'] - row['start_time']
                if duration >= self.min_artifact_duration:
                    valid_artifacts.append(row)

        if not valid_artifacts:
            return None

        artifact_counts = defaultdict(int)
        for row in valid_artifacts:
            label = self.artifact_mapping[row['label']]
            if label >= 0:
                artifact_counts[label] += 1

        return max(artifact_counts.items(), key=lambda x: x[1])[0] if artifact_counts else None

    def create_time_windows(self, raw: mne.io.Raw, annotations: pd.DataFrame) -> List[Tuple[np.ndarray, int]]:
        """Extract non-overlapping time windows with labels."""
        data, times = raw[:, :]
        data = data.T  # (samples, channels)
        windows = []

        for start_idx in range(0, len(data) - self.window_samples + 1, self.step_samples):
            end_idx = start_idx + self.window_samples
            start_time = times[start_idx]
            end_time = times[end_idx - 1]
            window_data = data[start_idx:end_idx, :]

            label = self.get_window_label(annotations, start_time, end_time)
            if label is not None:
                windows.append((window_data, label))
            elif self.include_clean_windows:
                windows.append((window_data, self.clean_label))

        return windows

    def process_single_file(self, edf_path: str, csv_path: str) -> List[Tuple[np.ndarray, int]]:
        """Process a single EDF+CSV pair."""
        if self.logger:
            self.logger.info(f"Processing {Path(edf_path).name}...")

        raw, annotations = self.load_and_validate_file(edf_path, csv_path)
        if raw is None:
            return []

        raw = self.resample_eeg(raw)
        raw = self.standardize_channels(raw)
        raw = self.apply_preprocessing(raw)
        windows = self.create_time_windows(raw, annotations)

        if len(windows) == 0:
            self.zero_window_files += 1
            if self.logger:
                self.logger.warning(f"No windows extracted from {Path(edf_path).name}")

        if self.logger:
            self.logger.info(f"Extracted {len(windows)} windows")
        return windows

    def process_dataset(
        self,
        edf_dir: str = 'edf/01_tcp_ar',
        max_files: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """Process all EDF+CSV file pairs in the directory."""
        if self.logger:
            self.logger.info(f"Processing dataset from {edf_dir}...")

        self.missing_channels_count = 0
        self.zero_window_files = 0

        edf_files = glob.glob(os.path.join(edf_dir, "*.edf"))
        if max_files:
            edf_files = edf_files[:max_files]

        file_pairs = []
        for edf_file in edf_files:
            base = os.path.splitext(edf_file)[0]
            csv_file = base + ".csv"
            if os.path.exists(csv_file) and not csv_file.endswith('_seiz.csv'):
                file_pairs.append((edf_file, csv_file))

        if self.logger:
            self.logger.info(f"Found {len(file_pairs)} valid file pairs")

        if n_jobs > 1:
            if self.logger:
                self.logger.info(f"Processing with {n_jobs} parallel jobs...")
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_single_file)(edf, csv) for edf, csv in file_pairs
            )
        else:
            results = []
            for i, (edf, csv) in enumerate(file_pairs):
                if (i + 1) % 20 == 0 and self.logger:
                    self.logger.info(f"Processed {i + 1}/{len(file_pairs)} files...")
                results.append(self.process_single_file(edf, csv))

        # Aggregate results
        all_windows = []
        all_sources = []
        class_counts = defaultdict(int)
        file_stats = []

        for i, windows in enumerate(results):
            for _, label in windows:
                class_counts[label] += 1

            file_stats.append({
                'file': Path(file_pairs[i][0]).name,
                'windows': len(windows),
                'classes': list(set(w[1] for w in windows))
            })

            all_windows.extend(windows)
            src_id = Path(file_pairs[i][0]).stem
            all_sources.extend([src_id] * len(windows))

        if self.logger:
            self.logger.info(f"Processing Summary:")
            self.logger.info(f"  Files with missing canonical channels: {self.missing_channels_count}")
            self.logger.info(f"  Files with zero windows: {self.zero_window_files}")

        return {
            'windows': all_windows,
            'window_sources': all_sources,
            'class_counts': dict(class_counts),
            'file_stats': file_stats,
            'total_files': len(edf_files),
            'processed_files': len(file_stats),
            'total_windows': len(all_windows),
            'missing_channels_count': self.missing_channels_count,
            'zero_window_files': self.zero_window_files
        }

    def prepare_training_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for Enhanced Deep lightweight CNN training with patient-level splitting."""
        if self.logger:
            self.logger.info("Preparing training data for Enhanced Deep lightweight CNN...")

        data_list = [w[0] for w in results['windows']]
        labels = [w[1] for w in results['windows']]

        X = np.array(data_list, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        if self.logger:
            self.logger.info(f"Data shape: {X.shape}")

        # Global normalization
        if self.logger:
            self.logger.info("Applying RobustScaler normalization...")
        flat = X.reshape(-1, X.shape[2])
        self.scaler = RobustScaler().fit(flat)
        X_norm = self.scaler.transform(flat).reshape(X.shape)

        # Patient-level split
        groups = np.array(results['window_sources'])
        indices = np.arange(len(y))

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss1.split(indices, y, groups=groups))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        tv_idx = indices[train_val_idx]
        tv_groups = groups[train_val_idx]
        tv_y = y[train_val_idx]
        train_rel, val_rel = next(gss2.split(tv_idx, tv_y, groups=tv_groups))

        train_idx = tv_idx[train_rel]
        val_idx = tv_idx[val_rel]

        return {
            'X_train_3d': X_norm[train_idx],
            'X_val_3d': X_norm[val_idx],
            'X_test_3d': X_norm[test_idx],
            'y_train': y[train_idx],
            'y_val': y[val_idx],
            'y_test': y[test_idx],
            'idx_train': train_idx,
            'idx_val': val_idx,
            'idx_test': test_idx,
            'X_all_3d': X_norm,
            'y_all': y,
            'original_shape': X.shape,
            'n_classes': len(np.unique(y))
        }

    def save_data(self, data_dict: Dict[str, Any], results: Dict[str, Any], output_path: str = '.') -> None:
        """Save preprocessed data and metadata."""
        if self.logger:
            self.logger.info(f"Saving data to {output_path}...")

        os.makedirs(output_path, exist_ok=True)

        # Save 3D arrays
        for key in ['X_train_3d', 'X_val_3d', 'X_test_3d', 'X_all_3d']:
            if key in data_dict:
                np.save(os.path.join(output_path, f"{key}.npy"), data_dict[key])
                if self.logger:
                    self.logger.info(f"Saved {key}.npy")

        # Save labels
        for key in ['y_train', 'y_val', 'y_test', 'y_all']:
            if key in data_dict:
                df = pd.DataFrame({key: data_dict[key]})
                df.to_csv(os.path.join(output_path, f"{key}.csv"), index=False)
                if self.logger:
                    self.logger.info(f"Saved {key}.csv")

        # Save metadata
        pd.DataFrame(results['file_stats']).to_csv(os.path.join(output_path, 'file_statistics.csv'), index=False)
        pd.DataFrame(list(results['class_counts'].items()), columns=['class', 'count']) \
            .to_csv(os.path.join(output_path, 'class_distribution.csv'), index=False)

        summary = {k: results[k] for k in ['total_files', 'processed_files', 'total_windows',
                                           'missing_channels_count', 'zero_window_files']}
        pd.DataFrame([summary]).to_csv(os.path.join(output_path, 'processing_summary.csv'), index=False)

        # Save scaler and indices
        from joblib import dump
        dump(self.scaler, os.path.join(output_path, 'robust_scaler.joblib'))
        np.savez_compressed(os.path.join(output_path, 'split_indices.npz'),
                            idx_train=data_dict['idx_train'],
                            idx_val=data_dict['idx_val'],
                            idx_test=data_dict['idx_test'])

        if self.logger:
            self.logger.info("Preprocessing completed and data saved.")


def main() -> None:
    """Run the full preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG preprocessing pipeline')
    parser.add_argument('--window-seconds', type=float, default=1.0,
                       help='Window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.0,
                       help='Window overlap fraction')
    
    args = parser.parse_args()
    
    logger.info("Starting EEG preprocessing pipeline")

    pipeline = EEGPreprocessingPipeline(
        target_sampling_rate=250,
        target_channels=22,
        window_size=args.window_seconds,
        overlap=args.overlap,
        min_artifact_duration=0.1,
        include_clean_windows=True,
        use_logging=True,
        verbose=True
    )

    import multiprocessing as mp
    n_jobs = max(1, mp.cpu_count() - 1)

    results = pipeline.process_dataset(
        edf_dir='edf/01_tcp_ar',
        max_files=150,
        n_jobs=n_jobs
    )

    data_dict = pipeline.prepare_training_data(results)
    pipeline.save_data(data_dict, results, output_path='binary_models_data')

    logger.info("Preprocessing completed successfully.")
    logger.info(f"Total windows: {results['total_windows']}, Classes: {data_dict['n_classes']}")


if __name__ == "__main__":
    main()