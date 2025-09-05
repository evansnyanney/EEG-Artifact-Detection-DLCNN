#!/usr/bin/env python3
"""
Binary Model Data Preparation for EEG Artifact Detection

This module prepares binary classification datasets for three specialized artifact
detection models using preprocessed EEG data. Each model is trained to distinguish
a specific artifact type from clean EEG signals, enabling targeted detection.

Key Features:
- Three binary models: Eye Movement, Muscle Artifacts, Non-Physiological
- Patient-level data splitting to prevent leakage
- Focal loss parameters and class weights for imbalance handling
- 3D data format (n_samples, n_timepoints, n_channels) for 1D CNN models
- Metadata generation for reproducible model training

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2025
"""

import os
import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['BinaryModelDataPreparer']


class BinaryModelDataPreparer:
    """
    Prepares binary classification datasets for EEG artifact detection models.

    This class handles the creation of three binary models:
    - Eye Movement Detector (Class 0 vs Clean)
    - Muscle Artifacts Detector (Classes 1, 3, 4 vs Clean)
    - Non-Physiological Detector (Classes 2, 5 vs Clean)

    It ensures patient-level data splitting and generates metadata for training.

    Attributes:
        input_dir (str): Directory containing preprocessed data.
        output_dir (str): Directory to save binary model data.
        X_all (np.ndarray): Full 3D EEG data (n_samples, timepoints, channels).
        y_all (np.ndarray): Full label array.
        split_indices (Dict[str, np.ndarray]): Train/val/test indices.
    """

    def __init__(self, input_dir: str = '.', output_dir: str = 'binary_models_data') -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.X_all: Optional[np.ndarray] = None
        self.y_all: Optional[np.ndarray] = None
        self.split_indices: Optional[Dict[str, np.ndarray]] = None

    def load_data(self) -> None:
        """Load preprocessed data and split indices."""
        logger.info("Loading preprocessed data for binary model preparation...")

        x_path = os.path.join(self.input_dir, 'X_all_3d.npy')
        y_path = os.path.join(self.input_dir, 'y_all.csv')
        split_path = os.path.join(self.input_dir, 'split_indices.npz')

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"X_all_3d.npy not found at {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"y_all.csv not found at {y_path}")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split_indices.npz not found at {split_path}")

        self.X_all = np.load(x_path)
        self.y_all = pd.read_csv(y_path)['y_all'].values
        self.split_indices = dict(np.load(split_path))

        logger.info(f"Loaded data: X={self.X_all.shape}, y={self.y_all.shape}")
        logger.info("Split indices loaded: train, val, test")

    def prepare_all_models(self) -> None:
        """Prepare all three binary models."""
        models_config = {
            'eye_movement': {
                'name': 'Eye Movement Detector',
                'positive_classes': [0],
                'negative_class': 6,
                'description': 'Detects eye blinks and eye movement artifacts'
            },
            'muscle_artifacts': {
                'name': 'Muscle Artifact Detector',
                'positive_classes': [1, 3, 4],
                'negative_class': 6,
                'description': 'Detects muscle artifacts, chewing, and shivering'
            },
            'non_physiological': {
                'name': 'Non-Physiological Detector (Revised)',
                'positive_classes': [2, 5],
                'negative_class': 6,
                'description': 'Detects electrode artifacts and non-physiological artifacts'
            }
        }

        os.makedirs(self.output_dir, exist_ok=True)

        for model_key, config in models_config.items():
            self._prepare_single_model(model_key, config)

    def _prepare_single_model(self, model_key: str, config: Dict[str, Any]) -> None:
        """Prepare data for a single binary model."""
        logger.info(f"Preparing data for: {config['name']}")

        # Create binary mask
        positive_mask = np.isin(self.y_all, config['positive_classes'])
        negative_mask = (self.y_all == config['negative_class'])
        binary_mask = positive_mask | negative_mask

        X_binary = self.X_all[binary_mask]
        y_binary = np.where(positive_mask[binary_mask], 1, 0)

        # Map global indices to this subset
        original_indices = np.where(binary_mask)[0]
        train_mask = np.isin(original_indices, self.split_indices['idx_train'])
        val_mask = np.isin(original_indices, self.split_indices['idx_val'])
        test_mask = np.isin(original_indices, self.split_indices['idx_test'])

        X_train = X_binary[train_mask]
        X_val = X_binary[val_mask]
        X_test = X_binary[test_mask]
        y_train = y_binary[train_mask]
        y_val = y_binary[val_mask]
        y_test = y_binary[test_mask]

        # Compute focal loss parameters and class weights
        focal_loss_params = {'gamma': 2.0, 'alpha': 0.25}
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights)}

        # Log statistics
        pos_ratio = np.sum(y_binary == 1) / len(y_binary)
        logger.info(f"  Positive ratio: {pos_ratio:.3f} ({np.sum(y_binary == 1)} / {len(y_binary)})")

        # Save data
        model_dir = os.path.join(self.output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)

        np.save(os.path.join(model_dir, 'X_train_3d.npy'), X_train)
        np.save(os.path.join(model_dir, 'X_val_3d.npy'), X_val)
        np.save(os.path.join(model_dir, 'X_test_3d.npy'), X_test)
        np.save(os.path.join(model_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(model_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(model_dir, 'y_test.npy'), y_test)

        # Save metadata
        metadata = {
            'model_name': config['name'],
            'description': config['description'],
            'positive_classes': config['positive_classes'],
            'negative_class': config['negative_class'],
            'total_samples': int(len(X_binary)),
            'positive_samples': int(np.sum(y_binary == 1)),
            'negative_samples': int(np.sum(y_binary == 0)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'split_source': 'patient-level (from split_indices.npz)',
            'focal_loss_params': focal_loss_params,
            'class_weights': class_weight_dict,
            'input_shape_3d': X_train.shape[1:],
            'data_format': '3D (n_samples, n_timepoints, n_channels) for 1D CNN',
            'n_timepoints': int(X_train.shape[1]),
            'n_channels': int(X_train.shape[2])
        }

        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Binary model data saved to: {model_dir}")

    def analyze_data(self) -> None:
        """Analyze and print summary statistics for all binary models."""
        logger.info("Analyzing prepared binary model data...")

        models = ['eye_movement', 'muscle_artifacts', 'non_physiological']

        for model in models:
            model_dir = os.path.join(self.output_dir, model)
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory not found: {model_dir}")
                continue

            metadata_path = os.path.join(model_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata not found for model: {model}")
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            print(f"\n{metadata['model_name']}")
            print(f"  Description: {metadata['description']}")
            print(f"  Total samples: {metadata['total_samples']}")
            pos_pct = metadata['positive_samples'] / metadata['total_samples']
            print(f"  Positive: {metadata['positive_samples']} ({pos_pct:.1%})")
            neg_pct = metadata['negative_samples'] / metadata['total_samples']
            print(f"  Negative: {metadata['negative_samples']} ({neg_pct:.1%})")
            print(f"  Split: {metadata['train_samples']}/{metadata['val_samples']}/{metadata['test_samples']}")
            print(f"  Input shape: {metadata['input_shape_3d']}")
            print(f"  Focal loss: γ={metadata['focal_loss_params']['gamma']}, α={metadata['focal_loss_params']['alpha']}")


def main() -> None:
    """Main execution function."""
    logger.info("Starting binary model data preparation")

    preparer = BinaryModelDataPreparer(input_dir='binary_models_data', output_dir='binary_models_data')
    preparer.load_data()
    preparer.prepare_all_models()
    preparer.analyze_data()

    print("\n" + "="*70)
    print("BINARY MODEL DATA PREPARATION COMPLETED")
    print("="*70)
    print("Three binary models for training:")
    print("  1. Eye Movement Detector")
    print("  2. Muscle Artifact Detector")
    print("  3. Non-Physiological Detector (includes electrode artifacts)")
    print("\nData format: 3D (n_samples, n_timepoints, n_channels) for Enhanced Deep lightweight CNN")
    print("All metadata, splits, and class weights saved.")
    print("="*70)


if __name__ == "__main__":
    main()