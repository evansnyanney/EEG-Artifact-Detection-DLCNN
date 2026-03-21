#!/usr/bin/env python3
"""
Non-Physiological Artifact Detector using Deep Lightweight CNN (DLCNN)

Binary classifier for detecting non-physiological artifacts (electrode artifacts,
cable noise) in EEG data.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import json
import random
import logging
import time
from typing import Tuple, Dict, Optional, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..losses import focal_loss_with_class_weights

import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['NonPhysiologicalDetector']

_LABEL_ARTIFACT = 'Non-Physiological'


class NonPhysiologicalDetector:
    """
    DLCNN for binary classification of non-physiological artifacts in EEG.

    Classifies EEG segments as clean or contaminated with non-physiological
    artifacts (electrode issues, cable noise). Uses lightweight or standard
    architecture options to promote generalization.

    Attributes:
        model_name: Identifier for saving/loading the model.
        model: The compiled Keras model.
        history: Training history.
        metadata: Dataset metadata.
        X_train, X_val, X_test: Input data splits.
        y_train, y_val, y_test: Label splits.
    """

    def __init__(
        self, model_name: str = "edl_cnn_non_physiological", verbose: bool = True
    ) -> None:
        self.model_name = model_name
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.verbose = verbose

        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        self.logger = logging.getLogger(self.__class__.__name__)
        if not verbose:
            self.logger.setLevel(logging.WARNING)

        self.results_dir = os.path.join('results', self.model_name)
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(
        self, data_dir: str = "binary_models_data/non_physiological"
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Load and split EEG data into train/validation/test sets.

        Args:
            data_dir: Path to preprocessed data directory.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        self.logger.info(f"Loading non-physiological data from: {data_dir}")

        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        X_train_full = np.load(os.path.join(data_dir, "X_train_3d.npy"))
        X_test_full = np.load(os.path.join(data_dir, "X_test_3d.npy"))
        y_train_full = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int32)
        y_test_full = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int32)

        X_combined = np.vstack([X_train_full, X_test_full])
        y_combined = np.hstack([y_train_full, y_test_full])

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        self.logger.info(f"Training set: {self.X_train.shape}")
        self.logger.info(f"Validation set: {self.X_val.shape}")
        self.logger.info(f"Test set: {self.X_test.shape}")

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def build_model(
        self, input_shape: Optional[Tuple[int, ...]] = None, model_type: str = 'lightweight'
    ) -> tf.keras.Model:
        """
        Build a CNN for non-physiological artifact detection.

        Args:
            input_shape: Shape of input (timesteps, channels). Inferred from data if None.
            model_type: ``'lightweight'`` (minimal) or ``'standard'`` (regularized).

        Returns:
            Compiled Keras model.
        """
        if input_shape is None:
            if self.X_train is None:
                raise ValueError("Input shape not provided and training data not loaded.")
            input_shape = self.X_train.shape[1:]

        self.logger.info(f"Building {model_type} model with input shape: {input_shape}")

        model = models.Sequential([layers.Input(shape=input_shape)])

        if model_type == 'lightweight':
            model.add(layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
        elif model_type == 'standard':
            for filters in [32, 64, 128]:
                model.add(layers.Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D(pool_size=3))
                model.add(layers.Dropout(0.3))
            model.add(layers.GlobalAveragePooling1D())
            for units in [128, 64]:
                model.add(layers.Dense(units, activation='relu'))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(0.3))
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        focal_params = self.metadata.get('focal_loss_params', {'gamma': 2.0, 'alpha': 0.25})
        class_weights = self.metadata.get('class_weights', {0: 1.0, 1: 1.0})

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=focal_loss_with_class_weights(
                alpha=focal_params['alpha'],
                gamma=focal_params['gamma'],
                class_weights=class_weights
            ),
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model = model
        self.logger.info(f"Model built. Total parameters: {model.count_params():,}")
        return model

    def train(self, epochs: int = 200, batch_size: int = 128) -> tf.keras.callbacks.History:
        """
        Train the model with learning rate reduction and checkpointing.

        Uses ReduceLROnPlateau and ModelCheckpoint only (no EarlyStopping).

        Args:
            epochs: Maximum number of epochs.
            batch_size: Batch size for training.

        Returns:
            Training history object.
        """
        self.logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")

        run_id = int(time.time())
        ckpt_dir = os.path.join('checkpoints', 'non_physiological')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{self.model_name}_best_{run_id}.weights.h5")

        callbacks_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.7, patience=3,
                min_lr=1e-7, verbose=1 if self.verbose else 0, cooldown=2
            ),
            callbacks.ModelCheckpoint(
                filepath=ckpt_path, monitor='val_loss',
                save_best_only=True, save_weights_only=True,
                verbose=1 if self.verbose else 0
            )
        ]

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks_list, verbose=1, shuffle=True
        )

        self.logger.info("Training completed.")
        return self.history

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate on the test set using validation-optimized thresholds.

        Supports Youden's J, fixed specificity, or max TPR at FPR constraint.

        Returns:
            Dictionary of evaluation metrics and predictions.
        """
        self.logger.info("Evaluating model on test set...")

        y_proba_val = self.model.predict(self.X_val).flatten()
        y_proba_test = self.model.predict(self.X_test).flatten()
        y_true_test = self.y_test

        precision_val, recall_val, thr_pr = precision_recall_curve(self.y_val, y_proba_val)
        f1_scores = 2 * (precision_val[:-1] * recall_val[:-1]) / (
            precision_val[:-1] + recall_val[:-1] + 1e-8
        )
        best_idx = np.nanargmax(f1_scores)
        best_threshold = thr_pr[best_idx]

        fpr, tpr, thr_roc = roc_curve(self.y_val, y_proba_val)
        roc_auc_val = auc(fpr, tpr)

        mode = os.getenv('THRESHOLD_MODE', 'youden').lower()
        fixed_spec = float(os.getenv('FIXED_SPEC', '0.95'))
        max_fpr_val = float(os.getenv('MAX_FPR', '0.10'))

        chosen_thr = thr_roc[np.argmax(tpr - fpr)]
        chosen_mode = 'youden'

        if mode == 'fixed_spec':
            spec = 1 - fpr
            mask = spec >= fixed_spec
            if np.any(mask):
                idx = np.argmax(tpr[mask])
                chosen_thr = thr_roc[mask][idx]
                chosen_mode = f"fixed_spec@{fixed_spec:.2f}"
            else:
                chosen_mode = "youden(fallback)"
        elif mode in ('max_tpr_at_fpr', 'max_tpr_at_fpr_le'):
            mask = fpr <= max_fpr_val
            if np.any(mask):
                idx = np.argmax(tpr[mask])
                chosen_thr = thr_roc[mask][idx]
                chosen_mode = f"max_tpr_at_fpr<={max_fpr_val:.2f}"
            else:
                chosen_mode = "youden(fallback)"

        y_pred_test = (y_proba_test >= chosen_thr).astype(int)

        acc = accuracy_score(y_true_test, y_pred_test)
        prec = precision_score(y_true_test, y_pred_test)
        rec = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        pr_auc = auc(recall_val, precision_val)

        try:
            pos = self.metadata.get('positive_samples', 0)
            total = self.metadata.get('total_samples', 1)
            target_prevalence = float(pos) / max(1.0, float(total))
        except Exception:
            target_prevalence = 0.05

        denom = (target_prevalence * tpr) + ((1 - target_prevalence) * fpr)
        precision_adj = np.divide(
            target_prevalence * tpr, denom,
            out=np.zeros_like(denom), where=(denom > 0)
        )
        order = np.argsort(tpr)
        pr_auc_adjusted = auc(tpr[order], precision_adj[order])

        fpr_limit = 0.1
        fpr_ext = np.insert(fpr, 0, 0.0)
        tpr_ext = np.insert(tpr, 0, 0.0)
        idx = np.searchsorted(fpr_ext, fpr_limit, side='right')
        if idx == 0:
            roc_auc_partial = 0.0
        else:
            fpr_clip = fpr_ext[:idx]
            tpr_clip = tpr_ext[:idx]
            if fpr_ext[idx - 1] < fpr_limit < fpr_ext[idx]:
                tpr_interp = tpr_ext[idx - 1] + (tpr_ext[idx] - tpr_ext[idx - 1]) * (
                    (fpr_limit - fpr_ext[idx - 1]) / (fpr_ext[idx] - fpr_ext[idx - 1])
                )
                fpr_clip = np.append(fpr_clip, fpr_limit)
                tpr_clip = np.append(tpr_clip, tpr_interp)
            roc_auc_partial = auc(fpr_clip, tpr_clip)
        roc_auc_partial_norm = roc_auc_partial / fpr_limit if fpr_limit > 0 else float('nan')

        cm = confusion_matrix(y_true_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        class_names = ['Clean', _LABEL_ARTIFACT]

        print("\n" + "=" * 60)
        print("FINAL TEST SET EVALUATION")
        print("=" * 60)
        print(f"Threshold (F1-optimal): {best_threshold:.3f}")
        print(f"Threshold ({chosen_mode}): {chosen_thr:.3f}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Precision    : {prec:.4f}")
        print(f"Recall       : {rec:.4f}")
        print(f"F1-Score     : {f1:.4f}")
        print(f"AUC (PR)     : {pr_auc:.4f}")
        print(f"AUC (ROC)    : {roc_auc_val:.4f}")
        print(f"PR-AUC (adj @pi={target_prevalence:.2f}): {pr_auc_adjusted:.4f}")
        print(f"Partial ROC AUC (FPR<=0.1): {roc_auc_partial:.4f}")
        print(f"Partial ROC AUC (norm): {roc_auc_partial_norm:.4f}")
        print(f"Sensitivity  : {sensitivity:.4f}")
        print(f"Specificity  : {specificity:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_test, y_pred_test, target_names=class_names))
        print("Confusion Matrix:")
        print(cm)

        self._save_evaluation_plots(
            fpr, tpr, roc_auc_val, precision_val, recall_val,
            pr_auc, cm, chosen_thr, y_proba_test, y_true_test
        )
        self._plot_pr_curve(recall_val, precision_val, best_threshold, best_idx)
        self.plot_confusion_matrix(y_true_test, y_pred_test, chosen_thr)

        return {
            'threshold': float(best_threshold),
            'youden_threshold': float(chosen_thr),
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'auc_pr': pr_auc, 'auc_roc': roc_auc_val,
            'pr_auc_adjusted': pr_auc_adjusted,
            'partial_roc_auc': roc_auc_partial,
            'sensitivity': sensitivity, 'specificity': specificity,
            'y_pred': y_pred_test, 'y_proba': y_proba_test,
            'y_true_test': y_true_test, 'mode': chosen_mode
        }

    def _plot_pr_curve(self, recall, precision, thr, idx):
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.scatter(recall[:-1][idx], precision[:-1][idx], s=100, c='red', label=f'Threshold={thr:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            'Non-Physiological Artifact Detector\n'
            'PR Curve (Validation) — F1-Optimal Threshold'
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_pr_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_evaluation_plots(
        self, fpr, tpr, roc_auc, precision, recall, pr_auc, cm, threshold,
        y_proba_test, y_true_test
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        labels = ['Clean', _LABEL_ARTIFACT]

        axes[0, 0].plot(fpr, tpr, lw=3, label=f'ROC (AUC = {roc_auc:.3f})', color='blue')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].fill_between(fpr, tpr, alpha=0.2, color='blue')
        axes[0, 0].set_xlabel('FPR')
        axes[0, 0].set_ylabel('TPR')
        axes[0, 0].set_title('Non-Physiological Artifact Detector — ROC Curve (Validation)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(recall, precision, lw=3, label=f'PR (AUC = {pr_auc:.3f})', color='green')
        axes[0, 1].fill_between(recall, precision, alpha=0.2, color='green')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Non-Physiological Artifact Detector — PR Curve (Validation)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=labels, yticklabels=labels
        )
        axes[1, 0].set_title(
            f'Non-Physiological Artifact Detector — Confusion Matrix\nThreshold = {threshold:.3f}'
        )

        axes[1, 1].hist(
            y_proba_test[y_true_test == 0], bins=50, alpha=0.7, density=True,
            label='Clean', color='lightblue'
        )
        axes[1, 1].hist(
            y_proba_test[y_true_test == 1], bins=50, alpha=0.7, density=True,
            label=_LABEL_ARTIFACT, color='orange'
        )
        axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        axes[1, 1].set_xlabel('Prediction Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Non-Physiological Artifact Detector — Prediction Distribution (Test)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Non-Physiological Artifact Detector — Comprehensive Evaluation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f"{self.model_name}_comprehensive_evaluation.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, threshold):
        labels = ['Clean', _LABEL_ARTIFACT]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='magma',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='white'
        )
        plt.title(
            f'Non-Physiological Artifact Detector — Confusion Matrix\nThreshold = {threshold:.3f}',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f"{self.model_name}_confusion_matrix.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.show()

    def plot_training_history(self):
        if self.history is None:
            self.logger.warning("No training history available.")
            return

        hist = self.history.history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = [
            ('loss', 'Loss', 'blue', 'orange'),
            ('accuracy', 'Accuracy', 'green', 'red'),
            ('precision', 'Precision', 'purple', 'brown'),
            ('recall', 'Recall', 'teal', 'coral')
        ]
        for i, (metric, title, tc, vc) in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            ax.plot(hist[metric], label='Train', color=tc, linewidth=2)
            ax.plot(hist[f'val_{metric}'], label='Validation', color=vc, linewidth=2)
            ax.set_title(f'Non-Physiological Artifact Detector — Model {title}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Non-Physiological Artifact Detector — Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f"{self.model_name}_training_history.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.show()

    def save_model(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(self.results_dir, f"{self.model_name}.keras")
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
        if self.history:
            hist_path = os.path.join(self.results_dir, f"{self.model_name}_history.json")
            with open(hist_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            self.logger.info(f"Training history saved to {hist_path}")

    def load_model(
        self,
        filepath: Optional[str] = None,
        compile_model: bool = True,
    ) -> tf.keras.Model:
        """
        Load a saved Keras model from disk.

        Custom focal loss is re-applied from ``self.metadata`` when
        ``compile_model`` is True. Call :meth:`load_data` first so metadata
        is available, or rely on default focal parameters.

        Args:
            filepath: Path to ``.keras`` model file. Defaults to
                ``results/<model_name>/<model_name>.keras``.
            compile_model: If True, recompile with focal loss (recommended for training).

        Returns:
            The loaded Keras model.
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, f"{self.model_name}.keras")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")

        self.model = tf.keras.models.load_model(filepath, compile=False)

        if compile_model:
            focal_params = {'gamma': 2.0, 'alpha': 0.25}
            class_weights = {0: 1.0, 1: 1.0}
            if self.metadata is not None:
                focal_params = self.metadata.get('focal_loss_params', focal_params)
                class_weights = self.metadata.get('class_weights', class_weights)

            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-3),
                loss=focal_loss_with_class_weights(
                    alpha=focal_params['alpha'],
                    gamma=focal_params['gamma'],
                    class_weights=class_weights
                ),
                metrics=['accuracy', 'precision', 'recall']
            )
            self.logger.info(f"Model loaded and compiled from {filepath}")
        else:
            self.logger.info(f"Model loaded (compile=False) from {filepath}")

        return self.model
