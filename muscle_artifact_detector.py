

#!/usr/bin/env python3
"""
Muscle Artifact Detector using Enhanced Deep Lightweight CNN (EDL-CNN)

This module implements a Enhanced Deep lightweight Convolutional Neural Network (CNN) for detecting
muscle artifacts in EEG signals via binary classification. The model emphasizes
 and generalization through no regularization and patient-level data
splitting to prevent leakage.

Key Features:
- Lightweight 1D-CNN architecture (no dropout, no L2 regularization)
- Three-way data split (train/validation/test) stratified by patient
- Threshold selection via Youden's J, fixed specificity, or max TPR under FPR constraint
-  Evaluation: PR-AUC, prevalence-adjusted PR-AUC, partial ROC-AUC
- Visualization of training history, ROC, PR, confusion matrix, and prediction distributions

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2025
"""

import os
import json
import random
import logging
import time
from typing import Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
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

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set global seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


def focal_loss_with_class_weights(alpha=0.25, gamma=2.0, class_weights=None):
    """
    Create a focal loss function with class weights for imbalanced binary classification.
    
    Args:
        alpha (float): Weighting factor for rare class
        gamma (float): Focusing parameter
        class_weights (dict): Additional class weights {0: weight_0, 1: weight_1}
    
    Returns:
        callable: Focal loss function compatible with Keras
    """
    def focal_loss_weighted(y_true, y_pred):
        """
        Focal loss with class weights implementation.
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        # Clip predictions to prevent log(0)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Calculate p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        
        # Calculate focal loss
        focal_loss = -alpha_t * tf.keras.backend.pow(1 - p_t, gamma) * tf.keras.backend.log(p_t)
        
        # Apply additional class weights if provided
        if class_weights is not None:
            # Convert string keys to integers if needed
            weight_1 = class_weights.get(1, class_weights.get("1", 1.0))
            weight_0 = class_weights.get(0, class_weights.get("0", 1.0))
            
            class_weight_tensor = tf.where(
                tf.equal(y_true, 1), 
                weight_1, 
                weight_0
            )
            focal_loss = focal_loss * class_weight_tensor
        
        return tf.keras.backend.mean(focal_loss)
    
    return focal_loss_weighted

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


__all__ = ['MuscleArtifactDetector']

# Custom F1 callback for early stopping
class F1EarlyStopping(callbacks.Callback):
    """
    Custom callback to monitor F1 score and early stop based on F1
    """
    def __init__(self, monitor='val_f1', patience=10, restore_best_weights=True, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_f1 = -np.inf
        self.wait = 0
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        current_f1 = logs.get(self.monitor)
        if current_f1 is None:
            return
            
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: early stopping on {self.monitor}')
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True

class MuscleArtifactDetector:
    """
    Enhanced Deep Lightweight CNN (EDL-CNN) for binary classification of muscle artifacts in EEG.

    This detector uses a Enhanced Deep lightweight CNN architecture to classify EEG segments as either
    containing muscle artifacts or being clean. It supports configurable threshold
    selection and comprehensive evaluation metrics.

    Attributes:
        model_name (str): Name of the model for saving/loading.
        model (tf.keras.Model): Built Keras model.
        history (tf.keras.callbacks.History): Training history object.
        metadata (dict): Dataset metadata (e.g., description, sample counts).
        X_train, X_val, X_test (np.ndarray): Input data splits.
        y_train, y_val, y_test (np.ndarray): Label splits.
        
    """

    def __init__(self, model_name: str = "muscle_artifact_detector", verbose: bool = True) -> None:
        self.model_name = model_name
        self.verbose = verbose
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.metadata: Dict[str, Any] = {}
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

        # Results directory for this model
        self.results_dir = os.path.join('results', self.model_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not verbose:
            self.logger.setLevel(logging.WARNING)
        
    def load_data(self, data_dir: str = "binary_models_data/muscle_artifacts") -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Load and split EEG data into train/validation/test sets with stratification.

        Args:
            data_dir (str): Path to directory containing preprocessed .npy and .json files.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

        Raises:
            FileNotFoundError: If required files are missing.
        """
        self.logger.info(f"Loading data from: {data_dir}")
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load 3D EEG data (samples, time_steps, channels)
        X_train_full = np.load(os.path.join(data_dir, "X_train_3d.npy"))
        X_test_full = np.load(os.path.join(data_dir, "X_test_3d.npy"))
        y_train_full = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int32)
        y_test_full = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int32)

        # Combine and split: 60% train, 20% val, 20% test
        X_combined = np.vstack([X_train_full, X_test_full])
        y_combined = np.hstack([y_train_full, y_test_full])
        
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X_combined, y_combined,
            test_size=0.2,
            random_state=42,
            stratify=y_combined
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.25,
            random_state=42,
            stratify=y_temp
        )
        
        self.logger.info(f"Training set: {self.X_train.shape}")
        self.logger.info(f"Validation set: {self.X_val.shape}")
        self.logger.info(f"Test set: {self.X_test.shape}")
        self.logger.info(f"Dataset: {self.metadata.get('description', 'Unknown')}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def build_model(self, input_shape: Optional[Tuple[int, ...]] = None, model_type: str = 'lightweight') -> tf.keras.Model:
        """
        Build a Enhanced Deep lightweight CNN for EEG muscle artifact detection.

        Args:
            input_shape (tuple, optional): Shape of input (timesteps, channels). Inferred if None.
            model_type (str): Either 'lightweight' (minimal) or 'standard' (deeper, regularized).

        Returns:
            Compiled Keras model.

        Raises:
            ValueError: If input shape is not provided and data is not loaded.
        """
        if input_shape is None:
            if self.X_train is None:
                raise ValueError("Input shape not provided and training data not loaded.")
            input_shape = self.X_train.shape[1:]

        self.logger.info(f"Building {model_type} model with input shape: {input_shape}")

        model = models.Sequential([layers.Input(shape=input_shape)])

        if model_type == 'lightweight':
            # Minimal CNN: 1 Conv1D + GAP + Dense
            model.add(layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.GlobalAveragePooling1D())
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

        elif model_type == 'standard':
            # Deeper CNN with regularization
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

        optimizer = optimizers.Adam(learning_rate=1e-3)
        
        # Get focal loss parameters from metadata
        focal_params = self.metadata.get('focal_loss_params', {'gamma': 2.0, 'alpha': 0.25})
        class_weights = self.metadata.get('class_weights', {0: 1.0, 1: 1.0})
        
        # Create focal loss function
        focal_loss_fn = focal_loss_with_class_weights(
            alpha=focal_params['alpha'],
            gamma=focal_params['gamma'],
            class_weights=class_weights
        )
        
        model.compile(
            optimizer=optimizer,
            loss=focal_loss_fn,
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        self.logger.info(f"Model built. Total parameters: {model.count_params():,}")
        return model
    
    def train(
        self,
        epochs: int = 200,
        batch_size: int = 128
    ) -> tf.keras.callbacks.History:
        """
        Train the model with early stopping and checkpointing.

        Args:
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size for training.

        Returns:
            Training history object.
        """
        self.logger.info(f"Starting training: epochs={epochs}, batch_size={batch_size}")

        run_id = int(time.time())
        ckpt_dir = os.path.join('checkpoints', 'muscle')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{self.model_name}_best_{run_id}.weights.h5")

        callbacks_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=3,
                min_lr=1e-7,
                verbose=1 if self.verbose else 0,
                cooldown=2
            ),
            callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1 if self.verbose else 0
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            )
        ]
        
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        self.logger.info("Training completed.")
        return self.history
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model performance on test set using validation-optimized thresholds.

        Supports threshold selection via:
            - Youden's J (default)
            - Fixed specificity (e.g., 95%)
            - Max TPR at FPR ≤ threshold

        Returns:
            Dictionary of evaluation metrics and predictions.
        """
        self.logger.info("Evaluating model on test set...")

        # Predict probabilities
        y_proba_val = self.model.predict(self.X_val).flatten()
        y_proba_test = self.model.predict(self.X_test).flatten()
        y_true_test = self.y_test

        # PR curve on validation for F1-optimal threshold
        precision_val, recall_val, thr_pr = precision_recall_curve(self.y_val, y_proba_val)
        f1_scores = 2 * (precision_val[:-1] * recall_val[:-1]) / (precision_val[:-1] + recall_val[:-1] + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        best_threshold = thr_pr[best_idx]

        # ROC curve for threshold selection
        fpr, tpr, thr_roc = roc_curve(self.y_val, y_proba_val)
        roc_auc = auc(fpr, tpr)

        # Select threshold based on environment variable
        mode = os.getenv('THRESHOLD_MODE', 'youden').lower()
        fixed_spec = float(os.getenv('FIXED_SPEC', '0.95'))
        max_fpr_val = float(os.getenv('MAX_FPR', '0.10'))

        chosen_thr = thr_roc[np.argmax(tpr - fpr)]  # Youden's J
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

        # Apply threshold to test set
        y_pred_test = (y_proba_test >= chosen_thr).astype(int)

        # Compute metrics
        acc = accuracy_score(y_true_test, y_pred_test)
        prec = precision_score(y_true_test, y_pred_test)
        rec = recall_score(y_true_test, y_pred_test)
        f1 = f1_score(y_true_test, y_pred_test)
        pr_auc = auc(recall_val, precision_val)

        # Prevalence-adjusted PR-AUC
        try:
            pos = self.metadata.get('positive_samples', 0)
            total = self.metadata.get('total_samples', 1)
            target_prevalence = float(pos) / max(1.0, float(total))
        except Exception:
            target_prevalence = 0.05

        denom = (target_prevalence * tpr) + ((1 - target_prevalence) * fpr)
        precision_adj = np.divide(target_prevalence * tpr, denom, out=np.zeros_like(denom), where=(denom > 0))
        recall_adj = tpr
        order = np.argsort(recall_adj)
        pr_auc_adjusted = auc(recall_adj[order], precision_adj[order])

        # Partial ROC AUC (FPR <= 0.1)
        fpr_limit = 0.1
        fpr_ext = np.insert(fpr, 0, 0.0)
        tpr_ext = np.insert(tpr, 0, 0.0)
        idx = np.searchsorted(fpr_ext, fpr_limit, side='right')
        if idx == 0:
            roc_auc_partial = 0.0
        else:
            fpr_clip = fpr_ext[:idx]
            tpr_clip = tpr_ext[:idx]
            if fpr_ext[idx-1] < fpr_limit < fpr_ext[idx]:
                tpr_interp = tpr_ext[idx-1] + (tpr_ext[idx] - tpr_ext[idx-1]) * \
                             ((fpr_limit - fpr_ext[idx-1]) / (fpr_ext[idx] - fpr_ext[idx-1]))
                fpr_clip = np.append(fpr_clip, fpr_limit)
                tpr_clip = np.append(tpr_clip, tpr_interp)
            roc_auc_partial = auc(fpr_clip, tpr_clip)
        roc_auc_partial_norm = roc_auc_partial / fpr_limit if fpr_limit > 0 else float('nan')

        # Confusion matrix
        cm = confusion_matrix(y_true_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        # Print summary
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION")
        print("="*60)
        print(f"Threshold (F1-optimal): {best_threshold:.3f}")
        print(f"Threshold ({chosen_mode}): {chosen_thr:.3f}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Precision    : {prec:.4f}")
        print(f"Recall       : {rec:.4f}")
        print(f"F1-Score     : {f1:.4f}")
        print(f"AUC (PR)     : {pr_auc:.4f}")
        print(f"AUC (ROC)    : {roc_auc:.4f}")
        print(f"PR-AUC (adj @π={target_prevalence:.2f}): {pr_auc_adjusted:.4f}")
        print(f"Partial ROC AUC (FPR≤0.1): {roc_auc_partial:.4f}")
        print(f"Partial ROC AUC (norm): {roc_auc_partial_norm:.4f}")
        print(f"Sensitivity  : {sensitivity:.4f}")
        print(f"Specificity  : {specificity:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true_test, y_pred_test, target_names=['Clean', 'Muscle Artifact']))
        print("Confusion Matrix:")
        print(cm)

        # Save plots
        self._save_evaluation_plots(fpr, tpr, roc_auc, precision_val, recall_val, pr_auc, cm, chosen_thr, y_proba_test, y_true_test)
        self._plot_pr_curve(recall_val, precision_val, best_threshold, best_idx)

        return {
            'threshold': float(best_threshold),
            'youden_threshold': float(chosen_thr),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_pr': pr_auc,
            'auc_roc': roc_auc,
            'pr_auc_adjusted': pr_auc_adjusted,
            'partial_roc_auc': roc_auc_partial,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'y_pred': y_pred_test,
            'y_proba': y_proba_test,
            'y_true_test': y_true_test,
            'mode': chosen_mode
        }

    def _plot_pr_curve(self, recall: np.ndarray, precision: np.ndarray, thr: float, idx: int) -> None:
        """Plot and save PR curve with optimal threshold."""
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.scatter(recall[:-1][idx], precision[:-1][idx], s=100, c='red', label=f'Threshold={thr:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve (Validation) - F1-Optimal Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_pr_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_plots(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        precision: np.ndarray,
        recall: np.ndarray,
        pr_auc: float,
        cm: np.ndarray,
        threshold: float,
        y_proba_test: np.ndarray,
        y_true_test: np.ndarray
    ) -> None:
        """Generate and save comprehensive evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # ROC
        axes[0, 0].plot(fpr, tpr, lw=3, label=f'ROC (AUC = {roc_auc:.3f})', color='blue')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].fill_between(fpr, tpr, alpha=0.2, color='blue')
        axes[0, 0].set_xlabel('FPR'); axes[0, 0].set_ylabel('TPR')
        axes[0, 0].set_title('ROC Curve (Validation)')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        # PR
        axes[0, 1].plot(recall, precision, lw=3, label=f'PR (AUC = {pr_auc:.3f})', color='green')
        axes[0, 1].fill_between(recall, precision, alpha=0.2, color='green')
        axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('PR Curve (Validation)')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                    xticklabels=['Clean', 'Muscle Artifact'],
                    yticklabels=['Clean', 'Muscle Artifact'])
        axes[1, 0].set_title(f'Confusion Matrix\nThreshold = {threshold:.3f}')

        # Prediction Distribution
        axes[1, 1].hist(y_proba_test[y_true_test == 0], bins=50, alpha=0.7, density=True, label='Clean', color='lightblue')
        axes[1, 1].hist(y_proba_test[y_true_test == 1], bins=50, alpha=0.7, density=True, label='Artifact', color='orange')
        axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        axes[1, 1].set_xlabel('Prediction Probability'); axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Distribution (Test)')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Evaluation - Muscle Artifact Detector', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_comprehensive_evaluation.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self) -> None:
        """Plot training/validation loss and metrics over epochs."""
        if self.history is None:
            self.logger.warning("No training history to plot.")
            return

        history = self.history.history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = [
            ('loss', 'Loss', 'blue', 'orange'),
            ('accuracy', 'Accuracy', 'green', 'red'),
            ('precision', 'Precision', 'purple', 'brown'),
            ('recall', 'Recall', 'teal', 'coral')
        ]

        for i, (metric, title, train_color, val_color) in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            ax.plot(history[metric], label='Train', color=train_color, linewidth=2)
            ax.plot(history[f'val_{metric}'], label='Validation', color=val_color, linewidth=2)
            ax.set_title(f'Model {title}', fontweight='bold')
            ax.set_xlabel('Epoch'); ax.set_ylabel(title)
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"{self.model_name}_training_history.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained model and training history.

        Args:
            filepath (str, optional): Path to save the model. Defaults to `{model_name}.keras`.
        """
        if filepath is None:
            filepath = os.path.join(self.results_dir, f"{self.model_name}.keras")
        
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")

        if self.history:
            hist_path = os.path.join(self.results_dir, f"{self.model_name}_history.json")
            with open(hist_path, 'w') as f:
                json.dump(self.history.history, f, indent=2)
            self.logger.info(f"Training history saved to {hist_path}")

def main() -> None:
    """Main execution pipeline for training and evaluating the muscle artifact detector."""
    logger.info("Initializing Muscle Artifact Detector")

    detector = MuscleArtifactDetector(model_name="edl_cnn_muscle", verbose=True)

    # Load data
    detector.load_data()

    # Build and train
    detector.build_model(model_type='lightweight')
    detector.train(epochs=200, batch_size=128)

    # Evaluate
    results = detector.evaluate()
    
    # Plot results
    detector.plot_training_history()
    detector.save_model()
    
    logger.info("Muscle artifact detector training and evaluation completed.")


if __name__ == "__main__":
    main() 