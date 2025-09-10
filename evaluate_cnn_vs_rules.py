#!/usr/bin/env python3
"""
CNN vs Rule-Based Artifact Detection Evaluation

This module provides comprehensive evaluation comparing trained CNN models against
rule-based  detectors for EEG artifact detection. 

Key Features:
- Loads preprocessed binary datasets for three artifact types
- Finds and loads compatible CNN models automatically
- Calibrates CNN thresholds using validation data
- Evaluates CNN, rule-based, and ensemble methods
- Computes comprehensive performance metrics
- Provides detailed comparison analysis

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2025
"""

import os
import glob
import logging
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)

from rule_based_detectors import run_rules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['evaluate_model']


def _load_binary_dataset(model_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed binary dataset for evaluation.

    Args:
        model_key: Dataset identifier (e.g., 'eye_movement', 'muscle_artifacts', 'non_physiological').

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).

    Raises:
        FileNotFoundError: If required data files are missing.
    """
    base_dir = os.path.join('binary_models_data', model_key)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    def load_npy_file(filename: str) -> np.ndarray:
        filepath = os.path.join(base_dir, f"{filename}.npy")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing data file: {filepath}")
        return np.load(filepath)

    X_train = load_npy_file('X_train_3d')
    X_val = load_npy_file('X_val_3d')
    X_test = load_npy_file('X_test_3d')
    y_train = load_npy_file('y_train').astype(np.int32)
    y_val = load_npy_file('y_val').astype(np.int32)
    y_test = load_npy_file('y_test').astype(np.int32)

    logger.info(f"Loaded dataset '{model_key}': Test shape {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def _find_compatible_model(prefix: str, target_shape: Tuple[int, int]) -> Optional[str]:
    """
    Find compatible CNN model with matching input shape.

    Args:
        prefix: Model file prefix.
        target_shape: Expected input shape (timesteps, channels).

    Returns:
        Path to compatible model, or None if not found.
    """
    # Search for timestamped models first, then fallback
    candidates = sorted(glob.glob(f"{prefix}_*.keras"), reverse=True)
    fallback = sorted(glob.glob(f"{prefix}.keras"), reverse=True)
    all_paths = candidates + fallback

    for path in all_paths:
        try:
            model = tf.keras.models.load_model(path)
            model_shape = model.input_shape[1:3]  # (timesteps, channels)
            if model_shape == target_shape:
                logger.info(f"Found compatible model: {path}")
                return path
            else:
                logger.debug(f"Shape mismatch: {model_shape} != {target_shape} in {path}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return None


def _calibrate_threshold(y_true_val: np.ndarray, y_proba_val: np.ndarray) -> Tuple[float, str]:
    """
    Calibrate CNN threshold using validation data.

    Supports multiple threshold selection methods:
    - Youden's J statistic (default)
    - Fixed specificity
    - Maximum TPR at FPR constraint

    Args:
        y_true_val: True labels for validation set.
        y_proba_val: Predicted probabilities for validation set.

    Returns:
        Tuple of (threshold, method_name).
    """
    mode = os.getenv('THRESH_MODE', 'youden').lower()
    fpr, tpr, thr = roc_curve(y_true_val, y_proba_val)

    # Default: Youden's J statistic
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    selected = thr[best_idx]

    # Alternative threshold methods
    if mode == 'fixed_spec':
        try:
            target_spec = float(os.getenv('FIXED_SPEC', '0.95'))
        except ValueError:
            target_spec = 0.95
        target_fpr = max(0.0, min(1.0, 1.0 - target_spec))
        mask = fpr <= target_fpr
        if np.any(mask):
            sub_idx = int(np.argmax(tpr[mask]))
            selected = thr[mask][sub_idx]

    elif mode in ('max_tpr_at_fpr_0.1', 'maxtpr_at_fpr_0.1', 'tpr_at_fpr_0.1'):
        mask = fpr <= 0.1
        if np.any(mask):
            sub_idx = int(np.argmax(tpr[mask]))
            selected = thr[mask][sub_idx]

    return float(selected), mode


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Tuple of (accuracy, precision, recall, f1_score, specificity, confusion_matrix).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    return acc, prec, rec, f1, spec, cm


def evaluate_model(model_key: str, model_prefix: str, pretty_name: str) -> None:
    """
    Comprehensive evaluation comparing Enhanced Deep lightweight CNN vs rule-based artifact detection methods.

    This function evaluates the performance of trained Enhanced Deep lightweight CNN models against
    rule-based  detectors for EEG artifact detection tasks.

    Args:
        model_key: Dataset identifier (e.g., 'eye_movement', 'muscle_artifacts', 'non_physiological').
        model_prefix: Prefix for Enhanced Deep lightweight CNN model files.
        
    """
    logger.info(f"Starting Enhanced Deep lightweight CNN vs rule-based evaluation: {pretty_name}")
    
    # Load dataset
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _load_binary_dataset(model_key)
    except Exception as e:
        logger.error(f"Failed to load dataset for {model_key}: {e}")
        return

    logger.info(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    logger.info(f"Test distribution: Clean={np.sum(y_test == 0)}, Artifacts={np.sum(y_test == 1)}")

    # Find compatible CNN model
    target_shape = X_train.shape[1:]
    model_path = _find_compatible_model(model_prefix, target_shape)

    if not model_path:
        logger.error(f"No compatible CNN model found for {model_key} with shape {target_shape}")
        return

    # Load CNN model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded CNN model: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load CNN model {model_path}: {e}")
        return

    # CNN evaluation
    logger.info("Evaluating CNN performance...")
    y_proba_val = model.predict(X_val, verbose=0).flatten()
    y_proba_test = model.predict(X_test, verbose=0).flatten()
    threshold, mode = _calibrate_threshold(y_val, y_proba_val)
    y_pred_cnn = (y_proba_test >= threshold).astype(int)
    logger.info(f"CNN threshold ({mode}): {threshold:.3f}")
    logger.info(f"CNN predictions: Clean={np.sum(y_pred_cnn == 0)}, Artifacts={np.sum(y_pred_cnn == 1)}")

    # Rule-based evaluation
    logger.info("Evaluating rule-based methods...")
    try:
        y_pred_rules = run_rules(X_test, model_key)
        logger.info("Rule-based detection completed")
        logger.info(f"Rule predictions: Clean={np.sum(y_pred_rules == 0)}, Artifacts={np.sum(y_pred_rules == 1)}")
    except Exception as e:
        logger.error(f"Rule-based detection failed: {e}")
        return

    # Ensemble approach (CNN AND Rules)
    y_pred_ensemble = (y_pred_cnn & y_pred_rules).astype(int)
    logger.info(f"Ensemble predictions: Clean={np.sum(y_pred_ensemble == 0)}, Artifacts={np.sum(y_pred_ensemble == 1)}")

    # Calculate metrics for all methods
    acc_c, prec_c, rec_c, f1_c, spec_c, cm_c = _compute_metrics(y_test, y_pred_cnn)
    acc_r, prec_r, rec_r, f1_r, spec_r, cm_r = _compute_metrics(y_test, y_pred_rules)
    acc_e, prec_e, rec_e, f1_e, spec_e, cm_e = _compute_metrics(y_test, y_pred_ensemble)

    # Results display
    print(f"\n{'='*90}")
    print(f"CNN vs RULE-BASED COMPARISON - {pretty_name.upper()}")
    print(f"{'='*90}")
    print(f"Model: {model_path}")
    print(f"Threshold calibration: {mode}, threshold={threshold:.3f}")
    
    print(f"\nCNN PERFORMANCE:")
    print(f"{'─'*40}")
    print(f"  Accuracy   : {acc_c:.4f}")
    print(f"  Precision  : {prec_c:.4f}")
    print(f"  Recall     : {rec_c:.4f}")
    print(f"  Specificity: {spec_c:.4f}")
    print(f"  F1-Score   : {f1_c:.4f}")
    print(f"  Confusion Matrix: TN={cm_c[0,0]}, FP={cm_c[0,1]}, FN={cm_c[1,0]}, TP={cm_c[1,1]}")

    print(f"\nRULE-BASED PERFORMANCE:")
    print(f"{'─'*40}")
    print(f"  Accuracy   : {acc_r:.4f}")
    print(f"  Precision  : {prec_r:.4f}")
    print(f"  Recall     : {rec_r:.4f}")
    print(f"  Specificity: {spec_r:.4f}")
    print(f"  F1-Score   : {f1_r:.4f}")
    print(f"  Confusion Matrix: TN={cm_r[0,0]}, FP={cm_r[0,1]}, FN={cm_r[1,0]}, TP={cm_r[1,1]}")

    print(f"\nENSEMBLE (CNN AND RULES):")
    print(f"{'─'*40}")
    print(f"  Accuracy   : {acc_e:.4f}")
    print(f"  Precision  : {prec_e:.4f}")
    print(f"  Recall     : {rec_e:.4f}")
    print(f"  Specificity: {spec_e:.4f}")
    print(f"  F1-Score   : {f1_e:.4f}")
    print(f"  Confusion Matrix: TN={cm_e[0,0]}, FP={cm_e[0,1]}, FN={cm_e[1,0]}, TP={cm_e[1,1]}")

    # Head-to-head comparison
    print(f"\nHEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'CNN':<8} {'Rules':<8} {'Winner':<10} {'Difference'}")
    print(f"{'─'*60}")
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    cnn_values = [acc_c, prec_c, rec_c, f1_c, spec_c]
    rule_values = [acc_r, prec_r, rec_r, f1_r, spec_r]
    
    cnn_wins = 0
    rule_wins = 0
    
    for i, metric in enumerate(metrics_names):
        cnn_val = cnn_values[i]
        rule_val = rule_values[i]
        diff = cnn_val - rule_val
        
        if abs(diff) < 0.01:  # Here we consider ties for very small differences
            winner = "Tie"
        elif diff > 0:
            winner = "CNN"
            cnn_wins += 1
        else:
            winner = "Rules"
            rule_wins += 1
        
        print(f"{metric:<12} {cnn_val:.4f}   {rule_val:.4f}   {winner:<10} {diff:+.4f}")
    
    print(f"{'─'*60}")
    print(f"SUMMARY: CNN wins {cnn_wins}, Rule-based wins {rule_wins} out of {len(metrics_names)} metrics")
    
    # Performance overview (neutral)
    print(f"\nPERFORMANCE OVERVIEW")
    print(f"{'='*50}")
    ties = len(metrics_names) - (cnn_wins + rule_wins)
    print(f"Metrics won — CNN: {cnn_wins}, Rules: {rule_wins}, Ties: {ties}")
    print(f"Comparison above shows per-metric values and absolute differences.")
    print(f"{'='*90}")


def main() -> None:
    """Run comprehensive CNN vs rule-based evaluation for all artifact types."""
    logger.info("Starting comprehensive CNN vs rule-based comparison")
    
    # Evaluate all three artifact types
    evaluations = [
        ('eye_movement', 'eye_movement_detector', 'Eye Movement Artifacts'),
        ('muscle_artifacts', 'muscle_artifact_detector', 'Muscle/EMG Artifacts'),
        ('non_physiological', 'non_physiological_detector', 'Non-Physiological Artifacts')
    ]
    
    completed = []
    
    for model_key, model_prefix, pretty_name in evaluations:
        try:
            evaluate_model(model_key, model_prefix, pretty_name)
            completed.append(pretty_name)
        except Exception as e:
            logger.error(f"Evaluation failed for {pretty_name}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("CNN vs RULE-BASED COMPARISON COMPLETED")
    print(f"{'='*80}")
    if completed:
        print("Successfully evaluated:")
        for name in completed:
            print(f"  {name}")
        print(f"\nResults provide comprehensive comparison.")
    else:
        print("No evaluations completed.")


if __name__ == '__main__':
    main()