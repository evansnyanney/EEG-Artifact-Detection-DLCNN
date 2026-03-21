#!/usr/bin/env python3
"""
CNN vs Rule-Based Artifact Detection Evaluation

Compares trained DLCNN models against rule-based heuristic detectors
for EEG artifact detection across three artifact categories.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import glob
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)

from ..detectors.rule_based import run_rules
from ..losses import focal_loss_with_class_weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['evaluate_model']


def _load_binary_dataset(model_key: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """Load preprocessed binary dataset for evaluation."""
    base_dir = os.path.join('binary_models_data', model_key)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    def load_npy(name):
        path = os.path.join(base_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        return np.load(path)

    X_train = load_npy('X_train_3d')
    X_val = load_npy('X_val_3d')
    X_test = load_npy('X_test_3d')
    y_train = load_npy('y_train').astype(np.int32)
    y_val = load_npy('y_val').astype(np.int32)
    y_test = load_npy('y_test').astype(np.int32)

    logger.info(f"Loaded dataset '{model_key}': Test shape {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


_CUSTOM_OBJECTS = {'focal_loss_weighted': focal_loss_with_class_weights()}


def _load_model(prefix: str):
    """Load a CNN model from the given prefix path."""
    candidates = sorted(glob.glob(f"{prefix}_*.keras"), reverse=True)
    fallback = sorted(glob.glob(f"{prefix}.keras"), reverse=True)

    for path in candidates + fallback:
        try:
            model = tf.keras.models.load_model(path, custom_objects=_CUSTOM_OBJECTS)
            logger.info(f"Loaded model: {path}  input_shape={model.input_shape}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return None


def _predict_with_rewindow(model, X, y=None, verbose=0):
    """Run predictions, re-windowing if data and model window sizes differ.

    If data windows are longer than the model expects, each sample is split
    into sub-windows and the max probability is taken.  If data windows are
    shorter, consecutive samples are concatenated to build model-sized inputs
    (only when the model window is an exact multiple of the data window).

    Returns (probabilities, labels) when y is provided, otherwise just
    probabilities.
    """
    model_len = model.input_shape[1]
    data_len = X.shape[1]

    if model_len == data_len:
        probs = model.predict(X, verbose=verbose).flatten()
        return (probs, y) if y is not None else probs

    if data_len > model_len:
        if data_len % model_len != 0:
            trim = (data_len // model_len) * model_len
            X = X[:, :trim, :]

        n_sub = X.shape[1] // model_len
        logger.info(
            f"Re-windowing (split): {data_len} -> {n_sub} sub-windows of {model_len}"
        )
        n_samples, _, n_ch = X.shape
        X_sub = X.reshape(n_samples * n_sub, model_len, n_ch)
        p_sub = model.predict(X_sub, verbose=verbose).flatten()
        p_per_sample = p_sub.reshape(n_samples, n_sub)
        probs = p_per_sample.max(axis=1)
        return (probs, y) if y is not None else probs

    if model_len % data_len != 0:
        raise ValueError(
            f"Model expects {model_len} samples but data has {data_len}. "
            f"{model_len} is not a multiple of {data_len}, cannot concatenate."
        )

    group_size = model_len // data_len
    n_samples = X.shape[0]
    usable = (n_samples // group_size) * group_size
    if usable == 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) to form even one group of "
            f"{group_size} (needed for model window {model_len})."
        )
    if usable < n_samples:
        logger.warning(
            f"Dropping last {n_samples - usable} samples to form complete groups"
        )
    X_used = X[:usable]
    n_ch = X.shape[2]
    X_concat = X_used.reshape(usable // group_size, model_len, n_ch)
    logger.info(
        f"Re-windowing (concat): {n_samples} x {data_len} -> "
        f"{X_concat.shape[0]} x {model_len}"
    )
    probs = model.predict(X_concat, verbose=verbose).flatten()

    if y is not None:
        y_used = y[:usable].reshape(usable // group_size, group_size)
        y_grouped = (y_used.max(axis=1)).astype(np.int32)
        return probs, y_grouped
    return probs


def _calibrate_threshold(y_true_val, y_proba_val):
    """Calibrate CNN threshold on validation data using Youden's J or env overrides."""
    mode = os.getenv('THRESH_MODE', 'youden').lower()
    fpr, tpr, thr = roc_curve(y_true_val, y_proba_val)

    best_idx = int(np.argmax(tpr - fpr))
    selected = thr[best_idx]

    if mode == 'fixed_spec':
        target_spec = float(os.getenv('FIXED_SPEC', '0.95'))
        mask = fpr <= max(0.0, min(1.0, 1.0 - target_spec))
        if np.any(mask):
            selected = thr[mask][int(np.argmax(tpr[mask]))]
    elif mode in ('max_tpr_at_fpr_0.1', 'maxtpr_at_fpr_0.1', 'tpr_at_fpr_0.1'):
        mask = fpr <= 0.1
        if np.any(mask):
            selected = thr[mask][int(np.argmax(tpr[mask]))]

    return float(selected), mode


def _compute_metrics(y_true, y_pred):
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
    Compare DLCNN vs rule-based detection for a given artifact type.

    Args:
        model_key: Dataset identifier (e.g., ``'eye_movement'``).
        model_prefix: Prefix for CNN model files.
        pretty_name: Human-readable name for display.
    """
    logger.info(f"Starting CNN vs rule-based evaluation: {pretty_name}")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _load_binary_dataset(model_key)
    except Exception as e:
        logger.error(f"Failed to load dataset for {model_key}: {e}")
        return

    model = _load_model(model_prefix)
    if model is None:
        logger.error(f"No CNN model found for {model_key} (prefix={model_prefix})")
        return

    y_proba_val, y_val_adj = _predict_with_rewindow(model, X_val, y_val, verbose=0)
    y_proba_test, y_test_adj = _predict_with_rewindow(model, X_test, y_test, verbose=0)
    threshold, mode = _calibrate_threshold(y_val_adj, y_proba_val)
    y_pred_cnn = (y_proba_test >= threshold).astype(int)

    try:
        y_pred_rules = run_rules(X_test, model_key)
    except Exception as e:
        logger.error(f"Rule-based detection failed: {e}")
        return

    model_len = model.input_shape[1]
    data_len = X_test.shape[1]
    if data_len < model_len:
        group_size = model_len // data_len
        usable = (len(y_pred_rules) // group_size) * group_size
        y_pred_rules = y_pred_rules[:usable].reshape(-1, group_size).max(axis=1)

    y_pred_ensemble = (y_pred_cnn & y_pred_rules).astype(int)

    acc_c, prec_c, rec_c, f1_c, spec_c, cm_c = _compute_metrics(y_test_adj, y_pred_cnn)
    acc_r, prec_r, rec_r, f1_r, spec_r, cm_r = _compute_metrics(y_test_adj, y_pred_rules)
    acc_e, prec_e, rec_e, f1_e, spec_e, cm_e = _compute_metrics(y_test_adj, y_pred_ensemble)

    print(f"\n{'=' * 90}")
    print(f"CNN vs RULE-BASED COMPARISON - {pretty_name.upper()}")
    print(f"{'=' * 90}")

    for label, vals, cm_val in [
        ('CNN', (acc_c, prec_c, rec_c, f1_c, spec_c), cm_c),
        ('RULE-BASED', (acc_r, prec_r, rec_r, f1_r, spec_r), cm_r),
        ('ENSEMBLE (CNN AND RULES)', (acc_e, prec_e, rec_e, f1_e, spec_e), cm_e),
    ]:
        print(f"\n{label} PERFORMANCE:")
        print(f"{'-' * 40}")
        for name, val in zip(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'], vals):
            print(f"  {name:12s}: {val:.4f}")
        print(f"  Confusion Matrix: TN={cm_val[0,0]}, FP={cm_val[0,1]}, FN={cm_val[1,0]}, TP={cm_val[1,1]}")

    print(f"\nHEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Metric':<12} {'CNN':<8} {'Rules':<8} {'Winner':<10} {'Difference'}")
    print(f"{'-' * 60}")

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    cnn_vals = [acc_c, prec_c, rec_c, f1_c, spec_c]
    rule_vals = [acc_r, prec_r, rec_r, f1_r, spec_r]
    cnn_wins = rule_wins = 0

    for name, cv, rv in zip(metrics_names, cnn_vals, rule_vals):
        diff = cv - rv
        if abs(diff) < 0.01:
            winner = "Tie"
        elif diff > 0:
            winner = "CNN"; cnn_wins += 1
        else:
            winner = "Rules"; rule_wins += 1
        print(f"{name:<12} {cv:.4f}   {rv:.4f}   {winner:<10} {diff:+.4f}")

    print(f"{'-' * 60}")
    print(f"SUMMARY: CNN wins {cnn_wins}, Rule-based wins {rule_wins} out of {len(metrics_names)} metrics")
    print(f"{'=' * 90}")
