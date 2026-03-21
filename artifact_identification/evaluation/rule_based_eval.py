#!/usr/bin/env python3
"""
Rule-Based Detector Evaluation

Evaluates heuristic artifact detectors on binary test data.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import os
import json
import logging
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
)

from ..detectors.rule_based import run_rules

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['evaluate_rule_based']


def _load_binary_dataset(model_key: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    base_dir = os.path.join('binary_models_data', model_key)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Model directory not found: {base_dir}")

    def load_path(name):
        path = os.path.join(base_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return np.load(path)

    return (
        load_path('X_train_3d'), load_path('X_val_3d'), load_path('X_test_3d'),
        load_path('y_train').astype(np.int32),
        load_path('y_val').astype(np.int32),
        load_path('y_test').astype(np.int32),
    )


def evaluate_rule_based(
    model_key: str, pretty_name: str, output_dir: str = 'results'
) -> Dict[str, Any]:
    """
    Evaluate a rule-based detector on the test set.

    Args:
        model_key: Model key (e.g., ``'eye_movement'``).
        pretty_name: Human-readable name for logging.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"Evaluating rule-based detector: {pretty_name} ({model_key})")

    try:
        _, X_val, X_test, _, y_val, y_test = _load_binary_dataset(model_key)
        logger.info(f"  Loaded dataset: val={X_val.shape}, test={X_test.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset for {model_key}: {e}")
        return {}

    try:
        logger.info(f"  Running rule-based detection on validation set ({X_val.shape[0]} samples)...")
        y_pred_val = run_rules(X_val, target=model_key)
        logger.info(f"  Running rule-based detection on test set ({X_test.shape[0]} samples)...")
        y_pred_test = run_rules(X_test, target=model_key)
        logger.info(f"  Detection complete. Computing metrics...")
    except Exception as e:
        logger.error(f"Failed to run rules on {model_key}: {e}")
        return {}

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    try:
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = float('nan')

    print("\n" + "=" * 80)
    print(f"RULE-BASED EVALUATION - {pretty_name.upper()}")
    print("=" * 80)
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"ROC AUC (val): {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC AUC: N/A")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, digits=4, target_names=['Clean', 'Artifact']))

    os.makedirs(output_dir, exist_ok=True)
    results = {
        'model': model_key, 'pretty_name': pretty_name,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'specificity': specificity, 'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
    }

    result_path = os.path.join(output_dir, f'rule_based_{model_key}_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {result_path}")

    return results
