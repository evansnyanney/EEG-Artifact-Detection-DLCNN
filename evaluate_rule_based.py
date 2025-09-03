#!/usr/bin/env python3
"""
Rule-Based Detector Evaluation Module

Evaluates the performance of heuristic-based artifact detectors on binary classification tasks:
- Eye Movement Detection
- Muscle Artifact Detection
- Non-Physiological Artifact Detection

Uses preprocessed test data from `binary_models_data/` and computes standard classification metrics.

Authors: Evans Nyanney, Zhaohui Geng, Parthasarathy Thirumala
Year: 2024
"""

import os
import json
import logging
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from rule_based_detectors import run_rules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

__all__ = ['evaluate_rule_based']


def _load_binary_dataset(model_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed 3D dataset for a binary model.

    Args:
        model_key: Subdirectory name in `binary_models_data/` (e.g., 'eye_movement').

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    base_dir = os.path.join('binary_models_data', model_key)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Model directory not found: {base_dir}")

    def load_path(name: str) -> np.ndarray:
        path = os.path.join(base_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return np.load(path)

    X_train = load_path('X_train_3d')
    X_val = load_path('X_val_3d')
    X_test = load_path('X_test_3d')
    y_train = load_path('y_train').astype(np.int32)
    y_val = load_path('y_val').astype(np.int32)
    y_test = load_path('y_test').astype(np.int32)

    logger.info(f"Loaded dataset '{model_key}' -> X_test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_rule_based(model_key: str, pretty_name: str, output_dir: str = 'results') -> Dict[str, Any]:
    """
    Evaluate a rule-based detector on the test set.

    Args:
        model_key: Model key (e.g., 'eye_movement').
        pretty_name: Human-readable name for logging.
        output_dir: Directory to save evaluation results.

    Returns:
        Dictionary of evaluation metrics and predictions.
    """
    logger.info(f"Evaluating rule-based detector: {pretty_name} ({model_key})")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = _load_binary_dataset(model_key)
    except Exception as e:
        logger.error(f"Failed to load dataset for {model_key}: {e}")
        return {}

    # Run rule-based detection
    try:
        y_pred_val = run_rules(X_val, target=model_key)
        y_pred_test = run_rules(X_test, target=model_key)
    except Exception as e:
        logger.error(f"Failed to run rules on {model_key}: {e}")
        return {}

    # Compute metrics
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    # ROC AUC (using val predictions as proxy scores)
    try:
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = float('nan')

    # Print results
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
    print(classification_report(
        y_test, y_pred_test,
        digits=4,
        target_names=['Clean', 'Artifact']
    ))

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'model': model_key,
        'pretty_name': pretty_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'y_true': y_test.tolist(),
        'y_pred': y_pred_test.tolist()
    }

    result_path = os.path.join(output_dir, f'rule_based_{model_key}_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {result_path}")

    return results


def main() -> None:
    """Evaluate all three rule-based detectors."""
    logger.info("Starting rule-based evaluation pipeline")

    models = [
        ('eye_movement', 'Eye Movement'),
        ('muscle_artifacts', 'Muscle Artifacts'),
        ('non_physiological', 'Non-Physiological')
    ]

    all_results = {}
    for model_key, pretty_name in models:
        results = evaluate_rule_based(model_key, pretty_name)
        all_results[model_key] = results

    # Final summary
    print("\n" + "=" * 80)
    print("RULE-BASED EVALUATION COMPLETED")
    print("=" * 80)
    for model_key, pretty_name in models:
        res = all_results.get(model_key, {})
        f1 = res.get('f1_score', float('nan'))
        print(f"{pretty_name:20s} : F1 = {f1:.4f}")

    logger.info("All evaluations completed.")


if __name__ == "__main__":
    main()