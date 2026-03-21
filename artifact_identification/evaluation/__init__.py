"""Evaluation modules for artifact detection models."""

from .cnn_vs_rules import evaluate_model
from .rule_based_eval import evaluate_rule_based

__all__ = ['evaluate_model', 'evaluate_rule_based']
