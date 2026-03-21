"""Tests for the shared focal loss function."""

import numpy as np
import pytest
import tensorflow as tf

from artifact_identification.losses import focal_loss_with_class_weights


class TestFocalLoss:
    """Unit tests for focal_loss_with_class_weights."""

    def test_returns_callable(self):
        loss_fn = focal_loss_with_class_weights()
        assert callable(loss_fn)

    def test_loss_is_non_negative(self):
        loss_fn = focal_loss_with_class_weights(alpha=0.25, gamma=2.0)
        y_true = tf.constant([0.0, 1.0, 1.0, 0.0])
        y_pred = tf.constant([0.1, 0.9, 0.8, 0.2])
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() >= 0

    def test_perfect_predictions_low_loss(self):
        loss_fn = focal_loss_with_class_weights(alpha=0.25, gamma=2.0)
        y_true = tf.constant([0.0, 1.0, 1.0, 0.0])
        y_pred = tf.constant([0.01, 0.99, 0.99, 0.01])
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() < 0.1

    def test_wrong_predictions_high_loss(self):
        loss_fn = focal_loss_with_class_weights(alpha=0.25, gamma=2.0)
        y_true = tf.constant([0.0, 1.0, 1.0, 0.0])
        y_pred_good = tf.constant([0.01, 0.99, 0.99, 0.01])
        y_pred_bad = tf.constant([0.99, 0.01, 0.01, 0.99])
        loss_good = loss_fn(y_true, y_pred_good)
        loss_bad = loss_fn(y_true, y_pred_bad)
        assert loss_bad.numpy() > loss_good.numpy()

    def test_class_weights_affect_loss(self):
        loss_no_weights = focal_loss_with_class_weights(alpha=0.25, gamma=2.0)
        loss_with_weights = focal_loss_with_class_weights(
            alpha=0.25, gamma=2.0, class_weights={0: 1.0, 1: 5.0}
        )
        y_true = tf.constant([1.0, 1.0, 0.0, 0.0])
        y_pred = tf.constant([0.5, 0.5, 0.5, 0.5])
        l1 = loss_no_weights(y_true, y_pred)
        l2 = loss_with_weights(y_true, y_pred)
        assert not np.isclose(l1.numpy(), l2.numpy())

    def test_string_class_weight_keys(self):
        loss_fn = focal_loss_with_class_weights(
            alpha=0.25, gamma=2.0, class_weights={"0": 1.0, "1": 2.0}
        )
        y_true = tf.constant([1.0, 0.0])
        y_pred = tf.constant([0.7, 0.3])
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() >= 0
