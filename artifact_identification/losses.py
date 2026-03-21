"""
Shared loss functions for EEG artifact detection models.

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import tensorflow as tf


def focal_loss_with_class_weights(alpha=0.25, gamma=2.0, class_weights=None):
    """
    Focal loss with optional class weights for imbalanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the rare class.
        gamma: Focusing parameter that down-weights easy examples.
        class_weights: Optional dict ``{0: w0, 1: w1}`` for additional
            per-class weighting.

    Returns:
        A Keras-compatible loss function.
    """

    def focal_loss_weighted(y_true, y_pred):
        y_pred = tf.keras.backend.clip(
            y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
        )
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_loss = -alpha_t * tf.keras.backend.pow(1 - p_t, gamma) * tf.keras.backend.log(p_t)

        if class_weights is not None:
            weight_1 = class_weights.get(1, class_weights.get("1", 1.0))
            weight_0 = class_weights.get(0, class_weights.get("0", 1.0))
            class_weight_tensor = tf.where(tf.equal(y_true, 1), weight_1, weight_0)
            focal_loss = focal_loss * class_weight_tensor

        return tf.keras.backend.mean(focal_loss)

    return focal_loss_weighted
