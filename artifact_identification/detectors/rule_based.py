#!/usr/bin/env python3
"""
Rule-Based Artifact Detection for Preprocessed EEG (TUH-EEG)

Heuristic detectors for eye movement, muscle, and non-physiological artifacts
in preprocessed EEG data (RobustScaler, 1-40 Hz filtering, average reference).

Authors: Evans Nyanney, Parthasarathy D Thirumala, Shyam Visweswaran, Zhaohui Geng
Year: 2025
License: MIT
"""

import numpy as np
from scipy.signal import welch
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

__all__ = [
    'detect_eye_movement_tuh_adapted',
    'detect_muscle_artifacts_tuh_adapted',
    'detect_non_physiological_tuh_adapted',
    'run_rules',
]


def _bandpower(signal: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    """
    Compute power in a frequency band using Welch's method.

    Args:
        signal: 1D time series.
        fs: Sampling frequency in Hz.
        f_low: Lower frequency bound.
        f_high: Upper frequency bound.

    Returns:
        Band power (float).
    """
    if len(signal) < 4:
        return 0.0
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), 128), nfft=len(signal))
        band_mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(band_mask):
            return 0.0
        return float(np.trapz(psd[band_mask], freqs[band_mask]))
    except Exception:
        return 0.0


def detect_eye_movement_tuh_adapted(X: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Detect eye movements using frontal-channel low-frequency dominance (0.5-4 Hz).

    Args:
        X: 3D array (n_windows, n_timepoints, n_channels).
        fs: Sampling frequency in Hz.

    Returns:
        Binary labels (1 = eye movement, 0 = clean).
    """
    n, t, c = X.shape
    labels = np.zeros(n, dtype=np.int32)
    frontal_channels = list(range(min(8, c)))

    low_freq_ratios = []
    amplitudes = []
    for i in range(n):
        win = X[i]
        for ch in frontal_channels:
            sig = win[:, ch]
            low_power = _bandpower(sig, fs, 0.5, 4.0)
            total_power = _bandpower(sig, fs, 0.5, 30.0)
            low_freq_ratios.append(low_power / (total_power + 1e-12))
            amplitudes.append(np.max(np.abs(sig)))

    amp_threshold = np.percentile(amplitudes, 75)
    ratio_threshold = np.percentile(low_freq_ratios, 70)

    for i in range(n):
        win = X[i]
        eye_count = 0
        for ch in frontal_channels:
            sig = win[:, ch]
            max_amp = np.max(np.abs(sig))
            low_power = _bandpower(sig, fs, 0.5, 4.0)
            total_power = _bandpower(sig, fs, 0.5, 30.0)
            ratio = low_power / (total_power + 1e-12)
            if max_amp > amp_threshold and ratio > ratio_threshold:
                eye_count += 1
        if eye_count >= 1:
            labels[i] = 1

    return labels


def detect_muscle_artifacts_tuh_adapted(X: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Detect muscle artifacts using high-frequency content (20-40 Hz) and variance.

    Args:
        X: 3D array (n_windows, n_timepoints, n_channels).
        fs: Sampling frequency in Hz.

    Returns:
        Binary labels (1 = muscle artifact, 0 = clean).
    """
    n, t, c = X.shape
    labels = np.zeros(n, dtype=np.int32)

    hf_ratios = []
    variances = []
    for i in range(n):
        win = X[i]
        for ch in range(c):
            sig = win[:, ch]
            hf_power = _bandpower(sig, fs, 20.0, 40.0)
            total_power = _bandpower(sig, fs, 1.0, 40.0)
            hf_ratios.append(hf_power / (total_power + 1e-12))
            variances.append(np.var(sig))

    hf_threshold = np.percentile(hf_ratios, 80)
    var_threshold = np.percentile(variances, 75)

    for i in range(n):
        win = X[i]
        emg_score = 0
        for ch in range(c):
            sig = win[:, ch]
            hf_power = _bandpower(sig, fs, 20.0, 40.0)
            total_power = _bandpower(sig, fs, 1.0, 40.0)
            hf_ratio = hf_power / (total_power + 1e-12)
            variance = np.var(sig)
            if hf_ratio > hf_threshold and variance > var_threshold:
                emg_score += 1
        if emg_score >= max(2, c // 6):
            labels[i] = 1

    return labels


def detect_non_physiological_tuh_adapted(X: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Detect non-physiological artifacts (flatlines, steps, outliers, saturation).

    Args:
        X: 3D array (n_windows, n_timepoints, n_channels).
        fs: Sampling frequency in Hz.

    Returns:
        Binary labels (1 = artifact, 0 = clean).
    """
    n, t, c = X.shape
    labels = np.zeros(n, dtype=np.int32)

    variances = []
    gradients = []
    outlier_ratios = []
    for i in range(n):
        win = X[i]
        for ch in range(c):
            sig = win[:, ch]
            variances.append(np.var(sig))
            gradients.append(np.mean(np.abs(np.diff(sig))))
            std = np.std(sig)
            outlier_ratios.append(np.sum(np.abs(sig) > 3 * std) / len(sig) if std > 0 else 0)

    var_low = np.percentile(variances, 5)
    grad_high = np.percentile(gradients, 98)
    outlier_thresh = np.percentile(outlier_ratios, 95)

    for i in range(n):
        win = X[i]
        artifact_count = 0
        for ch in range(c):
            sig = win[:, ch]
            is_flat = np.var(sig) < var_low
            has_step = np.mean(np.abs(np.diff(sig))) > grad_high
            std = np.std(sig)
            has_outliers = (np.sum(np.abs(sig) > 3 * std) / len(sig)) > outlier_thresh if std > 0 else False
            is_saturated = (
                (np.sum(sig == np.max(sig)) > 0.1 * len(sig)) or
                (np.sum(sig == np.min(sig)) > 0.1 * len(sig))
            )
            if is_flat or has_step or has_outliers or is_saturated:
                artifact_count += 1
        if artifact_count >= max(2, c // 6):
            labels[i] = 1

    return labels


def run_rules(X: np.ndarray, target: str, fs: float = 250.0) -> np.ndarray:
    """
    Run rule-based detection for a specified artifact type.

    Args:
        X: 3D EEG data (n_windows, n_timepoints, n_channels).
        target: One of ``'eye'``, ``'muscle'``, ``'nonphys'`` (and common aliases).
        fs: Sampling frequency.

    Returns:
        Binary prediction array.

    Raises:
        ValueError: If target is not supported.
    """
    target = target.lower().strip()
    if target in ('eye', 'eye_movement', 'eye-movement'):
        return detect_eye_movement_tuh_adapted(X, fs)
    elif target in ('muscle', 'muscle_artifacts', 'muscle-artifacts'):
        return detect_muscle_artifacts_tuh_adapted(X, fs)
    elif target in ('nonphys', 'non-physiological', 'non_physiological'):
        return detect_non_physiological_tuh_adapted(X, fs)
    else:
        raise ValueError(f"Unknown target: '{target}'. Must be 'eye', 'muscle', or 'nonphys'.")
