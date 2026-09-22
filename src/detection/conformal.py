"""Finite-sample conformal utilities shared by graph-level detectors."""

from __future__ import annotations

import numpy as np


def upper_tail_pvalues(calibration_scores, test_scores):
    calibration = np.asarray(calibration_scores, dtype=float)
    scores = np.asarray(test_scores, dtype=float)
    if calibration.size == 0:
        raise ValueError("Conformal calibration scores cannot be empty")
    return np.asarray([
        (float(np.count_nonzero(calibration >= score)) + 1.0) / (len(calibration) + 1.0)
        for score in scores
    ])


def conformal_predictions(calibration_scores, test_scores, alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("Conformal alpha must be in (0, 1)")
    return (upper_tail_pvalues(calibration_scores, test_scores) <= alpha).astype(np.int64)


def zero_exceedance_predictions(calibration_scores, test_scores):
    """Alert only when a score strictly exceeds every calibration score."""
    calibration = np.asarray(calibration_scores, dtype=float)
    scores = np.asarray(test_scores, dtype=float)
    if calibration.size == 0:
        raise ValueError("Calibration scores cannot be empty")
    return (scores > np.max(calibration)).astype(np.int64)
