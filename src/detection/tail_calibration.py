"""Validation-only peaks-over-threshold calibration for extreme online scores."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from detection.conformal import upper_tail_pvalues


class EmpiricalGPDCalibrator:
    def __init__(self, tail_fraction: float = 0.1, min_tail: int = 20):
        self.tail_fraction = tail_fraction
        self.min_tail = min_tail

    def fit(self, scores):
        self.scores = np.sort(np.asarray(scores, dtype=float))
        if len(self.scores) < self.min_tail:
            raise ValueError("GPD calibration requires at least min_tail validation scores")
        tail_size = max(self.min_tail, int(np.ceil(len(self.scores) * self.tail_fraction)))
        tail_size = min(tail_size, len(self.scores) - 1)
        self.threshold = float(self.scores[-tail_size - 1])
        excess = self.scores[-tail_size:] - self.threshold
        try:
            shape, _, scale = genpareto.fit(excess, floc=0.0)
        except (ValueError, FloatingPointError):
            shape, scale = 0.0, float(np.mean(excess))
        # Anomaly losses are treated as unbounded; negative-shape fits create a
        # finite endpoint and collapse every more-extreme test score to one tie.
        if shape < 0.0:
            shape, scale = 0.0, float(np.mean(excess))
        self.shape = float(shape)
        self.scale = max(float(scale), np.finfo(float).eps)
        self.fitted_tail_fraction = tail_size / len(self.scores)
        return self

    def pvalues(self, values):
        values = np.asarray(values, dtype=float)
        empirical = upper_tail_pvalues(self.scores, values)
        tail = values > self.threshold
        if np.any(tail):
            survival = genpareto.sf(
                values[tail] - self.threshold, self.shape, loc=0.0, scale=self.scale
            )
            empirical[tail] = np.minimum(
                empirical[tail], self.fitted_tail_fraction * np.asarray(survival)
            )
        return np.clip(empirical, np.finfo(float).tiny, 1.0)
