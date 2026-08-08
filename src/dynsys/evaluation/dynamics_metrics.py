"""Metrics that compare temporal and geometric dynamical properties."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def autocorrelation_1d(x, max_lags):
    """Normalized autocorrelation for non-negative lags of a scalar series."""
    x = np.asarray(x, dtype=np.float64)
    lags = min(max_lags, len(x) - 1)
    if lags < 1:
        return np.array([1.0])
    x = x - x.mean()
    variance = np.mean(x ** 2)
    if variance < 1e-12:
        return np.r_[1.0, np.zeros(lags)]
    return np.array([
        1.0 if lag == 0 else np.mean(x[:-lag] * x[lag:]) / variance
        for lag in range(lags + 1)
    ])


def autocorrelation_distance(a, b, max_lags=100):
    """Mean L1 difference of per-component normalized autocorrelation."""
    per_component = []
    for d in range(a.shape[1]):
        acf_a = autocorrelation_1d(a[:, d], max_lags)
        acf_b = autocorrelation_1d(b[:, d], max_lags)
        n = min(len(acf_a), len(acf_b))
        per_component.append(float(np.mean(np.abs(acf_a[:n] - acf_b[:n]))))
    return float(np.mean(per_component)), per_component


def dominant_frequency(x, dt):
    """Dominant non-DC frequency for each component of a trajectory."""
    if len(x) < 4:
        return [float("nan")] * x.shape[1]
    freqs = np.fft.rfftfreq(len(x), d=dt)
    spectrum = np.abs(np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0)) ** 2
    spectrum[0] = 0.0
    return freqs[np.argmax(spectrum, axis=0)].tolist()


def phase_space_chamfer(a, b, max_samples=2000):
    """Symmetric nearest-neighbour distance between point clouds in phase space."""
    def sample(x):
        if len(x) <= max_samples:
            return x
        return x[np.linspace(0, len(x) - 1, max_samples, dtype=int)]

    a, b = sample(a), sample(b)
    a_to_b = cKDTree(b).query(a, k=1)[0].mean()
    b_to_a = cKDTree(a).query(b, k=1)[0].mean()
    return float(0.5 * (a_to_b + b_to_a))


def boundedness_metrics(pred, truth, relative_pad=0.05):
    """Fraction and mean magnitude of predictions outside the truth envelope."""
    lo, hi = truth.min(axis=0), truth.max(axis=0)
    pad = np.maximum((hi - lo) * relative_pad, 1e-8)
    below = np.maximum(lo - pad - pred, 0.0)
    above = np.maximum(pred - hi - pad, 0.0)
    excess = below + above
    return float(np.mean(excess > 0)), float(np.mean(excess))


def finite_time_divergence_rate(x, dt, max_points=2000, theiler=10, fit_steps=20):
    """Neighbour-separation growth proxy; not a formal Lyapunov exponent."""
    stride = max(1, int(np.ceil(len(x) / max_points)))
    x = np.asarray(x[::stride], dtype=np.float64)
    if len(x) <= theiler + fit_steps + 2:
        return float("nan"), 0
    tree = cKDTree(x)
    pairs = []
    for i, point in enumerate(x):
        distances, indices = tree.query(point, k=min(16, len(x)))
        for distance, j in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            if abs(i - j) > theiler and distance > 1e-12:
                if i + fit_steps < len(x) and j + fit_steps < len(x):
                    pairs.append((i, int(j), float(distance)))
                break
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    growth = []
    for lag in range(1, fit_steps + 1):
        ratios = [
            np.linalg.norm(x[i + lag] - x[j + lag]) / distance
            for i, j, distance in pairs
        ]
        growth.append(np.mean(np.log(np.maximum(ratios, 1e-12))))
    rate = np.polyfit(np.arange(1, fit_steps + 1) * dt * stride, growth, deg=1)[0]
    return float(rate), len(pairs)
