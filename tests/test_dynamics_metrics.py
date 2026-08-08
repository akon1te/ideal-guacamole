import numpy as np

from src.dynsys.evaluation.dynamics_metrics import (
    autocorrelation_distance,
    boundedness_metrics,
    dominant_frequency,
    finite_time_divergence_rate,
    phase_space_chamfer,
)


def _circle(n=500, dt=0.01):
    t = np.arange(n) * dt
    return np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])


def test_identical_trajectories_match_temporal_and_geometric_metrics():
    truth = _circle()
    acf, _ = autocorrelation_distance(truth, truth, max_lags=30)
    assert acf == 0.0
    assert phase_space_chamfer(truth, truth) == 0.0
    assert boundedness_metrics(truth, truth) == (0.0, 0.0)


def test_dominant_frequency_recovers_known_oscillation():
    frequency = dominant_frequency(_circle(), dt=0.01)
    assert np.allclose(frequency, [1.0, 1.0], atol=0.01)


def test_boundedness_and_divergence_metrics_are_computable():
    truth = _circle()
    prediction = 3.0 * truth
    rate, pairs = finite_time_divergence_rate(truth, dt=0.01, fit_steps=10)
    outside_rate, excess = boundedness_metrics(prediction, truth, relative_pad=0.0)
    assert pairs > 0
    assert np.isfinite(rate)
    assert outside_rate > 0
    assert excess > 0
