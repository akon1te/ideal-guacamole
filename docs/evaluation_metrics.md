# Dynamical evaluation protocol

All validation metrics are calculated on the validation portion of an
autoregressive rollout. If a deliberately short diagnostic rollout ends before
the train/validation boundary, `validate.py` reports this and uses its available
region instead; such a run must not be compared with a full validation result.

## Pointwise and horizon metrics

- MAE and RMSE in original state units.
- `T_valid`: first time normalized RMSE exceeds one standard deviation.

## Distributional and temporal metrics

- Spectral L1: mean log-power-spectrum mismatch.
- Wasserstein-1 and symmetric histogram KL for marginal state distributions.
- ACF-L1: mean absolute difference of normalized autocorrelation functions.
- Dominant-frequency error: absolute difference of the strongest non-DC PSD
  frequency for each state component.

## Geometric and stability metrics

- Phase-space Chamfer: symmetric nearest-neighbour distance between normalized
  predicted and true attractor point clouds. Lower is better.
- Outside-truth-envelope rate: fraction of predictions outside a padded range
  of validation truth values; mean excess gives its magnitude.
- Finite-time divergence rate: slope of local neighbour-separation growth. It
  is a reproducible proxy for local chaotic divergence, **not** a formal
  largest Lyapunov exponent.

For Hindmarsh-Rose, burst duration and inter-spike interval remain additional
domain-specific metrics.
