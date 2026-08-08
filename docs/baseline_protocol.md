# Baseline protocol (legacy-v1)

This document freezes the pre-refactor comparison protocol. It does not claim
that a model is better until runs using the same system, seed, data split,
training budget and evaluation rollout have been completed.

- Models: `mlp`, `rnn`, `node`, `anode`, paper-aligned `csode`, and the
  proposed stable `fnode`.
- Data: trajectories generated from the equations in `src/dynsys/data/systems.py`,
  component-wise z-score normalization, chronological 80/20 split.
- Training: `hidden=64`, `batch=256`, `lr=1e-3`, `epochs=1000`, unless an
  experiment configuration explicitly overrides a field for every model.
- Evaluation: one-step autoregressive rollout from the first generated state.
  Record RMSE, MAE, `T_valid` (1 standard deviation threshold), PSD distance,
  Wasserstein-1, and symmetric histogram KL.

The original repository contains no committed checkpoints or result manifests.
Consequently this file records the reproducible *protocol*, rather than
inventing numerical baseline results. Future baseline runs must save their
resolved configuration, checkpoint, training history and `metrics.json`.
