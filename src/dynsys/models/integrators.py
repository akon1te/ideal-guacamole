"""Numerical integrators used by the learned continuous-time models."""
from __future__ import annotations

import torch


def euler_integrate(func, y0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Integrate a vector field with fixed-step forward Euler."""
    if t_grid.ndim != 1 or t_grid.numel() < 2:
        raise ValueError("t_grid must be one-dimensional with at least two points.")
    ys, y = [y0], y0
    for index in range(t_grid.numel() - 1):
        step = t_grid[index + 1] - t_grid[index]
        y = y + step * func(t_grid[index], y)
        ys.append(y)
    return torch.stack(ys, dim=0)


def rk4_integrate(func, y0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Integrate ``dy/dt = func(t, y)`` with fixed-step classical RK4.

    Args:
        func: Callable accepting ``(scalar_time, batch_state)``.
        y0: Initial state of shape ``(batch, state_dim)``.
        t_grid: Monotonic one-dimensional grid, including the initial time.
    Returns:
        States of shape ``(len(t_grid), batch, state_dim)``.
    """
    if t_grid.ndim != 1 or t_grid.numel() < 2:
        raise ValueError("t_grid must be one-dimensional with at least two points.")
    ys, y = [y0], y0
    for index in range(t_grid.numel() - 1):
        t0 = t_grid[index]
        step = t_grid[index + 1] - t0
        k1 = func(t0, y)
        k2 = func(t0 + 0.5 * step, y + 0.5 * step * k1)
        k3 = func(t0 + 0.5 * step, y + 0.5 * step * k2)
        k4 = func(t0 + step, y + step * k3)
        y = y + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return torch.stack(ys, dim=0)
