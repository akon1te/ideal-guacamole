"""Stable augmented FNODE used in the public five-model benchmark.

This is the former experimental ``csode_full`` variant, renamed to FNODE for
the main research comparison.  Its state-dependent negative-definite term
keeps the learned vector field dissipative during long rollouts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .integrators import rk4_integrate


def _tanh_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class FNODEFunc(nn.Module):
    """f(z) = -exp(alpha) z - W(z)^T W(z) z + g(z)."""

    def __init__(self, total_dim: int, hidden: int, hidden_w: int = 16,
                 rank: int | None = None, alpha_init: float = -2.25):
        super().__init__()
        self.rank = total_dim if rank is None else rank
        self.g = _tanh_mlp(total_dim, hidden, total_dim)
        self.w_hidden = nn.Sequential(nn.Linear(total_dim, hidden_w), nn.Tanh())
        self.w_out = nn.Linear(hidden_w, self.rank * total_dim)
        nn.init.normal_(self.w_out.weight, std=1e-3)
        nn.init.zeros_(self.w_out.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, t, z):
        batch, dim = z.shape
        w = self.w_out(self.w_hidden(z)).view(batch, self.rank, dim)
        wz = torch.einsum("brd,bd->br", w, z)
        wt_wz = torch.einsum("brd,br->bd", w, wz)
        return -torch.exp(self.alpha) * z - wt_wz + self.g(z)


class FNODE(nn.Module):
    """Augmented stable NeuralODE returning only the observed state."""

    def __init__(self, dim: int, hidden: int, aug: int = 4, hidden_w: int = 16,
                 rank: int | None = None, alpha_init: float = -2.25):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = FNODEFunc(dim + aug, hidden, hidden_w, rank, alpha_init)

    def forward(self, y0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        batch = y0.shape[0]
        zeros = y0.new_zeros(batch, self.aug)
        z0 = torch.cat([y0, zeros], dim=-1)
        trajectory = rk4_integrate(self.func, z0, t_grid)
        return trajectory[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, x: torch.Tensor):
        """Return Jacobian and kinetic penalties compatible with ``train.py``."""
        batch = x.shape[0]
        z = torch.cat([x, x.new_zeros(batch, self.aug)], dim=-1)
        z = z.detach().requires_grad_(True)
        f = self.func(torch.zeros((), device=x.device, dtype=x.dtype), z)
        vector = torch.randn_like(z)
        (jacobian_vector,) = torch.autograd.grad(
            f, z, grad_outputs=vector, create_graph=True, retain_graph=True,
        )
        return {"jac": jacobian_vector.square().mean(), "kin": f.square().mean()}
