"""ControlSynth Neural ODE baseline aligned with the authors' code.

The official main-experiment implementation evolves a synthesized control
state from ``u0 = x0`` and uses its instantaneous update in both equations:

    dx/dt = f_theta(t, x) + g_phi(t, u)
    du/dt = g_phi(t, u)

``f_theta`` is the main time-conditioned MLP and ``g_phi`` is the auxiliary
control network.  This is the executable experimental form of the paper's
``+ g(u(t))`` control term.  The paper's LMI certificate is not solved here,
so this module is an architecture-faithful baseline, not LMI-certified.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .integrators import euler_integrate, rk4_integrate


class TimeMLP(nn.Module):
    """MLP receiving a scalar solver time and a batched state."""

    def __init__(self, dim: int, hidden: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive.")
        # The authors' default ODEFuncG is a single linear time-conditioned
        # control layer.  Deeper values select their MLP control variant.
        if layers == 1:
            self.net = nn.Linear(dim + 1, dim)
            return
        modules: list[nn.Module] = []
        in_dim = dim + 1
        for _ in range(layers):
            modules.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        modules.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*modules)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time = torch.ones(y.shape[0], 1, device=y.device, dtype=y.dtype) * t
        return self.net(torch.cat([time, y], dim=-1))


class CSODEFunc(nn.Module):
    """Joint ControlSynth vector field over ``z = [x, u]``."""

    def __init__(self, dim: int, hidden: int, layers: int = 3,
                 control_hidden: int | None = None,
                 control_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.main = TimeMLP(dim, hidden, layers)
        self.control = TimeMLP(
            dim,
            hidden if control_hidden is None else control_hidden,
            control_layers,
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x, u = z[:, :self.dim], z[:, self.dim:]
        control_update = self.control(t, u)
        dx = self.main(t, x) + control_update
        return torch.cat([dx, control_update], dim=-1)


class CSODE(nn.Module):
    """Autonomous ControlSynthODE using the project-wide model interface."""

    def __init__(self, dim: int, hidden: int, layers: int = 3,
                 control_hidden: int | None = None,
                 control_layers: int = 1, solver: str = "rk4"):
        super().__init__()
        if solver not in {"rk4", "euler"}:
            raise ValueError("CSODE solver must be 'rk4' or 'euler'.")
        self.dim = dim
        self.solver = solver
        self.func = CSODEFunc(dim, hidden, layers, control_hidden, control_layers)

    def forward(self, y0: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
        # The authors initialise the synthesized control trajectory at x0.
        z0 = torch.cat([y0, y0], dim=-1)
        integrate = rk4_integrate if self.solver == "rk4" else euler_integrate
        trajectory = integrate(self.func, z0, t_grid)
        return trajectory[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, x: torch.Tensor):
        z = torch.cat([x, x], dim=-1).detach().requires_grad_(True)
        t = torch.zeros((), device=x.device, dtype=x.dtype)
        f = self.func(t, z)
        vector = torch.randn_like(z)
        (jacobian_vector,) = torch.autograd.grad(
            f, z, grad_outputs=vector, create_graph=True, retain_graph=True,
        )
        return {"jac": jacobian_vector.square().mean(), "kin": f.square().mean()}
