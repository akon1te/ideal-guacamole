"""Reference trajectory generators used by every experiment.

The equations and numerical tolerances intentionally match the legacy
``systems.py`` module, so moving to this package does not change a baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def rossler(t, y, a=0.2, b=0.2, c=5.7):
    x, y_value, z = y
    return [-y_value - z, x + a * y_value, b + z * (x - c)]


def hindmarsh_rose(
    t, y, I=3.25, r=0.006, s=4.0, alpha=-1.6, a=1.0, b=3.0, c=1.0, d=5.0
):
    x, y_value, z = y
    dx = y_value - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - y_value
    dz = r * (s * (x - alpha) - z)
    return [dx, dy, dz]


def quasi_periodic(t, y, lam=0.5, beta=1 / 18, kappa=0.02, omega0=5.1):
    x, y_value, z = y
    dx = y_value
    dy = (lam + z + x**2 - beta * x**4) * y_value - omega0**2 * x
    dz = -z - kappa * y_value**2
    return [dx, dy, dz]


SYSTEMS = {
    "rossler": (rossler, [1.0, 0.0, 0.0], (0, 200), 0.05),
    "hindmarsh_rose": (hindmarsh_rose, [0.0, 0.0, 0.0], (0, 2000), 0.1),
    "quasi_periodic": (quasi_periodic, [0.1, 0.0, 0.0], (0, 200), 0.02),
}


def generate(system: str, discard: float = 0.2):
    """Simulate a system and discard its initial transient."""
    fn, y0, tspan, dt = SYSTEMS[system]
    t_eval = np.arange(tspan[0], tspan[1], dt)
    sol = solve_ivp(fn, tspan, y0, t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9)
    t, y = sol.t, sol.y.T
    cut = int(len(t) * discard)
    return t[cut:] - t[cut], y[cut:]
