"""Single source of truth for the five public experiment models.

``fnode`` is the former ``csode_full`` model: an augmented ODE with a
state-dependent negative-definite contraction matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    constructor: Callable
    is_ode: bool
    supports_regularization: bool = False


def _specs() -> dict[str, ModelSpec]:
    # Local imports keep the legacy classes usable while the project migrates
    # to ``src/dynsys/models`` module-by-module.
    from .csode import CSODE
    from .fnode import FNODE
    from models import ANODE, MLP, NeuralODE, RNN

    return {
        "node": ModelSpec("node", lambda d, h, l, a, **_: NeuralODE(d, h), True),
        "anode": ModelSpec("anode", lambda d, h, l, a, **_: ANODE(d, h, aug=a), True),
        "csode": ModelSpec("csode", lambda d, h, l, a, csode_layers=3, csode_control_hidden=None, csode_control_layers=1, csode_solver="rk4", **_: CSODE(d, h, layers=csode_layers, control_hidden=csode_control_hidden, control_layers=csode_control_layers, solver=csode_solver), True, True),
        "mlp": ModelSpec("mlp", lambda d, h, l, a, **_: MLP(d, h, l), False),
        "rnn": ModelSpec("rnn", lambda d, h, l, a, **_: RNN(d, h, l), False),
        "fnode": ModelSpec("fnode", lambda d, h, l, a, fnode_hidden_w=16, fnode_rank=None, fnode_alpha_init=-2.25, **_: FNODE(d, h, aug=a or 4, hidden_w=fnode_hidden_w, rank=fnode_rank, alpha_init=fnode_alpha_init), True, True),
    }


MODEL_SPECS = _specs()


def available_model_names() -> tuple[str, ...]:
    return tuple(MODEL_SPECS)


def ode_model_names() -> frozenset[str]:
    return frozenset(name for name, spec in MODEL_SPECS.items() if spec.is_ode)


def regularized_model_names() -> frozenset[str]:
    return frozenset(name for name, spec in MODEL_SPECS.items() if spec.supports_regularization)


def build_model(model_type: str, dim: int, hidden: int, seq_len: int, aug: int = 2, **kwargs):
    """Build a registered model or raise a helpful error for unsupported names."""
    try:
        spec = MODEL_SPECS[model_type]
    except KeyError as error:
        options = ", ".join(available_model_names())
        raise ValueError(f"Unknown model_type: {model_type}. Available: {options}") from error
    return spec.constructor(dim, hidden, seq_len, aug, **kwargs)
