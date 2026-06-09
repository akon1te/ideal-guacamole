from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# === RK4 integrator (copy from models.py for self-containment) ============
def rk4_integrate(func, y0, t_grid):
    """Fixed-step RK4 over an arbitrary 1-D time grid.

    func(t, y) : (B, D) -> (B, D)
    y0         : (B, D)
    t_grid     : (T+1,)
    returns traj: (T+1, B, D)
    """
    ys = [y0]
    y = y0
    for i in range(t_grid.numel() - 1):
        t0 = t_grid[i]
        h = t_grid[i + 1] - t0
        k1 = func(t0,           y)
        k2 = func(t0 + 0.5 * h, y + 0.5 * h * k1)
        k3 = func(t0 + 0.5 * h, y + 0.5 * h * k2)
        k4 = func(t0 + h,       y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return torch.stack(ys, dim=0)


# === Building-block MLPs ===================================================
def _tanh_mlp(in_dim, hidden, out_dim, n_hidden_layers=1):
    """Plain MLP with Tanh activations, matching the baseline CSODE backbone.

    n_hidden_layers=1 -> Linear(in,h)->Tanh->Linear(h,h)->Tanh->Linear(h,out)
    """
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


# =========================================================================
# Idea 2 — CSODE-diag : per-dimension diagonal contraction γ ∈ R^d
# =========================================================================
class CSODEDiagFunc(nn.Module):
    """f(y) = -diag(softplus(γ)) · y + MLP(y), with γ ∈ R^{total_dim}.

    The only change from `CSODEFunc` is that γ is a vector rather than a
    scalar, so each coordinate gets its own learnable decay rate. The
    contraction guarantee (linear part is negative-definite) still holds
    because softplus(·) > 0.
    """
    def __init__(self, total_dim, hidden, gamma_init=0.1):
        super().__init__()
        self.net = _tanh_mlp(total_dim, hidden, total_dim, n_hidden_layers=1)
        # γ ∈ R^{total_dim}, initialised to the same scalar as baseline CSODE
        self.gamma = nn.Parameter(
            torch.full((total_dim,), float(gamma_init))
        )

    def forward(self, t, y):
        # softplus(γ) ∈ R^D, broadcast against y ∈ (B, D)
        return -F.softplus(self.gamma) * y + self.net(y)


# =========================================================================
# Idea 1 — CSODE-full : state-dependent contraction matrix
# =========================================================================
class CSODEFullFunc(nn.Module):
    """f(y) = A(y)·y + g(y),   A(y) = -exp(α)·I - W(y)ᵀ·W(y).

    `W(y)` is a small MLP producing an (r × D) matrix (low-rank PSD factor).
    `A(y)` is therefore always negative-definite, guaranteeing pointwise
    contractivity of the linear-in-y part.

    Args:
      total_dim  : full state dim (dim + aug)
      hidden     : width of the free MLP g(y)
      hidden_W   : width of the W(y) sub-MLP
      rank       : rank r of W; r=total_dim recovers full rank, r<total_dim
                   gives a parameter-efficient low-rank approximation
      alpha_init : initial value of the scalar α (controls baseline decay
                   exp(α); α=log(softplus(0.1))≈-2.25 ≈ baseline CSODE)
    """
    def __init__(self, total_dim, hidden, hidden_W=16, rank=None,
                 alpha_init=-2.25):
        super().__init__()
        self.total_dim = total_dim
        self.rank = rank if rank is not None else total_dim

        # free MLP g(y)
        self.g = _tanh_mlp(total_dim, hidden, total_dim, n_hidden_layers=1)

        # W(y) : R^D -> R^{rank × D} flattened. Build the final Linear
        # explicitly so we can type-safely initialise its tensors.
        self.W_hidden = nn.Sequential(
            nn.Linear(total_dim, hidden_W), nn.Tanh(),
        )
        self.W_out = nn.Linear(hidden_W, self.rank * total_dim)
        # Initialise the final layer small so that A(y) ≈ -exp(α)·I at start.
        nn.init.normal_(self.W_out.weight, std=1e-3)
        nn.init.zeros_(self.W_out.bias)

        # scalar α — controls the isotropic part exp(α)·I
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, t, y):
        B, D = y.shape
        # W(y) shaped (B, rank, D)
        W = self.W_out(self.W_hidden(y)).view(B, self.rank, D)
        # A(y)·y = -exp(α)·y - W^T (W y)
        Wy = torch.einsum("brd,bd->br", W, y)              # (B, rank)
        WtWy = torch.einsum("brd,br->bd", W, Wy)           # (B, D)
        decay = torch.exp(self.alpha)
        return -decay * y - WtWy + self.g(y)


# =========================================================================
# Idea 7 — CSODE-Res : pre-activation residual blocks + outer contraction
# =========================================================================
class _ResBlock(nn.Module):
    """Pre-activation residual MLP block: y' = y + W2·tanh(W1·tanh(y))."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.act = nn.Tanh()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        # zero-init last layer => initial flow ≈ identity (clean starting point)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, y):
        h = self.act(self.fc1(self.act(y)))
        return y + self.fc2(h)


class CSODEResFunc(nn.Module):
    """f(y) = -softplus(γ)·y + ResStack(y).

    Combines CSODE-style scalar contraction with a stack of N residual
    blocks (RNODE-like). γ may be scalar (default) or per-dim diagonal
    by setting `gamma_per_dim=True`.

    Args:
      total_dim     : full state dim (dim + aug)
      hidden        : width of each residual block
      n_blocks      : number of residual blocks stacked
      gamma_init    : initial γ (passed to softplus)
      gamma_per_dim : if True, γ ∈ R^D (Idea 2 + Idea 7 combo)
    """
    def __init__(self, total_dim, hidden, n_blocks=2, gamma_init=0.1,
                 gamma_per_dim=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ResBlock(total_dim, hidden) for _ in range(n_blocks)]
        )
        # input encoder so the residual stack operates in a hidden space
        # is overkill for d≈3..7; keep it minimal and route y through blocks
        if gamma_per_dim:
            self.gamma = nn.Parameter(
                torch.full((total_dim,), float(gamma_init))
            )
        else:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_per_dim = gamma_per_dim

    def forward(self, t, y):
        z = y
        for blk in self.blocks:
            z = blk(z)
        # `z - y` is the net residual contribution (the original `y` part is
        # already absorbed into the contraction term below)
        residual = z - y
        return -F.softplus(self.gamma) * y + residual


# =========================================================================
# Idea 10 — CSODE-statedep : state-dependent γ(y)
# =========================================================================
class CSODEStateDepFunc(nn.Module):
    """f(y) = -diag(γ(y))·y + MLP(y),  γ(y) = softplus(MLP_small(y)).

    The decay rate is adapted to the local state, allowing the model to
    learn where to dissipate stronger / weaker (e.g. stronger near
    Rössler's z-spikes, weaker on the flat part of the attractor).

    A scalar bias `gamma_bias_init` is added inside softplus so that at
    initialisation γ(y) ≈ softplus(gamma_bias_init) matches the baseline
    CSODE scalar γ.
    """
    def __init__(self, total_dim, hidden, hidden_gamma=16,
                 gamma_bias_init=0.1):
        super().__init__()
        self.main = _tanh_mlp(total_dim, hidden, total_dim, n_hidden_layers=1)
        # Split the gamma sub-MLP so we can type-safely zero-init the head.
        self.gamma_hidden = nn.Sequential(
            nn.Linear(total_dim, hidden_gamma), nn.Tanh(),
        )
        self.gamma_head = nn.Linear(hidden_gamma, total_dim)
        # zero-init the gamma head so γ(y) starts at softplus(gamma_bias_init)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        self.gamma_bias = nn.Parameter(
            torch.full((total_dim,), float(gamma_bias_init))
        )

    def forward(self, t, y):
        # raw + bias, then softplus -> guaranteed positive
        gamma_raw = self.gamma_head(self.gamma_hidden(y))
        gamma = F.softplus(gamma_raw + self.gamma_bias)           # (B, D)
        return -gamma * y + self.main(y)


# =========================================================================
# Generic augmented wrapper (mirrors `_AugmentedODE` in models.py)
# =========================================================================
class _AugmentedCSODE(nn.Module):
    """Generic augmented-state CSODE wrapper.

    Subclasses just need to set `self.func` (an `nn.Module` whose forward
    has signature `(t, y_aug)`) and `self.total_dim` in __init__.

    forward(y0, t_grid) -> (B, T-1, dim)
    """
    def __init__(self, dim, aug):
        super().__init__()
        self.dim = dim
        self.aug = aug
        self.total_dim = dim + aug

    def forward(self, y0, t_grid):
        # y0: (B, dim) -> augment with zeros
        B = y0.shape[0]
        if self.aug > 0:
            zeros = torch.zeros(B, self.aug, device=y0.device, dtype=y0.dtype)
            z0 = torch.cat([y0, zeros], dim=-1)
        else:
            z0 = y0
        traj = rk4_integrate(self.func, z0, t_grid)        # (T+1, B, total_dim)
        traj = traj.permute(1, 0, 2)                       # (B, T+1, total_dim)
        return traj[:, 1:, :self.dim]                      # (B, T, dim)

    # Optional kinetic regularisation hook (used by train_csode.py if
    # --lambda_kin > 0).  Mirrors the contract used by HNODE/WP-NODE.
    def regularization(self, x):
        """Compute Hutchinson Jacobian + kinetic penalties at sample y=aug(x)."""
        B = x.shape[0]
        if self.aug > 0:
            zeros = torch.zeros(B, self.aug, device=x.device, dtype=x.dtype)
            y = torch.cat([x, zeros], dim=-1)
        else:
            y = x
        y = y.detach().requires_grad_(True)
        t = torch.tensor(0.0, device=x.device)
        f = self.func(t, y)
        kin = (f ** 2).mean()
        # Hutchinson estimator: || J^T v ||^2 ≈ tr(J^T J)
        v = torch.randn_like(y)
        Jv = torch.autograd.grad(f, y, grad_outputs=v,
                                 create_graph=True, retain_graph=True)[0]
        jac = (Jv ** 2).mean()
        return {"jac": jac, "kin": kin}


class CSODEDiag(_AugmentedCSODE):
    """Idea 2: CSODE with diagonal per-dim γ."""
    def __init__(self, dim, hidden, aug=4, gamma_init=0.1):
        super().__init__(dim, aug)
        self.func = CSODEDiagFunc(self.total_dim, hidden, gamma_init=gamma_init)


class CSODEFull(_AugmentedCSODE):
    """Idea 1: CSODE-full with state-dependent contraction matrix."""
    def __init__(self, dim, hidden, aug=4, hidden_W=16, rank=None,
                 alpha_init=-2.25):
        super().__init__(dim, aug)
        self.func = CSODEFullFunc(
            self.total_dim, hidden,
            hidden_W=hidden_W, rank=rank, alpha_init=alpha_init,
        )


class CSODERes(_AugmentedCSODE):
    """Idea 7: CSODE-Res — residual stack + outer contraction."""
    def __init__(self, dim, hidden, aug=4, n_blocks=2, gamma_init=0.1,
                 gamma_per_dim=False):
        super().__init__(dim, aug)
        self.func = CSODEResFunc(
            self.total_dim, hidden,
            n_blocks=n_blocks, gamma_init=gamma_init,
            gamma_per_dim=gamma_per_dim,
        )


class CSODEStateDep(_AugmentedCSODE):
    """Idea 10: CSODE with state-dependent γ(y)."""
    def __init__(self, dim, hidden, aug=4, hidden_gamma=16,
                 gamma_bias_init=0.1):
        super().__init__(dim, aug)
        self.func = CSODEStateDepFunc(
            self.total_dim, hidden,
            hidden_gamma=hidden_gamma, gamma_bias_init=gamma_bias_init,
        )


# === Factory ==============================================================
# Names registered by `train_csode.py`'s --model choice.
CSODE_VARIANTS = {
    "csode_diag",       # Idea 2
    "csode_full",       # Idea 1
    "csode_res",        # Idea 7
    "csode_statedep",   # Idea 10
}


def build_csode_variant(model_type, dim, hidden, aug=4,
                        # csode_full
                        hidden_W=16, rank=None, alpha_init=-2.25,
                        # csode_res
                        n_blocks=2, gamma_per_dim=False,
                        # csode_statedep
                        hidden_gamma=16,
                        # shared
                        gamma_init=0.1):
    """Build a CSODE variant by name. Raises if `model_type` is unknown."""
    if model_type == "csode_diag":
        return CSODEDiag(dim, hidden, aug=aug, gamma_init=gamma_init)
    if model_type == "csode_full":
        return CSODEFull(dim, hidden, aug=aug,
                         hidden_W=hidden_W, rank=rank, alpha_init=alpha_init)
    if model_type == "csode_res":
        return CSODERes(dim, hidden, aug=aug,
                        n_blocks=n_blocks, gamma_init=gamma_init,
                        gamma_per_dim=gamma_per_dim)
    if model_type == "csode_statedep":
        return CSODEStateDep(dim, hidden, aug=aug,
                             hidden_gamma=hidden_gamma,
                             gamma_bias_init=gamma_init)
    raise ValueError(f"Unknown CSODE variant: {model_type}. "
                     f"Available: {sorted(CSODE_VARIANTS)}")
