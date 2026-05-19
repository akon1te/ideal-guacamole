"""Models: NeuralODE, ANODE, WP-NODE, CSODE, HNODE, MLP, RNN baselines.

NeuralODE-family models use an inline RK4 solver (no torchdiffeq) and can be
optionally compiled via torch.compile for ~2-3x speed-up on GPU.

Model summary
-------------
NODE     - plain Neural ODE (Chen et al. 2018), tanh MLP
ANODE    - Augmented Neural ODE (Dupont et al. 2019)
WP-NODE  - Well-Posed NODE: NODE + Jacobian (Hutchinson) + kinetic regularization
CSODE    - Continuously-Stable ODE: f(y) = -softplus(gamma)*y + MLP(y)
HNODE    - Hierarchical NODE (proposed): augmented + dual-branch (fast SiLU /
           slow tanh) + soft contraction + spectral loss (applied in train.py)
MLP/RNN  - non-ODE baselines
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# === Inline RK4 integrator =================================================
def rk4_integrate(func, y0, t_grid):
    """Fixed-step RK4 over an arbitrary 1-D time grid.

    func(t, y)  : (B, D) -> (B, D)
    y0          : (B, D)
    t_grid      : (T+1,)
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


# === Vector-field modules ==================================================
class ODEFunc(nn.Module):
    """Plain MLP vector field used by NODE, ANODE, WP-NODE."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        return self.net(y)


class CSODEFunc(nn.Module):
    """Continuously-Stable ODE function: f(y) = -softplus(gamma) * y + MLP(y).

    The contraction term -softplus(gamma) * y guarantees that the linear part
    of the vector field is dissipative (negative-definite) which keeps long-
    horizon rollouts bounded.
    """
    def __init__(self, dim, hidden, gamma_init=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, t, y):
        return -F.softplus(self.gamma) * y + self.net(y)


class HNODEFunc(nn.Module):
    """Hierarchical NODE vector field (proposed model, v3).

    f(z) = MLP_3layer_tanh(z)

    Design rationale:
      * 3-layer MLP with tanh activations (one extra hidden layer compared to
        the baseline 2-layer NODE/ANODE) gives more capacity to fit fast
        components of the vector field (e.g. the high-gradient z-spikes in
        Rossler, the fast x-y subsystem in Hindmarsh-Rose).
      * Augmented state handled by the HNODE wrapper (dim_in = dim + aug).
      * No spectral normalization. v2 used `spectral_norm` on every layer to
        enforce a global Lipschitz bound, but on Rossler this clamped the
        achievable Lipschitz constant of f below the physical one (Rossler's
        z-spikes have local Lipschitz ~5-10), causing the optimizer to plateau
        with train/val loss orders of magnitude above the unconstrained model.
      * If a soft Lipschitz penalty is desired, use `--lambda_jac` with a
        small weight (e.g. 1e-5) -- the Hutchinson Jacobian regularizer in
        train.py works for HNODE just like for WP-NODE.

    Args:
      dim     : input/output dim (= original dim + aug for HNODE)
      hidden  : hidden width
      gamma_init, alpha_init: kept for API compatibility (unused in v3).
    """
    def __init__(self, dim, hidden, gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,    hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        return self.net(y)


# === Group A: architectural variants of HNODE (same backbone shape) =======
class RNODEFunc(nn.Module):
    """Residual vector field with INTERNAL pre-activation residual blocks.

    f(z) = W_out * h_3
      where each h_k = h_{k-1} + Tanh(Linear(h_{k-1})), h_0 = Tanh(W_in z).

    The internal residual connections (a la "pre-activation ResNet") help
    gradient flow through the depth of the MLP without changing the overall
    asymptotic behaviour of the ODE.  Unlike a residual at the OUTPUT
    (f(z) = z + MLP(z)) this does NOT bias the ODE toward exponential
    growth dy/dt = y -- the output residual idea is wrong for NODE because
    `dy/dt = y` is the divergent identity flow, not a stationary solution.
    """
    def __init__(self, dim, hidden, gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.lin_in = nn.Linear(dim, hidden)
        self.lin_h1 = nn.Linear(hidden, hidden)
        self.lin_h2 = nn.Linear(hidden, hidden)
        self.lin_out = nn.Linear(hidden, dim)
        # zero-init last layer -> ODE starts as f(z) = 0 (stationary flow)
        nn.init.zeros_(self.lin_out.weight)
        nn.init.zeros_(self.lin_out.bias)

    def forward(self, t, y):
        h = torch.tanh(self.lin_in(y))
        h = h + torch.tanh(self.lin_h1(h))
        h = h + torch.tanh(self.lin_h2(h))
        return self.lin_out(h)


class SNODEFunc(nn.Module):
    """SiLU-activation vector field: 3-layer MLP with SiLU (Swish) instead
    of Tanh.  SiLU does not saturate -> non-vanishing gradients for fast
    components (e.g. Rossler z-spikes, Hindmarsh-Rose bursts).
    """
    def __init__(self, dim, hidden, gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,    hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        return self.net(y)


class LNODEFunc(nn.Module):
    """LayerNorm-stabilised vector field: 3-layer Tanh MLP with LayerNorm
    after each hidden activation.  LayerNorm in vector fields tends to
    stabilise long autoregressive rollouts by re-centering the hidden
    activations at every integration step.
    """
    def __init__(self, dim, hidden, gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,    hidden), nn.Tanh(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, y):
        return self.net(y)


# === Hybrid FNODE ===========================================================
class FNODEFunc(nn.Module):
    """Non-residual FNODE vector field.

    This version deliberately removes RNODE-style hidden residual connections.
    It keeps the useful non-residual parts of the previous synthesis:
      * SNODE-style SiLU activations for smooth fast dynamics;
      * LNODE-style LayerNorm after each hidden projection for stable hidden
        statistics during long rollouts;
      * small final-layer initialization so the initial vector field is gentle
        but gradients still flow through all layers from the first step.
    """
    def __init__(self, dim, hidden, n_layers=3,
                 gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            in_dim = hidden
        self.out = nn.Linear(hidden, dim)
        layers.append(self.out)
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, t, y):
        return self.net(y)


class FNODEV2Func(nn.Module):
    """FNODE v2: non-residual multi-branch stable vector field.

    Improvements over plain FNODE without reintroducing residual blocks:
      * dual fast/slow branches (SiLU and Tanh) capture bursty and smooth modes;
      * LayerNorm in each branch stabilizes hidden distributions;
      * learned soft mixing combines branches per hidden channel;
      * small dissipative linear term helps keep long rollouts bounded;
      * small final initialization keeps the initial vector field gentle while
        preserving gradients through both branches.
    """
    def __init__(self, dim, hidden, gamma_init=-4.0, alpha_init=0.0):
        super().__init__()
        self.fast = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.slow = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Tanh(),
        )
        self.mix_logit = nn.Parameter(torch.full((hidden,), float(alpha_init)))
        self.mid = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.out = nn.Linear(hidden, dim)
        self.gamma_raw = nn.Parameter(torch.full((dim,), float(gamma_init)))
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, t, y):
        mix = torch.sigmoid(self.mix_logit).view(1, -1)
        h = mix * self.fast(y) + (1.0 - mix) * self.slow(y)
        f = self.out(self.mid(h))
        return f - F.softplus(self.gamma_raw).view(1, -1) * y


class FNODE(nn.Module):
    """Non-residual FNODE wrapper with augmented state."""
    def __init__(self, dim, hidden, aug=4, n_blocks=3,
                 gamma_init=0.1, alpha_init=0.0,
                 use_decay=False):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = FNODEFunc(
            dim + aug, hidden,
            n_layers=n_blocks,
            gamma_init=gamma_init, alpha_init=alpha_init,
        )

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, y_sample, n_jac_samples=1):
        B = y_sample.size(0)
        z = torch.cat([y_sample, y_sample.new_zeros(B, self.aug)], dim=1)
        return {
            "jac": hutchinson_jacobian_norm(self.func, z, n_samples=n_jac_samples),
            "kin": kinetic_norm(self.func, z),
        }


class FNODEV2(nn.Module):
    """FNODE v2 wrapper with augmented state."""
    def __init__(self, dim, hidden, aug=4,
                 gamma_init=-4.0, alpha_init=0.0):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = FNODEV2Func(
            dim + aug, hidden,
            gamma_init=gamma_init, alpha_init=alpha_init,
        )

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, y_sample, n_jac_samples=1):
        B = y_sample.size(0)
        z = torch.cat([y_sample, y_sample.new_zeros(B, self.aug)], dim=1)
        return {
            "jac": hutchinson_jacobian_norm(self.func, z, n_samples=n_jac_samples),
            "kin": kinetic_norm(self.func, z),
        }


# === Regularization helpers ================================================
def hutchinson_jacobian_norm(func, y, t=None, n_samples=1):
    """Estimate ||J_f(y)||_F^2 with Hutchinson's stochastic trick.

    For v ~ N(0, I_d):  E_v[ ||J^T v||^2 ] = ||J||_F^2.
    We compute one (or more) random projections per call.

    Returns: scalar tensor (mean over batch and samples).
    """
    if t is None:
        t = torch.zeros((), device=y.device, dtype=y.dtype)
    y = y.detach().requires_grad_(True)
    f = func(t, y)                                  # (B, D)
    total = 0.0
    for _ in range(n_samples):
        v = torch.randn_like(f)
        # vector-Jacobian product: v^T J = grad(v . f, y)
        (vjp,) = torch.autograd.grad(
            outputs=(f * v).sum(), inputs=y, create_graph=True, retain_graph=True,
        )
        total = total + vjp.pow(2).sum(dim=-1).mean()
    return total / n_samples


def kinetic_norm(func, y, t=None):
    """Mean squared velocity ||f(y)||^2."""
    if t is None:
        t = torch.zeros((), device=y.device, dtype=y.dtype)
    f = func(t, y)
    return f.pow(2).sum(dim=-1).mean()


# === Model wrappers ========================================================
class NeuralODE(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.func = ODEFunc(dim, hidden)

    def forward(self, y0, t_grid):
        traj = rk4_integrate(self.func, y0, t_grid)   # (T+1, B, dim)
        return traj[1:].permute(1, 0, 2)              # (B, T, dim)


class ANODE(nn.Module):
    """Augmented Neural ODE (Dupont et al. 2019)."""
    def __init__(self, dim, hidden, aug=2):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = ODEFunc(dim + aug, hidden)

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)


class CSODE(nn.Module):
    """Continuously-Stable ODE."""
    def __init__(self, dim, hidden, gamma_init=0.1):
        super().__init__()
        self.func = CSODEFunc(dim, hidden, gamma_init=gamma_init)

    def forward(self, y0, t_grid):
        traj = rk4_integrate(self.func, y0, t_grid)
        return traj[1:].permute(1, 0, 2)


class HNODE(nn.Module):
    """Hierarchical Neural ODE (proposed, v3).

    Wraps a 3-layer Tanh MLP as the vector field, integrated in an augmented
    state of size dim + aug.  See HNODEFunc for design rationale (spectral
    normalization removed in v3 because it clamped the achievable Lipschitz
    constant below the physical one).
    """
    def __init__(self, dim, hidden, aug=4, gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = HNODEFunc(dim + aug, hidden, gamma_init=gamma_init, alpha_init=alpha_init)

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, y_sample, n_jac_samples=1):
        B = y_sample.size(0)
        z = torch.cat([y_sample, y_sample.new_zeros(B, self.aug)], dim=1)
        return {
            "jac": hutchinson_jacobian_norm(self.func, z, n_samples=n_jac_samples),
            "kin": kinetic_norm(self.func, z),
        }


class _AugmentedODE(nn.Module):
    """Generic augmented-state ODE wrapper used by RNODE/SNODE/LNODE.

    Builds an augmented state z0 = [y0 | 0_aug], integrates the supplied
    `func_class` via RK4, and returns only the first `dim` coords.
    """
    def __init__(self, dim, hidden, aug, func_class,
                 gamma_init=0.1, alpha_init=0.0):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = func_class(dim + aug, hidden,
                               gamma_init=gamma_init, alpha_init=alpha_init)

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)

    def regularization(self, y_sample, n_jac_samples=1):
        B = y_sample.size(0)
        z = torch.cat([y_sample, y_sample.new_zeros(B, self.aug)], dim=1)
        return {
            "jac": hutchinson_jacobian_norm(self.func, z, n_samples=n_jac_samples),
            "kin": kinetic_norm(self.func, z),
        }


class RNODE(_AugmentedODE):
    """Residual NODE (group A)."""
    def __init__(self, dim, hidden, aug=4, gamma_init=0.1, alpha_init=0.0):
        super().__init__(dim, hidden, aug, RNODEFunc,
                         gamma_init=gamma_init, alpha_init=alpha_init)


class SNODE(_AugmentedODE):
    """SiLU NODE (group A)."""
    def __init__(self, dim, hidden, aug=4, gamma_init=0.1, alpha_init=0.0):
        super().__init__(dim, hidden, aug, SNODEFunc,
                         gamma_init=gamma_init, alpha_init=alpha_init)


class LNODE(_AugmentedODE):
    """LayerNorm NODE (group A)."""
    def __init__(self, dim, hidden, aug=4, gamma_init=0.1, alpha_init=0.0):
        super().__init__(dim, hidden, aug, LNODEFunc,
                         gamma_init=gamma_init, alpha_init=alpha_init)


class MLP(nn.Module):
    def __init__(self, dim, hidden, seq_len):
        super().__init__()
        self.seq_len, self.dim = seq_len, dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * seq_len),
        )

    def forward(self, y0, t_grid=None):
        return self.net(y0).view(-1, self.seq_len, self.dim)


class RNN(nn.Module):
    def __init__(self, dim, hidden, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRUCell(dim, hidden)
        self.proj = nn.Linear(hidden, dim)

    def forward(self, y0, t_grid=None):
        h = torch.zeros(y0.size(0), self.gru.hidden_size, device=y0.device)
        x, preds = y0, []
        for _ in range(self.seq_len):
            h = self.gru(x, h)
            x = self.proj(h)
            preds.append(x)
        return torch.stack(preds, dim=1)


class LinearDynamics(nn.Module):
    """Линейная часть f_lin со стабильной параметризацией.

    A_eff = (W - W^T) / 2 * skew_scale  +  W_sym * sym_scale  -  softplus(d) * I

    - Антисимметричная часть отвечает за вращения (колебания) без затухания
    - Симметричная часть (мала) — за слабую асимметрию
    - `-softplus(d)·I` гарантирует строго отрицательный сдвиг диагонали:
      собственные числа имеют Re ≤ -min(softplus(d)).
    Все компоненты обучаемы, поэтому скорость затухания подбирается данными.
    """
    def __init__(self, dim, init_decay=0.05, skew_scale=1.0, sym_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.1)
        # softplus(d_init) ≈ init_decay  =>  d_init = log(exp(init_decay) - 1)
        import math
        d_init = math.log(math.expm1(max(init_decay, 1e-3)))
        self.d = nn.Parameter(torch.full((dim,), float(d_init)))
        self.skew_scale = skew_scale
        self.sym_scale = sym_scale
        self.register_buffer("I", torch.eye(dim))

    def effective_A(self):
        W = self.W
        skew = 0.5 * (W - W.T) * self.skew_scale
        sym = 0.5 * (W + W.T) * self.sym_scale
        decay = F.softplus(self.d)              # (dim,) >= 0
        return skew + sym - torch.diag(decay)

    def forward(self, x):
        A = self.effective_A()
        return x @ A.T


class NonlinearDynamics(nn.Module):
    """Нелинейная часть f_nonlin: моделирует сложную динамику."""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class CorrectorDynamics(nn.Module):
    """Корректор: обучается на остаточной ошибке предсказателя.

    Финальный слой инициализирован нулём, поэтому в начале обучения
    corrector не вносит вклад и линейная часть задаёт устойчивую динамику.
    """
    def __init__(self, dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ControlTerm(nn.Module):
    """Истинно «информированный» control: берёт только aug-часть состояния
    и выдаёт вклад в полную правую часть dx (а не только в aug-каналы).

    Финальный слой инициализирован нулём — на старте контроль не активен.
    """
    def __init__(self, aug_dim, total_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(aug_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, total_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, t, aug_part):
        # aug_part: (batch, aug_dim)
        return self.fc2(self.act(self.fc1(aug_part)))


class AugEncoder(nn.Module):
    """Энкодер начальных условий aug-каналов из исходного состояния y0.

    aug0 = MLP(y0)  — позволяет стартовать каждую траекторию из
    своего «информированного» расширенного состояния, а не из общей точки.
    """
    def __init__(self, original_dim, aug_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, aug_dim),
        )
        # Финальный слой ≈ 0, чтобы стартовать близко к aug0=const на ранних эпохах
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y0):
        return self.net(y0)


class ResilientODEFunc(nn.Module):
    """
    Правая часть ОДУ для Res-NODE.
    total_dim = original_dim + augmented_dim
    """
    def __init__(self, original_dim, augmented_dim,
                 hidden_lin=64, hidden_nonlin=64, hidden_corr=32, hidden_ctrl=32,
                 use_control=True, use_nonlinear=False,
                 init_decay=0.05, skew_scale=1.0, sym_scale=0.1):
        super().__init__()
        self.original_dim = original_dim
        self.augmented_dim = augmented_dim
        self.total_dim = original_dim + augmented_dim
        self.use_control = use_control
        self.use_nonlinear = use_nonlinear

        # 1) Линейная часть со стабильной параметризацией (skew + small sym − decay·I)
        self.linear = LinearDynamics(
            self.total_dim,
            init_decay=init_decay,
            skew_scale=skew_scale,
            sym_scale=sym_scale,
        )

        # 2) (опционально) отдельная нелинейная часть predictor'а
        if use_nonlinear:
            self.nonlinear = NonlinearDynamics(self.total_dim, hidden_nonlin)

        # 3) Corrector — основной нелинейный путь поверх linear (residual)
        self.corrector = CorrectorDynamics(self.total_dim, hidden_dim=hidden_corr)

        # 4) Управляющий член: берёт только aug-часть, отдаёт вклад в полное dx
        if use_control:
            self.control = ControlTerm(augmented_dim, self.total_dim, hidden_dim=hidden_ctrl)

    def forward(self, t, state):
        # state: (batch, total_dim)
        dx = self.linear(state) + self.corrector(state)
        if self.use_nonlinear:
            dx = dx + self.nonlinear(state)
        if self.use_control:
            aug_part = state[..., self.original_dim:]
            dx = dx + self.control(t, aug_part)
        return dx


class ResilientNeuralODE(nn.Module):
    """Res-NODE модель с информированным расширением пространства состояний.

    Опции:
      use_control     : включить «информированный» control от aug-каналов
      use_nonlinear   : включить отдельную MLP-ветку (дублирует corrector)
      encode_aug      : aug_init = MLP(y0), вместо константного nn.Parameter
      init_decay      : начальное затухание диагонали линейной части
                        (по умолчанию 0.05 — мягче, чем 0.5 в первой версии)
    """
    def __init__(self, original_dim, augmented_dim=2,
                 use_control=True, use_nonlinear=False, encode_aug=True,
                 hidden_lin=64, hidden_nonlin=64, hidden_corr=32,
                 hidden_ctrl=32, hidden_enc=32,
                 init_decay=0.05, skew_scale=1.0, sym_scale=0.1,
                 solver='dopri5', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.original_dim = original_dim
        self.augmented_dim = augmented_dim
        self.total_dim = original_dim + augmented_dim
        self.use_control = use_control
        self.encode_aug = encode_aug

        if encode_aug:
            self.aug_encoder = AugEncoder(original_dim, augmented_dim, hidden_dim=hidden_enc)
        else:
            self.aug_init = nn.Parameter(torch.zeros(1, augmented_dim), requires_grad=True)

        self.ode_func = ResilientODEFunc(
            original_dim=original_dim,
            augmented_dim=augmented_dim,
            hidden_lin=hidden_lin,
            hidden_nonlin=hidden_nonlin,
            hidden_corr=hidden_corr,
            hidden_ctrl=hidden_ctrl,
            use_control=use_control,
            use_nonlinear=use_nonlinear,
            init_decay=init_decay,
            skew_scale=skew_scale,
            sym_scale=sym_scale,
        )
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def initial_aug(self, y0):
        if self.encode_aug:
            return self.aug_encoder(y0)
        return self.aug_init.expand(y0.shape[0], -1)

    def forward(self, y0, t_span):
        """
        y0: (batch, original_dim)
        t_span: (T,) временные точки (включая начальную)
        Возвращает: (batch, T-1, original_dim) – предсказания на шагах 1..T-1
        """
        aug0 = self.initial_aug(y0)
        state0 = torch.cat([y0, aug0], dim=-1)

        states = odeint(self.ode_func, state0, t_span,
                        method=self.solver, atol=self.atol, rtol=self.rtol)
        # states: (len(t_span), batch, total_dim)
        states = states.permute(1, 0, 2)          # (batch, len(t_span), total_dim)
        # Возвращаем только исходные переменные, без начальной точки
        return states[:, 1:, :self.original_dim]  # (batch, T-1, original_dim)

    # --- Регуляризации, доступные тренеру ----------------------------------
    def spectral_radius_penalty(self):
        """Мягкий штраф на положительную часть real-spectrum матрицы A.

        Penalty = sum_i softplus(Re(eig_i(A)) + eps).
        Стремится держать собственные числа A в левой полуплоскости.
        """
        A = self.ode_func.linear.effective_A()
        eig = torch.linalg.eigvals(A)
        re = eig.real
        # softplus сглаживает max(0, re), не штрафует сильно отрицательные значения
        return F.softplus(re + 1e-3).sum()

    def aug_init_penalty(self):
        """L2 на абсолютную величину начальных aug-каналов (или их предсказаний)."""
        if self.encode_aug:
            # штраф на матрицы энкодера — лёгкая регуляризация
            return sum((p ** 2).sum() for p in self.aug_encoder.parameters())
        return (self.aug_init ** 2).sum()


# --------------


# === Factory ===============================================================
def build_model(model_type, dim, hidden, seq_len, aug=2, gamma_init=0.1, alpha_init=0.0,
                n_freqs=4, base_freq=1.0, use_control=False,
                # --- resnode-specific knobs ---
                use_nonlinear=False, encode_aug=True,
                init_decay=0.05, skew_scale=1.0, sym_scale=0.1,
                hidden_corr=32, hidden_ctrl=32, hidden_enc=32,
                resnode_solver='dopri5'):
    if model_type == "node":
        return NeuralODE(dim, hidden)
    if model_type == "anode":
        return ANODE(dim, hidden, aug=aug)
    if model_type == "csode":
        return CSODE(dim, hidden, gamma_init=gamma_init)
    if model_type == "hnode":
        return HNODE(dim, hidden, aug=aug if aug > 0 else 4,
                     gamma_init=gamma_init, alpha_init=alpha_init)
    if model_type == "rnode":
        return RNODE(dim, hidden, aug=aug if aug > 0 else 4,
                     gamma_init=gamma_init, alpha_init=alpha_init)
    if model_type == "snode":
        return SNODE(dim, hidden, aug=aug if aug > 0 else 4,
                     gamma_init=gamma_init, alpha_init=alpha_init)
    if model_type == "lnode":
        return LNODE(dim, hidden, aug=aug if aug > 0 else 4,
                     gamma_init=gamma_init, alpha_init=alpha_init)
    if model_type == "fnode":
        return FNODE(dim, hidden, aug=aug if aug > 0 else 4,
                     gamma_init=gamma_init, alpha_init=alpha_init)
    if model_type == "fnode_v2":
        return FNODEV2(dim, hidden, aug=aug if aug > 0 else 4,
                       gamma_init=-4.0, alpha_init=alpha_init)
    if model_type == 'resnode':
        return ResilientNeuralODE(
            original_dim=dim,
            augmented_dim=aug,
            use_control=use_control,
            use_nonlinear=use_nonlinear,
            encode_aug=encode_aug,
            hidden_lin=hidden, hidden_nonlin=hidden,
            hidden_corr=hidden_corr, hidden_ctrl=hidden_ctrl,
            hidden_enc=hidden_enc,
            init_decay=init_decay,
            skew_scale=skew_scale,
            sym_scale=sym_scale,
            solver=resnode_solver,
        )

    if model_type == "mlp":
        return MLP(dim, hidden, seq_len)
    if model_type == "rnn":
        return RNN(dim, hidden, seq_len)
    raise ValueError(f"Unknown model_type: {model_type}")


# Models that support .regularization(y_sample) for Jacobian/kinetic penalties
REGULARIZED_MODELS = {"wpnode", "hnode", "tnode", "rnode", "snode", "lnode", "fnode", "fnode_v2", "resnode"}
ODE_MODELS = {"node", "anode", "csode", "hnode",
              "rnode", "snode", "lnode", "fnode", "fnode_v2", "resnode"}

