import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynsys.models.integrators import rk4_integrate

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


class LegacyDissipativeODEFunc(nn.Module):
    """Archived pre-refactor stable ODE vector field; not publicly registered.

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

class RNODEFunc(nn.Module):

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


class FNODEFunc(nn.Module):

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


def hutchinson_jacobian_norm(func, y, t=None, n_samples=1):

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


class NeuralODE(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.func = ODEFunc(dim, hidden)

    def forward(self, y0, t_grid):
        traj = rk4_integrate(self.func, y0, t_grid)   # (T+1, B, dim)
        return traj[1:].permute(1, 0, 2)              # (B, T, dim)


class ANODE(nn.Module):
    def __init__(self, dim, hidden, aug=2):
        super().__init__()
        self.dim, self.aug = dim, aug
        self.func = ODEFunc(dim + aug, hidden)

    def forward(self, y0, t_grid):
        B = y0.size(0)
        z0 = torch.cat([y0, y0.new_zeros(B, self.aug)], dim=1)
        traj = rk4_integrate(self.func, z0, t_grid)
        return traj[1:, :, :self.dim].permute(1, 0, 2)


class LegacyDissipativeODE(nn.Module):
    """Archived wrapper for the pre-refactor stable ODE."""
    def __init__(self, dim, hidden, gamma_init=0.1):
        super().__init__()
        self.func = LegacyDissipativeODEFunc(dim, hidden, gamma_init=gamma_init)

    def forward(self, y0, t_grid):
        traj = rk4_integrate(self.func, y0, t_grid)
        return traj[1:].permute(1, 0, 2)


class HNODE(nn.Module):

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

    def __init__(self, original_dim, aug_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, aug_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y0):
        return self.net(y0)



# === Factory ===============================================================
def build_model(model_type, dim, hidden, seq_len, aug=2, gamma_init=0.1, alpha_init=0.0,
                n_freqs=4, base_freq=1.0, use_control=False,
                # --- resnode-specific knobs ---
                use_nonlinear=False, encode_aug=True,
                init_decay=0.05, skew_scale=1.0, sym_scale=0.1,
                hidden_corr=32, hidden_ctrl=32, hidden_enc=32,
                resnode_solver='dopri5'):
    # Compatibility shim: new code imports the registry directly.
    from src.dynsys.models.registry import build_model as _build_model
    return _build_model(
        model_type, dim, hidden, seq_len, aug=aug, gamma_init=gamma_init,
        alpha_init=alpha_init, n_freqs=n_freqs, base_freq=base_freq,
        use_control=use_control, use_nonlinear=use_nonlinear,
        encode_aug=encode_aug, init_decay=init_decay, skew_scale=skew_scale,
        sym_scale=sym_scale, hidden_corr=hidden_corr, hidden_ctrl=hidden_ctrl,
        hidden_enc=hidden_enc, resnode_solver=resnode_solver,
    )


# Models that support .regularization(y_sample) for Jacobian/kinetic penalties
# Kept for compatibility with code importing the legacy module directly.
from src.dynsys.models.csode import CSODE, CSODEFunc

REGULARIZED_MODELS = {"csode", "fnode"}
ODE_MODELS = {"node", "anode", "csode", "fnode"}
