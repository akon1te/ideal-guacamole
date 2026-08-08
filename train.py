
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.dynsys.data import (
    SYSTEMS,
    apply_training_constraints,
    generate,
    make_windows,
    normalise_train_validation,
    split_trajectory,
)
from src.dynsys.models.registry import (
    available_model_names,
    build_model,
    ode_model_names,
    regularized_model_names,
)


class Tee:
    """Write to both stdout and a (line-buffered) file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

parser = argparse.ArgumentParser()
parser.add_argument("--system", choices=list(SYSTEMS), default="rossler")
parser.add_argument("--model", choices=available_model_names(), default="node")
parser.add_argument("--seq_len", type=int,   default=50)
parser.add_argument("--hidden",  type=int,   default=64)
parser.add_argument("--aug",     type=int,   default=2,  help="augmentation dims (ANODE/HNODE)")
parser.add_argument("--lr",      type=float, default=1e-3)
parser.add_argument("--epochs",  type=int,   default=200)
parser.add_argument("--batch",   type=int,   default=256)
parser.add_argument("--compile", action="store_true", help="wrap model in torch.compile")

# Curriculum learning
parser.add_argument("--curriculum", action="store_true",
                    help="enable curriculum: grow horizon 5 -> seq_len over training")
parser.add_argument("--curriculum_schedule", type=str, default="5,15,30,50",
                    help="comma-separated list of horizons; epoch range split evenly. "
                         "Last value should equal --seq_len.")

# Regularization (only used by wpnode/hnode; ignored otherwise)
parser.add_argument("--lambda_jac",  type=float, default=0.0,
                    help="weight on Jacobian (Hutchinson) regularization")
parser.add_argument("--lambda_kin",  type=float, default=0.0,
                    help="weight on kinetic ||f(y)||^2 regularization")
parser.add_argument("--lambda_spec", type=float, default=0.0,
                    help="weight on spectral (FFT PSD L1) loss; only for HNODE")

# Seeding & logging
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log_every", type=int, default=50,
                    help="print/log progress every N epochs")
parser.add_argument("--log_dir", type=str, default="logs",
                    help="directory for per-run text logs")
parser.add_argument("--run_name", type=str, default=None,
                    help="custom run name; default: <system>_<model>_<timestamp>")
parser.add_argument("--tvalid_every", type=int, default=20,
                    help="compute T_valid (autoregressive rollout) every N epochs; 0 disables")
parser.add_argument("--tvalid_threshold", type=float, default=0.5,
                    help="normalized-RMSE threshold for T_valid (in sigma units)")
parser.add_argument("--tvalid_max_steps", type=int, default=2000,
                    help="cap rollout length when computing T_valid")

# Teacher forcing (chunked rollout): split horizon L_curr into pieces of size
# tf_chunk; each piece starts from the TRUE trajectory point (not the model's
# previous prediction).  tf_chunk=0 disables (current behavior: single rollout).
# tf_chunk=1 = pure teacher forcing (1-step prediction).
parser.add_argument("--tf_chunk", type=int, default=0,
                    help="teacher-forcing chunk size: split each training sequence "
                         "into rollouts of this length, each starting from truth. "
                         "0=disabled (one long rollout), 1=pure teacher forcing. "
                         "Only affects ODE-family models.")

# === Group B (training-side tricks) ========================================
# B1: Scheduled sampling -- replace truth chunk-start with model prediction
#     (from previous chunk) with probability p. p schedule:
#       * "constant:0.0"  -- pure TF (default)
#       * "constant:0.3"  -- always 30% prob to use prediction
#       * "linear:0:0.5"  -- linear ramp from 0 to 0.5 over training
#       * "inv_sigmoid:k" -- inverse-sigmoid schedule (Bengio 2015), parameter k
parser.add_argument("--ss_schedule", type=str, default="constant:0.0",
                    help="scheduled sampling schedule for chunked TF; format: "
                         "constant:p | linear:p0:p1 | inv_sigmoid:k. "
                         "p = probability to feed PREDICTION instead of truth.")
# B2: Input noise on chunk-starts (data-augmentation form)
parser.add_argument("--input_noise", type=float, default=0.0,
                    help="Gaussian noise sigma added to chunk-start states "
                         "during training (in normalised units). 0=disabled.")
# B3: Gradient clipping + cosine LR schedule
parser.add_argument("--grad_clip", type=float, default=0.0,
                    help="max gradient norm (0=disabled).")
parser.add_argument("--cosine_lr", action="store_true",
                    help="enable cosine LR decay from --lr to lr*0.01")
parser.add_argument("--warmup_epochs", type=int, default=0,
                    help="linear LR warmup duration; only used with --cosine_lr.")
parser.add_argument("--fnode_rank", type=int, default=None,
                    help="rank of FNODE's state-dependent contraction factor")
parser.add_argument("--fnode_hidden_w", type=int, default=16,
                    help="hidden width of FNODE's contraction-factor network")
parser.add_argument("--fnode_alpha_init", type=float, default=-2.25,
                    help="initial log isotropic decay for FNODE")
parser.add_argument("--csode_layers", type=int, default=3,
                    help="depth of CSODE's time-conditioned main MLP")
parser.add_argument("--csode_control_hidden", type=int, default=None,
                    help="hidden width of CSODE control networks; default=--hidden")
parser.add_argument("--csode_control_layers", type=int, default=1,
                    help="depth of CSODE's time-conditioned auxiliary network")
parser.add_argument("--csode_solver", choices=["rk4", "euler"], default="rk4",
                    help="CSODE integration method; use euler to reproduce the paper's solver")


# === Robustness experiments ===============================================
# C1: Measurement noise on the TRAINING trajectory (validation kept clean for
# fair metric).  Sigma is in NORMALISED units (i.e. fraction of the per-coord
# standard deviation); 0=clean, 0.05=5% noise, etc.
parser.add_argument("--data_noise", type=float, default=0.0,
                    help="Gaussian measurement noise sigma applied to the "
                         "TRAINING trajectory (normalised units). "
                         "Validation trajectory is left clean. 0=disabled.")
# C2: Low-data regime: take only a CONTIGUOUS PREFIX of the training half of
# the trajectory (the validation half is unchanged).
parser.add_argument("--data_frac", type=float, default=1.0,
                    help="Fraction (0,1] of the training trajectory used; "
                         "a contiguous prefix is taken so the dynamics is "
                         "preserved. Default 1.0 = all training data.")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# === Logger setup ==========================================================
os.makedirs(args.log_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = args.run_name or f"{args.system}_{args.model}_{ts}"
log_path = os.path.join(args.log_dir, f"{run_name}.log")
log_file = open(log_path, "w", buffering=1)   # line-buffered
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"# Run: {run_name}")
print(f"# Log file: {log_path}")
print(f"# Args: {vars(args)}")
print(f"device={device}  system={args.system}  model={args.model}  "
      f"curriculum={args.curriculum}")

# === Data ==================================================================
t_np, y_np = generate(args.system)
y_np = y_np.astype(np.float32)
dim = y_np.shape[1]
dt  = float(t_np[1] - t_np[0])

y_train_raw, y_val_raw, _ = split_trajectory(y_np)
full_train_steps = len(y_train_raw)

# C2: restrict the available raw training prefix before fitting normalization
# statistics.  A low-data run must not observe the withheld train remainder.
if args.data_frac < 1.0:
    requested_keep = max(args.seq_len + 2, int(round(full_train_steps * args.data_frac)))
    n_keep = min(requested_keep, full_train_steps)
    print(f"data_frac={args.data_frac:.3f} -> using {n_keep}/{full_train_steps} "
          f"training timesteps (~{n_keep/full_train_steps*100:.1f}%)")
y_train_raw, _ = apply_training_constraints(
    y_train_raw, args.seq_len, data_fraction=args.data_frac,
    data_noise=0.0, seed=args.seed,
)

y_train_full, y_val_full, mean, std = normalise_train_validation(
    y_train_raw, y_val_raw,
)

# C1: add noise after normalisation, so sigma remains a fraction of the
# standard deviation of the actually available training trajectory.
if args.data_noise > 0:
    y_train_full, _ = apply_training_constraints(
        y_train_full, args.seq_len, data_fraction=1.0,
        data_noise=args.data_noise, seed=args.seed,
    )
    print(f"data_noise={args.data_noise:.3f} sigma added to training trajectory "
          f"(in normalised units, validation kept clean)")

X_tr, Y_tr = make_windows(y_train_full, args.seq_len)
X_va, Y_va = make_windows(y_val_full,   args.seq_len)
print(f"#train_windows={len(X_tr)}  #val_windows={len(X_va)}")

train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=args.batch, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, Y_va), batch_size=args.batch)

t_grid = torch.linspace(0, dt * args.seq_len, args.seq_len + 1).to(device)


# === Curriculum scheduler ==================================================
def build_curriculum(schedule_str, max_len, n_epochs, enabled):
    """Return list of length n_epochs with training horizon for each epoch.

    Two formats supported in schedule_str:
      * "20,40,60,100,150"            -- horizons split equally over n_epochs
      * "20:50,40:100,60:150,150:700" -- explicit (horizon:n_epochs) pairs;
                                         supports any positive weights, gets
                                         normalised to sum to n_epochs.
    Pairs format lets you spend MORE epochs on long horizons (which need more
    optimization steps to stabilise).
    """
    if not enabled:
        return [max_len] * n_epochs

    parts = [p.strip() for p in schedule_str.split(",") if p.strip()]
    if any(":" in p for p in parts):
        # explicit weighted form
        horizons, weights = [], []
        for p in parts:
            h, w = p.split(":")
            horizons.append(int(h))
            weights.append(float(w))
        total = sum(weights)
        # normalise weights -> per-phase epoch counts
        raw = [w / total * n_epochs for w in weights]
        counts = [max(1, int(round(r))) for r in raw]
        # adjust last phase to make total exactly n_epochs
        diff = n_epochs - sum(counts)
        counts[-1] = max(1, counts[-1] + diff)
    else:
        # equal split (original behaviour)
        horizons = [int(x) for x in parts]
        n_phases = len(horizons)
        per = n_epochs // n_phases
        counts = [per] * n_phases
        counts[-1] += n_epochs - sum(counts)

    assert horizons[-1] <= max_len, \
        f"last curriculum horizon {horizons[-1]} > seq_len {max_len}"

    sched = []
    for h, c in zip(horizons, counts):
        sched.extend([h] * c)
    # safety: pad/truncate to exactly n_epochs
    if len(sched) < n_epochs:
        sched.extend([horizons[-1]] * (n_epochs - len(sched)))
    return sched[:n_epochs]


curriculum = build_curriculum(args.curriculum_schedule, args.seq_len,
                              args.epochs, args.curriculum)
if args.curriculum:
    # summarize phases as (horizon, n_epochs) pairs in encountered order
    phase_summary = []
    for h in curriculum:
        if not phase_summary or phase_summary[-1][0] != h:
            phase_summary.append([h, 1])
        else:
            phase_summary[-1][1] += 1
    summary_str = ", ".join(f"{h}x{n}" for h, n in phase_summary)
    print(f"curriculum schedule: [{summary_str}]  (max seq_len={args.seq_len})")

if args.tf_chunk > 0:
    print(f"teacher forcing: tf_chunk={args.tf_chunk} "
          f"({'pure 1-step TF' if args.tf_chunk == 1 else 'chunked TF'})")


# === Loss helpers ==========================================================
def spectral_loss(pred, target):
    """L1 distance between power spectra of pred and target along time axis.

    pred, target: (B, T, D)
    """
    P = torch.fft.rfft(pred,   dim=1).abs().pow(2)
    Q = torch.fft.rfft(target, dim=1).abs().pow(2)
    return F.l1_loss(P, Q)


# === T_valid (autoregressive rollout horizon until error > threshold) =====
@torch.no_grad()
def compute_tvalid(model, y_full_norm, dt, threshold=1.0,
                   max_steps=2000, is_ode=True, ode_seq_len=50):
    """Run an autoregressive 1-step rollout from y_full_norm[0] and return
    the physical time at which normalized RMSE first exceeds `threshold`.

    y_full_norm: numpy (T, D) full normalized trajectory (used as ground truth)
    Returns: (tvalid_seconds, valid_steps)
    """
    n_steps = min(len(y_full_norm) - 1, max_steps)
    if is_ode:
        t_grid_local = torch.tensor([0.0, dt], device=device, dtype=torch.float32)
    else:
        t_grid_local = torch.linspace(0.0, dt * ode_seq_len,
                                      ode_seq_len + 1, device=device,
                                      dtype=torch.float32)

    cur = torch.tensor(y_full_norm[0:1], device=device, dtype=torch.float32)
    truth_t = torch.tensor(y_full_norm[: n_steps + 1], device=device, dtype=torch.float32)

    valid_steps = n_steps
    for k in range(1, n_steps + 1):
        nxt = model(cur, t_grid_local)[:, 0, :]
        err = torch.sqrt(((nxt[0] - truth_t[k]) ** 2).mean()).item()
        if err > threshold:
            valid_steps = k - 1
            break
        cur = nxt
    return valid_steps * dt, valid_steps


# === Model =================================================================
base_model = build_model(
    args.model, dim, args.hidden, args.seq_len, aug=args.aug,
    fnode_rank=args.fnode_rank,
    fnode_hidden_w=args.fnode_hidden_w,
    fnode_alpha_init=args.fnode_alpha_init,
    csode_layers=args.csode_layers,
    csode_control_hidden=args.csode_control_hidden,
    csode_control_layers=args.csode_control_layers,
    csode_solver=args.csode_solver,
).to(device)
model = torch.compile(base_model) if args.compile else base_model
opt     = torch.optim.Adam(base_model.parameters(), lr=args.lr)
mse_fn  = nn.MSELoss()

n_params = sum(p.numel() for p in base_model.parameters())
print(f"#params = {n_params}")

# === Scheduled sampling probability (B1) ===================================
def ss_prob(epoch, total_epochs, schedule_str):
    """Return probability of using model prediction (instead of truth) as
    the chunk-start during this epoch.  Format: 'constant:p' | 'linear:p0:p1'
    | 'inv_sigmoid:k' (Bengio 2015: p = 1 - k / (k + exp(epoch/k))).
    """
    parts = schedule_str.split(":")
    kind = parts[0]
    if kind == "constant":
        return float(parts[1])
    if kind == "linear":
        p0, p1 = float(parts[1]), float(parts[2])
        return p0 + (p1 - p0) * (epoch - 1) / max(1, total_epochs - 1)
    if kind == "inv_sigmoid":
        k = float(parts[1])
        import math
        return 1.0 - k / (k + math.exp((epoch - 1) / k))
    raise ValueError(f"Unknown ss_schedule: {schedule_str}")


_ss_p = ss_prob(1, args.epochs, args.ss_schedule)
if _ss_p > 0 or args.ss_schedule != "constant:0.0":
    print(f"scheduled sampling: schedule={args.ss_schedule}  p(epoch=1)={_ss_p:.3f}")
if args.input_noise > 0:
    print(f"input noise: sigma={args.input_noise}")
if args.grad_clip > 0:
    print(f"grad clipping: max_norm={args.grad_clip}")

# === Cosine LR + warmup (B3) ===============================================
if args.cosine_lr:
    import math
    def lr_lambda(epoch):
        # epoch is 0-indexed here (PyTorch convention)
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        # cosine decay from 1.0 to 0.01 after warmup
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    print(f"cosine LR: warmup={args.warmup_epochs}  base_lr={args.lr}  min_lr={args.lr*0.01}")
else:
    scheduler = None

is_ode = args.model in ode_model_names()
is_reg = args.model in regularized_model_names()

best_val, best_state = float("inf"), None
history = {"train": [], "val": [], "horizon": []}

# === Training loop =========================================================
for epoch in range(1, args.epochs + 1):
    L_curr = curriculum[epoch - 1]
    # slice the time grid to the curriculum horizon
    if is_ode:
        t_curr = t_grid[: L_curr + 1]
    else:
        # MLP/RNN are hard-wired to seq_len; can't shrink horizon at inference
        t_curr = t_grid

    # B1: scheduled sampling probability for this epoch
    p_ss = ss_prob(epoch, args.epochs, args.ss_schedule)

    model.train(); t0 = time.time(); tr_total = 0.0
    # decide whether teacher forcing is active for this batch (ODE only)
    tf_active = is_ode and args.tf_chunk > 0 and args.tf_chunk < L_curr
    # When scheduled sampling is on (p_ss > 0), we MUST run chunks
    # sequentially (batched B*K stacking can't pass predictions from
    # chunk k-1 to chunk k).
    ss_active = tf_active and p_ss > 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        # B2: input noise on the start point (non-TF: just x; TF: applied to
        # every chunk start below)
        if args.input_noise > 0 and not tf_active:
            x = x + args.input_noise * torch.randn_like(x)
        if is_ode:
            if tf_active:
                B = x.shape[0]
                chunk = args.tf_chunk
                traj = torch.cat([x.unsqueeze(1), y[:, :L_curr, :]], dim=1)   # (B, L_curr+1, D)
                starts = list(range(0, L_curr, chunk))
                chunk_starts = []
                chunk_targets = []
                for s in starts:
                    L_i = min(chunk, L_curr - s)
                    chunk_starts.append(traj[:, s, :])
                    chunk_targets.append(traj[:, s + 1 : s + 1 + L_i, :])

                if ss_active:
                    # Sequential chunk rollout with scheduled sampling.
                    # For each chunk k>=1, with prob p_ss replace truth start
                    # with the prediction from the previous chunk's last step.
                    preds_pieces = []
                    prev_last = None
                    for i, s in enumerate(starts):
                        L_i = chunk_targets[i].shape[1]
                        truth_start = chunk_starts[i]
                        if i == 0 or prev_last is None:
                            start = truth_start
                        else:
                            # per-sample bernoulli mask: 1 = use prediction
                            mask = (torch.rand(B, 1, device=device) < p_ss).float()
                            start = mask * prev_last + (1 - mask) * truth_start
                        if args.input_noise > 0:
                            start = start + args.input_noise * torch.randn_like(start)
                        t_chunk = t_grid[: L_i + 1]
                        p = model(start, t_chunk)                              # (B, L_i, D)
                        preds_pieces.append(p)
                        prev_last = p[:, -1, :]
                    pred = torch.cat(preds_pieces, dim=1)
                    tgt = torch.cat(chunk_targets, dim=1)
                else:
                    # Pure TF: stacked B*K batched rollout (fast path).
                    if args.input_noise > 0:
                        chunk_starts = [
                            cs + args.input_noise * torch.randn_like(cs)
                            for cs in chunk_starts
                        ]
                    if all(t.shape[1] == chunk for t in chunk_targets):
                        starts_t = torch.stack(chunk_starts, dim=1)            # (B, K, D)
                        K = starts_t.shape[1]
                        starts_flat = starts_t.reshape(B * K, -1)
                        t_chunk = t_grid[: chunk + 1]
                        pred_flat = model(starts_flat, t_chunk)
                        pred = pred_flat.reshape(B, K * chunk, -1)
                        tgt = torch.cat(chunk_targets, dim=1)
                    else:
                        full_idx = [i for i, t in enumerate(chunk_targets) if t.shape[1] == chunk]
                        short_idx = [i for i, t in enumerate(chunk_targets) if t.shape[1] != chunk]
                        preds_pieces = []
                        if full_idx:
                            starts_full = torch.stack([chunk_starts[i] for i in full_idx], dim=1)
                            Kf = starts_full.shape[1]
                            starts_flat = starts_full.reshape(B * Kf, -1)
                            t_chunk = t_grid[: chunk + 1]
                            p = model(starts_flat, t_chunk).reshape(B, Kf * chunk, -1)
                            preds_pieces.append(p)
                        for i in short_idx:
                            L_i = chunk_targets[i].shape[1]
                            t_short = t_grid[: L_i + 1]
                            p = model(chunk_starts[i], t_short)
                            preds_pieces.append(p)
                        pred = torch.cat(preds_pieces, dim=1)
                        tgt = torch.cat(chunk_targets, dim=1)
            else:
                # Original: single long rollout from x
                pred = model(x, t_curr)                  # (B, L_curr, D)
                tgt  = y[:, :L_curr, :]
        else:
            pred = model(x, t_curr)                  # (B, seq_len, D)
            tgt  = y                                  # full

        loss = mse_fn(pred, tgt)

        # Optional regularization (HNODE / WP-NODE)
        if is_reg and (args.lambda_jac > 0 or args.lambda_kin > 0):
            reg = base_model.regularization(x)
            if args.lambda_jac > 0:
                loss = loss + args.lambda_jac * reg["jac"]
            if args.lambda_kin > 0:
                loss = loss + args.lambda_kin * reg["kin"]

        # Optional spectral loss (HNODE only by convention; activates if lambda > 0)
        if args.lambda_spec > 0 and is_ode and L_curr >= 4:
            loss = loss + args.lambda_spec * spectral_loss(pred, tgt)

        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        opt.step()
        tr_total += loss.item() * len(x)
    tr_total /= len(X_tr)
    if scheduler is not None:
        scheduler.step()

    # === Validation always uses full seq_len for fair, comparable numbers ===
    model.eval(); va_total = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, t_grid)
            va_total += mse_fn(pred, y).item() * len(x)
    va_total /= len(X_va)

    history["train"].append(tr_total)
    history["val"].append(va_total)
    history["horizon"].append(L_curr)

    if va_total < best_val:
        best_val = va_total
        best_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    # === T_valid (autoregressive rollout from val-region start) ===========
    do_tvalid = (
        args.tvalid_every > 0
        and (epoch % args.tvalid_every == 0 or epoch == 1 or epoch == args.epochs)
    )
    tvalid_str = ""
    if do_tvalid:
        tvalid, vsteps = compute_tvalid(
            model, y_val_full, dt,
            threshold=args.tvalid_threshold,
            max_steps=args.tvalid_max_steps,
            is_ode=is_ode,
            ode_seq_len=args.seq_len,
        )
        history.setdefault("tvalid_epoch", []).append(epoch)
        history.setdefault("tvalid", []).append(tvalid)
        tvalid_str = f"  T_valid={tvalid:.2f}({vsteps} steps)"

    if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs or do_tvalid:
        extra = f" L={L_curr}" if args.curriculum else ""
        print(f"epoch {epoch:4d}/{args.epochs}  train={tr_total:.5f}  val={va_total:.5f}  "
              f"best={best_val:.5f}  t={time.time() - t0:.1f}s{extra}{tvalid_str}")

# === Save ==================================================================
# Save into a per-run directory so concurrent / sequential runs do not
# overwrite each other.  Layout:
#   checkpoints/<run_name>/<system>_<model>.pt
ckpt_dir = os.path.join("checkpoints", run_name)
os.makedirs(ckpt_dir, exist_ok=True)
path = os.path.join(ckpt_dir, f"{args.system}_{args.model}.pt")
torch.save({
    "model_state": best_state,
    "history": history,
    "args": vars(args),
    "mean": mean, "std": std, "dim": dim, "dt": dt,
    "run_name": run_name,
    "log_path": log_path,
}, path)
print(f"\nSaved -> {path}  best_val={best_val:.5f}")
print(f"Log    -> {log_path}")
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
