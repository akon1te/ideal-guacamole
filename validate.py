"""
Validate and visualise a trained model via autoregressive rollout.

Reports:
  - Trajectory MAE / RMSE on train and val regions
  - Time-of-validity T_valid (first crossing of normalized RMSE threshold)
  - Spectral distance (L1 between power spectra)
  - Per-component Wasserstein-1 distance
  - Per-component histogram-based KL divergence
  - Burst statistics (Hindmarsh--Rose only): mean burst duration and ISI

Usage:
    python validate.py --ckpt checkpoints/rossler_node.pt
    python validate.py --ckpt checkpoints/hindmarsh_rose_hnode.pt --t_max 1500
"""
import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dynsys.data import generate
from src.dynsys.evaluation.dynamics_metrics import (
    autocorrelation_1d,
    autocorrelation_distance,
    boundedness_metrics,
    dominant_frequency,
    finite_time_divergence_rate,
    phase_space_chamfer,
)
from src.dynsys.models.registry import build_model, ode_model_names

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",  required=True)
parser.add_argument("--t_max", type=float, default=None,
                    help="Rollout horizon (physical time). Default: full trajectory.")
parser.add_argument("--split", type=float, default=0.8,
                    help="Train/val split fraction used during training (visual marker).")
parser.add_argument("--bins",  type=int,   default=50,
                    help="Histogram bins for KL divergence estimation.")
parser.add_argument("--acf_lags", type=int, default=100,
                    help="Maximum lag for autocorrelation comparison.")
parser.add_argument("--phase_samples", type=int, default=2000,
                    help="Maximum deterministic samples for phase-space Chamfer distance.")
parser.add_argument("--bound_pad", type=float, default=0.05,
                    help="Relative padding around the truth range for boundedness checks.")
parser.add_argument("--divergence_points", type=int, default=2000,
                    help="Maximum samples for finite-time divergence-rate estimation.")
parser.add_argument("--divergence_steps", type=int, default=20,
                    help="Number of future steps used by the finite-time divergence fit.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
cfg  = ckpt["args"]
mean, std, dim, dt = ckpt["mean"], ckpt["std"], ckpt["dim"], ckpt["dt"]
system, model_type, hidden, seq_len = cfg["system"], cfg["model"], cfg["hidden"], cfg["seq_len"]
aug = cfg.get("aug", 2)
model = build_model(
    model_type, dim, hidden, seq_len, aug=aug,
    fnode_rank=cfg.get("fnode_rank", None),
    fnode_hidden_w=cfg.get("fnode_hidden_w", 16),
    fnode_alpha_init=cfg.get("fnode_alpha_init", -2.25),
    csode_layers=cfg.get("csode_layers", 3),
    csode_control_hidden=cfg.get("csode_control_hidden", None),
    csode_control_layers=cfg.get("csode_control_layers", 1),
    csode_solver=cfg.get("csode_solver", "rk4"),
).to(device)
model.load_state_dict(ckpt["model_state"]); model.eval()

# === Ground truth ==========================================================
_, y_np = generate(system)
y_np = y_np.astype(np.float32)
y_norm = (y_np - mean) / std

# === Autoregressive rollout (1-step at a time) =============================
if model_type in ode_model_names():
    t_grid = torch.tensor([0.0, dt], device=device)
else:
    t_grid = torch.linspace(0, dt * seq_len, seq_len + 1).to(device)

max_steps = len(y_norm) - 1
n_steps = max_steps if args.t_max is None else min(int(args.t_max / dt), max_steps)

preds_norm = [y_norm[0]]
cur = torch.tensor(y_norm[0:1], device=device)
with torch.no_grad():
    for _ in range(n_steps):
        nxt = model(cur, t_grid)[:, 0, :]
        preds_norm.append(nxt.squeeze(0).cpu().numpy())
        cur = nxt

preds_norm = np.array(preds_norm)
preds = preds_norm * std + mean
truth = y_np[:n_steps + 1]

# Train/val split boundary
split_idx = int(len(y_norm) * args.split)
split_t   = split_idx * dt


# === Metrics ===============================================================
err_norm = np.sqrt(np.mean((preds_norm - y_norm[:n_steps + 1]) ** 2, axis=1))
err_abs  = np.abs(preds - truth)


def stats(s, e):
    if e <= s:
        return None
    return {
        "mae":  err_abs[s:e].mean(),
        "rmse": np.sqrt(((preds[s:e] - truth[s:e]) ** 2).mean()),
    }


tr_idx_end = min(split_idx, n_steps + 1)
tr_stats = stats(0, tr_idx_end)
va_stats = stats(split_idx, n_steps + 1)

threshold = 1.0
above = np.where(err_norm > threshold)[0]
valid_steps = int(above[0]) if len(above) else n_steps
valid_time  = valid_steps * dt


def spectral_distance(a, b):
    """L1 distance between mean log-power spectra of a and b. Shape (T, D)."""
    A = np.abs(np.fft.rfft(a, axis=0)) ** 2
    B = np.abs(np.fft.rfft(b, axis=0)) ** 2
    # log-scale to balance dominant peaks vs background
    eps = 1e-12
    return np.mean(np.abs(np.log(A + eps) - np.log(B + eps)))


def wasserstein_1d(a, b):
    """Per-component Wasserstein-1 distance via sorted-CDF identity.

    a, b: (T, D) arrays. Returns mean over components.
    """
    out = []
    for d in range(a.shape[1]):
        # Resample to equal length for pure W1 between empirical 1D dists
        n = min(len(a), len(b))
        sa = np.sort(np.random.choice(a[:, d], size=n, replace=False)) \
            if len(a) > n else np.sort(a[:, d])
        sb = np.sort(np.random.choice(b[:, d], size=n, replace=False)) \
            if len(b) > n else np.sort(b[:, d])
        out.append(np.mean(np.abs(sa - sb)))
    return np.mean(out), out


def kl_hist_1d(a, b, bins=50):
    """Per-component histogram-based symmetric-KL = KL(a||b) + KL(b||a)."""
    out = []
    for d in range(a.shape[1]):
        lo = min(a[:, d].min(), b[:, d].min())
        hi = max(a[:, d].max(), b[:, d].max())
        edges = np.linspace(lo, hi, bins + 1)
        pa, _ = np.histogram(a[:, d], bins=edges, density=True)
        pb, _ = np.histogram(b[:, d], bins=edges, density=True)
        # Add small constant + renormalize so KL is finite
        pa = pa + 1e-9
        pb = pb + 1e-9
        pa = pa / pa.sum()
        pb = pb / pb.sum()
        kl_ab = np.sum(pa * np.log(pa / pb))
        kl_ba = np.sum(pb * np.log(pb / pa))
        out.append(0.5 * (kl_ab + kl_ba))
    return np.mean(out), out


def burst_stats(x_signal, dt, threshold=0.0, min_gap_steps=10):
    """Estimate mean burst duration and inter-spike interval (ISI) for the
    Hindmarsh--Rose `x` component.

    A burst is detected by upward zero-crossings of `x_signal - threshold`
    that are at least `min_gap_steps` apart (groups close crossings into
    a single burst).

    Returns: (mean_burst_dur, mean_ISI) in physical time units.
    Returns (nan, nan) if fewer than 2 bursts detected.
    """
    s = x_signal - threshold
    crossings = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if len(crossings) < 2:
        return float("nan"), float("nan")
    # Group consecutive crossings closer than min_gap_steps -> one burst
    burst_starts = [crossings[0]]
    for c in crossings[1:]:
        if c - burst_starts[-1] > min_gap_steps:
            burst_starts.append(c)
    burst_starts = np.array(burst_starts)
    if len(burst_starts) < 2:
        return float("nan"), float("nan")

    # Burst duration: time from start to next downward zero-crossing within window
    burst_durs = []
    for i, st in enumerate(burst_starts):
        end_limit = burst_starts[i + 1] if i + 1 < len(burst_starts) else len(s) - 1
        # find first downward crossing after st
        sub = s[st:end_limit]
        down = np.where((sub[:-1] >= 0) & (sub[1:] < 0))[0]
        if len(down):
            burst_durs.append(down[0] * dt)
    isi = np.diff(burst_starts) * dt
    mean_dur = float(np.mean(burst_durs)) if burst_durs else float("nan")
    mean_isi = float(np.mean(isi))
    return mean_dur, mean_isi


# === Compute distributional metrics on val region ==========================
# A short diagnostic rollout can end before the validation boundary.  In that
# case use its available region rather than passing an empty array to FFT and
# distribution metrics; full experiment runs still score validation only.
v1 = n_steps + 1
v0 = split_idx if split_idx < v1 else 0
if v0 == 0 and split_idx >= v1:
    print("  note: rollout ends before validation split; dynamic metrics use "
          "the available rollout region.")
spec_d = spectral_distance(preds_norm[v0:v1], y_norm[v0:v1])
wd_mean, wd_per = wasserstein_1d(preds_norm[v0:v1], y_norm[v0:v1])
kl_mean, kl_per = kl_hist_1d(preds_norm[v0:v1], y_norm[v0:v1], bins=args.bins)
acf_mean, acf_per = autocorrelation_distance(
    preds_norm[v0:v1], y_norm[v0:v1], max_lags=args.acf_lags,
)
dominant_pred = dominant_frequency(preds_norm[v0:v1], dt)
dominant_true = dominant_frequency(y_norm[v0:v1], dt)
dominant_error = np.abs(np.asarray(dominant_pred) - np.asarray(dominant_true))
phase_chamfer = phase_space_chamfer(
    preds_norm[v0:v1], y_norm[v0:v1], max_samples=args.phase_samples,
)
bound_rate, bound_excess = boundedness_metrics(
    preds[v0:v1], truth[v0:v1], relative_pad=args.bound_pad,
)
divergence_pred, divergence_pred_pairs = finite_time_divergence_rate(
    preds_norm[v0:v1], dt, max_points=args.divergence_points,
    fit_steps=args.divergence_steps,
)
divergence_true, divergence_true_pairs = finite_time_divergence_rate(
    y_norm[v0:v1], dt, max_points=args.divergence_points,
    fit_steps=args.divergence_steps,
)

# Burst stats (Hindmarsh--Rose only)
burst_pred = burst_stats(preds[v0:v1, 0], dt)
burst_true = burst_stats(truth[v0:v1, 0], dt)


# === Print summary =========================================================
print(f"\n{system}  |  {model_type}   horizon={n_steps * dt:.1f}  "
      f"({n_steps} steps, dt={dt:.4f})")
print(f"split @ t={split_t:.1f} (idx={split_idx})")
if tr_stats:
    print(f"  train-region  MAE={tr_stats['mae']:.5f}  RMSE={tr_stats['rmse']:.5f}")
if va_stats:
    print(f"  val-region    MAE={va_stats['mae']:.5f}  RMSE={va_stats['rmse']:.5f}")
print(f"  T_valid       = {valid_time:.2f}  (threshold={threshold} sigma, "
      f"steps={valid_steps})")
print(f"  spectral-L1   = {spec_d:.5f}")
print(f"  Wasserstein-1 = {wd_mean:.5f}  (per-component: "
      f"{', '.join(f'{w:.4f}' for w in wd_per)})")
print(f"  sym-KL        = {kl_mean:.5f}  (per-component: "
      f"{', '.join(f'{k:.4f}' for k in kl_per)})")
print(f"  ACF-L1        = {acf_mean:.5f}  (per-component: "
      f"{', '.join(f'{a:.4f}' for a in acf_per)})")
print(f"  dom-freq err  = {np.nanmean(dominant_error):.5f} Hz  (per-component: "
      f"{', '.join(f'{f:.4f}' for f in dominant_error)})")
print(f"  phase-Chamfer = {phase_chamfer:.5f}")
print(f"  out-of-bounds = {bound_rate:.2%}  mean-excess={bound_excess:.5f}")
print(f"  finite-time divergence rate: pred={divergence_pred:.5f}, "
      f"truth={divergence_true:.5f}")
if system == "hindmarsh_rose":
    print(f"  burst dur     pred={burst_pred[0]:.3f}  truth={burst_true[0]:.3f}")
    print(f"  ISI           pred={burst_pred[1]:.3f}  truth={burst_true[1]:.3f}")
print()


# === Plots: time-series + per-step error ==================================
t_axis = np.arange(n_steps + 1) * dt
var_names = ["x", "y", "z"][:dim]


def shade_split(ax):
    if split_t < t_axis[-1]:
        ax.axvspan(split_t, t_axis[-1], color="orange", alpha=0.08, zorder=0)
        ax.axvline(split_t, color="orange", lw=1.0, ls=":", zorder=1)


fig, axes = plt.subplots(dim + 1, 1, figsize=(12, 2.5 * (dim + 1)))
for i in range(dim):
    shade_split(axes[i])
    axes[i].plot(t_axis, truth[:, i], lw=1.0, label="truth")
    axes[i].plot(t_axis, preds[:, i], lw=1.0, ls="--", label="pred")
    axes[i].set_ylabel(var_names[i]); axes[i].legend(fontsize=8, loc="upper right")
shade_split(axes[-1])
axes[-1].semilogy(t_axis, err_norm, color="red")
axes[-1].axhline(threshold, color="gray", ls="--")
axes[-1].set_ylabel("norm. RMSE"); axes[-1].set_xlabel("t")
fig.suptitle(f"{system} -- {model_type}   (orange = val region, t >= {split_t:.1f})")
plt.tight_layout()
out = args.ckpt.replace(".pt", "_val.png")
plt.savefig(out, dpi=120); print(f"saved -> {out}")

# === Plots: phase portrait + training curve + PSD =========================
fig2 = plt.figure(figsize=(20, 4))
ax_ph    = fig2.add_subplot(1, 4, 1)
ax_hist  = fig2.add_subplot(1, 4, 2)
ax_psd   = fig2.add_subplot(1, 4, 3)
ax_acf   = fig2.add_subplot(1, 4, 4)

s = min(split_idx, n_steps + 1)
ax_ph.plot(truth[:s, 0],  truth[:s, 1],  lw=0.5, color="C0", label="truth (train)")
ax_ph.plot(truth[s:, 0],  truth[s:, 1],  lw=0.5, color="C2", label="truth (val)")
ax_ph.plot(preds[:s, 0],  preds[:s, 1],  lw=0.5, ls="--", color="C1", label="pred (train)")
ax_ph.plot(preds[s:, 0],  preds[s:, 1],  lw=0.5, ls="--", color="C3", label="pred (val)")
ax_ph.set_xlabel("x"); ax_ph.set_ylabel("y"); ax_ph.legend(fontsize=7)
ax_ph.set_title(f"phase  {system} {model_type}")

# Histogram on val region (component 0)
ax_hist.hist(truth[v0:v1, 0], bins=args.bins, alpha=0.5, density=True, label="truth")
ax_hist.hist(preds[v0:v1, 0], bins=args.bins, alpha=0.5, density=True, label="pred")
ax_hist.set_xlabel(var_names[0]); ax_hist.set_ylabel("density")
ax_hist.set_title(f"distribution {var_names[0]}  (sym-KL={kl_per[0]:.3f})")
ax_hist.legend(fontsize=8)

# PSD on val region (component 0)
freqs = np.fft.rfftfreq(v1 - v0, d=dt)
psd_t = np.abs(np.fft.rfft(truth[v0:v1, 0])) ** 2
psd_p = np.abs(np.fft.rfft(preds[v0:v1, 0])) ** 2
ax_psd.semilogy(freqs, psd_t + 1e-12, label="truth")
ax_psd.semilogy(freqs, psd_p + 1e-12, label="pred", ls="--")
ax_psd.set_xlabel("frequency"); ax_psd.set_ylabel("|X(f)|^2")
ax_psd.set_title(f"PSD {var_names[0]}  (L1={spec_d:.3f})")
ax_psd.legend(fontsize=8)

# Autocorrelation on validation region (component 0)
acf_truth = autocorrelation_1d(y_norm[v0:v1, 0], args.acf_lags)
acf_pred = autocorrelation_1d(preds_norm[v0:v1, 0], args.acf_lags)
lags = np.arange(min(len(acf_truth), len(acf_pred))) * dt
ax_acf.plot(lags, acf_truth[:len(lags)], label="truth")
ax_acf.plot(lags, acf_pred[:len(lags)], label="pred", ls="--")
ax_acf.set_xlabel("lag"); ax_acf.set_ylabel("ACF")
ax_acf.set_title(f"autocorrelation {var_names[0]}")
ax_acf.legend(fontsize=8)

plt.tight_layout()
out2 = args.ckpt.replace(".pt", "_phase.png")
plt.savefig(out2, dpi=120); print(f"saved -> {out2}")

# === Plots: training history ==============================================
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 4))
ax3.semilogy(ckpt["history"]["train"], label="train")
ax3.semilogy(ckpt["history"]["val"],   label="val")
if "horizon" in ckpt["history"] and len(set(ckpt["history"]["horizon"])) > 1:
    ax3b = ax3.twinx()
    ax3b.plot(ckpt["history"]["horizon"], color="gray", lw=0.8, alpha=0.6,
              label="curriculum L")
    ax3b.set_ylabel("curriculum horizon", color="gray")
ax3.set_xlabel("epoch"); ax3.set_ylabel("MSE"); ax3.legend()
ax3.set_title(f"training history -- {system} {model_type}")
plt.tight_layout()
out3 = args.ckpt.replace(".pt", "_history.png")
plt.savefig(out3, dpi=120); print(f"saved -> {out3}")

# === Metrics dump (JSON next to the checkpoint) ===========================
def _to_py(x):
    """Recursively coerce numpy/torch scalars to plain Python for JSON."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [_to_py(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    return x


hist = ckpt.get("history", {})
train_hist = hist.get("train", [])
val_hist   = hist.get("val", [])
best_val   = float(min(val_hist)) if len(val_hist) else None
final_val  = float(val_hist[-1])  if len(val_hist) else None
final_tr   = float(train_hist[-1]) if len(train_hist) else None

# Parameter count (from the loaded model)
n_params = int(sum(p.numel() for p in model.parameters()))

metrics = {
    "ckpt":          os.path.abspath(args.ckpt),
    "system":        system,
    "model_type":    model_type,
    "hidden":        hidden,
    "aug":           aug,
    "seq_len":       seq_len,
    "dim":           dim,
    "dt":            float(dt),
    "n_params":      n_params,
    "horizon": {
        "n_steps":   int(n_steps),
        "t_max":     float(n_steps * dt),
    },
    "split": {
        "fraction":  float(args.split),
        "idx":       int(split_idx),
        "t":         float(split_t),
    },
    "rmse_mae": {
        "train":     _to_py(tr_stats) if tr_stats else None,
        "val":       _to_py(va_stats) if va_stats else None,
    },
    "t_valid": {
        "time":      float(valid_time),
        "steps":     int(valid_steps),
        "threshold": float(threshold),
    },
    "spectral_l1":   float(spec_d),
    "wasserstein_1": {
        "mean":      float(wd_mean),
        "per_component": _to_py(wd_per),
    },
    "sym_kl": {
        "mean":      float(kl_mean),
        "per_component": _to_py(kl_per),
        "bins":      int(args.bins),
    },
    "temporal_structure": {
        "autocorrelation_l1": {
            "mean": float(acf_mean),
            "per_component": _to_py(acf_per),
            "max_lags": int(args.acf_lags),
        },
        "dominant_frequency_hz": {
            "pred": _to_py(dominant_pred),
            "truth": _to_py(dominant_true),
            "abs_error_mean": float(np.nanmean(dominant_error)),
            "abs_error_per_component": _to_py(dominant_error),
        },
    },
    "attractor_geometry": {
        "phase_space_chamfer": float(phase_chamfer),
        "max_samples": int(args.phase_samples),
    },
    "boundedness": {
        "relative_truth_range_pad": float(args.bound_pad),
        "outside_truth_envelope_rate": float(bound_rate),
        "mean_excess": float(bound_excess),
    },
    "finite_time_divergence": {
        "pred_rate": float(divergence_pred),
        "truth_rate": float(divergence_true),
        "abs_error": float(abs(divergence_pred - divergence_true)),
        "pred_pairs": int(divergence_pred_pairs),
        "truth_pairs": int(divergence_true_pairs),
        "fit_steps": int(args.divergence_steps),
        "max_points": int(args.divergence_points),
        "note": "Local neighbour-separation proxy, not a formal Lyapunov exponent.",
    },
    "burst_stats": {
        "pred_dur":  float(burst_pred[0]),
        "true_dur":  float(burst_true[0]),
        "pred_isi":  float(burst_pred[1]),
        "true_isi":  float(burst_true[1]),
    } if system == "hindmarsh_rose" else None,
    "training": {
        "best_val":  best_val,
        "final_val": final_val,
        "final_train": final_tr,
        "n_epochs":  len(val_hist),
    },
    "train_args":    _to_py(cfg),
}

out4 = args.ckpt.replace(".pt", "_metrics.json")
with open(out4, "w") as f:
    json.dump(metrics, f, indent=2, default=float)
print(f"saved -> {out4}")
