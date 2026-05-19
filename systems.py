"""
Trajectory generators for three dynamical systems.
Usage: python systems.py --system rossler --save data/rossler.npz
"""
import argparse
import numpy as np
from scipy.integrate import solve_ivp


def rossler(t, y, a=0.2, b=0.2, c=5.7):
    x, y_, z = y
    return [-y_ - z, x + a * y_, b + z * (x - c)]


def hindmarsh_rose(t, y, I=3.25, r=0.006, s=4.0, alpha=-1.6, a=1.0, b=3.0, c=1.0, d=5.0):
    x, yv, z = y
    dx = yv - a * x**3 + b * x**2 - z + I
    dy = c - d * x**2 - yv
    dz = r * (s * (x - alpha) - z)
    return [dx, dy, dz]


def quasi_periodic(t, y, lam=0.5, beta=1/18, kappa=0.02, omega0=5.1):
    x, yv, z = y
    dx = yv
    dy = (lam + z + x**2 - beta * x**4) * yv - omega0**2 * x
    dz = -z - kappa * yv**2
    return [dx, dy, dz]


SYSTEMS = {
    "rossler":        (rossler,        [1.0, 0.0, 0.0],  (0, 200),  0.05),
    "hindmarsh_rose": (hindmarsh_rose, [0.0, 0.0, 0.0],  (0, 2000), 0.1),
    "quasi_periodic": (quasi_periodic, [0.1, 0.0, 0.0],  (0, 200),  0.02),
}


def generate(system: str, discard: float = 0.2):
    fn, y0, tspan, dt = SYSTEMS[system]
    t_eval = np.arange(tspan[0], tspan[1], dt)
    sol = solve_ivp(fn, tspan, y0, t_eval=t_eval, method="RK45",
                    rtol=1e-9, atol=1e-9, dense_output=False)
    t, y = sol.t, sol.y.T          # y: (T, D)
    cut = int(len(t) * discard)    # discard transient
    return t[cut:] - t[cut], y[cut:]


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=list(SYSTEMS), required=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    t, y = generate(args.system)
    print(f"{args.system}: t={t.shape}, y={y.shape}, dim={y.shape[1]}")

    save = args.save or f"data/{args.system}.npz"
    os.makedirs(os.path.dirname(save), exist_ok=True)
    np.savez(save, t=t, y=y)
    print(f"Saved → {save}")

    if args.plot:
        var_names = ["x", "y", "z"][:y.shape[1]]
        dim = y.shape[1]

        # One figure: 3 time-series subplots (left) + 3D phase portrait (right)
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(args.system, fontsize=14)

        # time series — left column (3 rows)
        for i in range(dim):
            ax = fig.add_subplot(dim, 2, 2 * i + 1)
            ax.plot(t, y[:, i], lw=0.6)
            ax.set_ylabel(var_names[i])
            if i < dim - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t")

        # 3D phase portrait — right column (spans all rows)
        ax3d = fig.add_subplot(1, 2, 2, projection="3d")
        ax3d.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.3, alpha=0.8)
        ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
        ax3d.set_title("Phase portrait x–y–z")

        fig.tight_layout()
        out = f"data/{args.system}_dataset.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Plot saved → {out}")
        plt.show()
