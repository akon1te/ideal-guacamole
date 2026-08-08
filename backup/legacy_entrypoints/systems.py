"""Backward-compatible CLI for trajectory generation.

The reusable system definitions now live in :mod:`src.dynsys.data.systems`.
"""
import argparse
import os

import numpy as np

from src.dynsys.data.systems import (
    SYSTEMS,
    generate,
    hindmarsh_rose,
    quasi_periodic,
    rossler,
)

__all__ = ["SYSTEMS", "generate", "hindmarsh_rose", "quasi_periodic", "rossler"]


if __name__ == "__main__":
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
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(args.system, fontsize=14)
        names = ["x", "y", "z"][: y.shape[1]]
        for index, name in enumerate(names):
            axis = fig.add_subplot(len(names), 2, 2 * index + 1)
            axis.plot(t, y[:, index], lw=0.6)
            axis.set_ylabel(name)
            if index == len(names) - 1:
                axis.set_xlabel("t")
        axis3d = fig.add_subplot(1, 2, 2, projection="3d")
        axis3d.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.3, alpha=0.8)
        axis3d.set(xlabel="x", ylabel="y", zlabel="z", title="Phase portrait x–y–z")
        fig.tight_layout()
        output = f"data/{args.system}_dataset.png"
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"Plot saved → {output}")
        plt.show()
