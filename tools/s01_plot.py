#!/usr/bin/env python3
"""
Figure for the S01 SMB-phase tests (diagnostic only): inferred SMB(z) of chosen reruns against the
synthetic law (left) and their data-misfit history (right).

    python tools/s01_plot.py out.png slid_0.4519 base:original fx_rmse:"border fix" ...
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(CX_DIR, "results", "EXP01_S01_cnn")
# categorical slots in fixed order (dataviz reference palette, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"


def hat(z, v, z0, dz):
    t = (z - z0) / dz
    return (np.clip(1.0 - np.abs(t[..., None] - np.arange(v.size)), 0.0, None) * v).sum(-1)


def main():
    out, member, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    zf = np.arange(1669, 2894, 5.0)
    fig, (ax, al) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax.plot(np.minimum(0.01 * (zf - 2350), 5.0), zf, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=10, label="truth (S01 law)")
    for i, spec in enumerate(specs):
        tag, _, label = spec.partition(":")
        sd = os.path.join(ROOT, f"smb_{tag}", member, "smb_inference")
        v = np.load(os.path.join(sd, "smb_vec.npy"))
        z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
        p = hat(zf, v, z0, dz)
        rmse = np.sqrt(np.mean((p - np.minimum(0.01 * (zf - 2350), 5.0)) ** 2))
        c = COLORS[i % len(COLORS)]
        ax.plot(p, zf, color=c, lw=2.0, label=f"{label or tag}  (RMSE {rmse:.2f})")
        dh = np.load(os.path.join(sd, "data_history.npy"))
        al.plot(np.arange(1, dh.size + 1), dh / dh[0], color=c, lw=2.0, marker="o", ms=3, label=label or tag)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m)")
    ax.grid(True, color="#e4e3df", lw=0.5); ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title(f"S01 inferred profile vs truth ({member})", fontsize=9, color=INK)
    al.set_xlabel("L-BFGS iteration"); al.set_ylabel("data misfit / first iteration")
    al.grid(True, color="#e4e3df", lw=0.5); al.legend(fontsize=7, frameon=False)
    al.set_title("data-term history (relative)", fontsize=9, color=INK)
    for a_ in (ax, al):
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
