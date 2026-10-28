#!/usr/bin/env python3
"""
S01 family-mean SMB(z) of exported runs against the synthetic law (diagnostic only).

    python tools/s01_plot_family.py out.png EXP01 GROUP_UNIL:original A_rampfx:"..." E_bands_fam:"..."
A spec is <tag>:<label>; tag GROUP_UNIL means the current submission file.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import xarray as xr               # noqa: E402

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    out, exp, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    zf = np.arange(1669, 2894, 5.0)
    law = np.minimum(0.01 * (zf - 2350), 5.0)
    fig, (ax, ae) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    ax.plot(law, zf, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=10, label="truth (S01 law)")
    for i, spec in enumerate(specs):
        tag, _, label = spec.partition(":")
        f = (os.path.join(CX_DIR, "GROUP_UNIL", exp, f"{exp}_S01_method01.nc") if tag == "GROUP_UNIL" else
             os.path.join(CX_DIR, "results", f"{exp}_S01_cnn", f"smb_{tag}", "export", f"{exp}_S01_method01.nc"))
        d = xr.open_dataset(f)
        p = np.interp(zf, d.z.values, d.SMB_PROFILE.values)
        rmse = np.sqrt(np.mean((p - law) ** 2))
        c = COLORS[i % len(COLORS)]
        ax.plot(p, zf, color=c, lw=2.0, label=f"{label or tag}  (RMSE {rmse:.2f})")
        ae.plot(p - law, zf, color=c, lw=2.0)
    for a_ in (ax, ae):
        a_.axvline(0, color=MUTED, lw=0.6)
        a_.grid(True, color="#e4e3df", lw=0.5)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m)")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title(f"{exp} S01: family-mean SMB(z) vs truth", fontsize=9, color=INK)
    ae.set_xlabel("inferred - truth (m i.e. yr$^{-1}$)")
    ae.set_title("profile error", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
