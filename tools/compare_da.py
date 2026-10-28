#!/usr/bin/env python3
"""
Side-by-side DA comparison of two Optuna sweeps (best completed trial of each): observed surface velocity vs
the DA's modelled velocity, velocity misfit, and DA thickness, on a shared colour scale.

    python tools/compare_da.py --old results/EXP01_G01_vel/sweep --new results/EXP01_S02_cnn/sweep \
        --old-label "SIADecompNet (paper stack)" --new-label "bundled CNN (dev stack)" \
        --out results/EXP01_S02_cnn/compare_da.png

Colour: one-hue blue ramp (light -> dark) for magnitudes, blue <-> neutral <-> red for signed differences
(dataviz skill palette). Both sweeps must be on the same input grid.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                      "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_bluered", ["#1c5cab", "#f0efec", "#d03b3b"])


def best_trial(sweep_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(sweep_dir, "trial_*", "trial_meta.json"))):
        m = json.load(open(p))
        if m.get("status") == "completed":
            m["dir"] = os.path.dirname(p)
            rows.append(m)
    if not rows:
        raise SystemExit(f"no completed trial in {sweep_dir}")
    return min(rows, key=lambda m: m["final_velsurf"]), len(rows)


def load(trial):
    ds = xr.open_dataset(os.path.join(trial["dir"], "geology-optimized.nc"))
    mask = ds["icemask"].values > 0.5
    out = dict(x=ds.x.values, y=ds.y.values, mask=mask, thk=ds["thk"].values.astype(float),
               vel=ds["velsurf_mag"].values.astype(float), obs=ds["velsurfobs_mag"].values.astype(float))
    ds.close()
    return out


def metrics(d, trial):
    m, dx = d["mask"], float(d["x"][1] - d["x"][0])
    err = (d["vel"] - d["obs"])[m]
    return {"DA cost (velsurf)": f"{trial['final_velsurf']:.2f}",
            "velocity RMSE": f"{np.sqrt(np.nanmean(err ** 2)):.1f} m/yr",
            "velocity bias": f"{np.nanmean(err):+.1f} m/yr",
            "mean thickness": f"{np.nanmean(d['thk'][m]):.0f} m",
            "ice volume": f"{np.nansum(d['thk'][m]) * dx ** 2 * 1e-9:.2f} km3",
            "slidingco": f"{trial['init_slidingco']:.3f}",
            "arrhenius": f"{trial['init_arrhenius']:.0f}"}


def show(ax, d, field, cmap, vmin, vmax, title, mask_ice=True):
    a = np.where(d["mask"], field, np.nan) if mask_ice else field
    asc = d["y"][0] < d["y"][-1]
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower" if asc else "upper",
                   extent=[d["x"].min(), d["x"].max(), d["y"].min(), d["y"].max()], interpolation="nearest")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old-label", default="old")
    ap.add_argument("--new-label", default="new")
    ap.add_argument("--title", default="Data assimilation: old vs new emulator")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    (to, no), (tn, nn) = best_trial(a.old), best_trial(a.new)
    do, dn = load(to), load(tn)
    vmax = float(np.nanpercentile(np.where(do["mask"], do["obs"], np.nan), 99))
    emax = 60.0
    hmax = float(np.nanpercentile(np.where(do["mask"], np.maximum(do["thk"], dn["thk"]), np.nan), 99.5))
    dmax = 150.0

    fig, axs = plt.subplots(3, 3, figsize=(13.5, 12), facecolor=SURFACE)
    fig.suptitle(a.title, fontsize=15, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.952, f"best finished Optuna trial of each sweep (lowest final DA velsurf cost): "
             f"old = best of {no}, new = best of {nn}. Same grid, same colour scales.",
             fontsize=10, color=INK2, ha="left")

    im = show(axs[0, 0], do, do["obs"], SEQ, 0, vmax, "Observed surface velocity")
    cbar(fig, im, axs[0, 0], "m/yr")
    im = show(axs[0, 1], do, do["vel"], SEQ, 0, vmax, f"Modelled: {a.old_label}")
    cbar(fig, im, axs[0, 1], "m/yr")
    im = show(axs[0, 2], dn, dn["vel"], SEQ, 0, vmax, f"Modelled: {a.new_label}")
    cbar(fig, im, axs[0, 2], "m/yr")

    # metrics table (a text view of the same data)
    ax = axs[1, 0]
    ax.axis("off")
    mo, mn = metrics(do, to), metrics(dn, tn)
    ax.text(0.0, 1.0, "Summary", fontsize=11, color=INK, va="top", transform=ax.transAxes)
    ax.text(0.50, 0.93, "old", fontsize=9, color=MUTED, va="top", ha="right", transform=ax.transAxes)
    ax.text(0.98, 0.93, "new", fontsize=9, color=MUTED, va="top", ha="right", transform=ax.transAxes)
    for i, k in enumerate(mo):
        y = 0.83 - 0.105 * i
        ax.text(0.0, y, k, fontsize=10, color=INK2, va="top", transform=ax.transAxes)
        ax.text(0.50, y, mo[k], fontsize=10, color=INK, va="top", ha="right", transform=ax.transAxes)
        ax.text(0.98, y, mn[k], fontsize=10, color=INK, va="top", ha="right", transform=ax.transAxes)

    im = show(axs[1, 1], do, do["vel"] - do["obs"], DIV, -emax, emax, f"Misfit (model - obs): {a.old_label}")
    cbar(fig, im, axs[1, 1], "m/yr")
    im = show(axs[1, 2], dn, dn["vel"] - dn["obs"], DIV, -emax, emax, f"Misfit (model - obs): {a.new_label}")
    cbar(fig, im, axs[1, 2], "m/yr")

    im = show(axs[2, 0], do, do["thk"], SEQ, 0, hmax, f"DA thickness: {a.old_label}")
    cbar(fig, im, axs[2, 0], "m")
    im = show(axs[2, 1], dn, dn["thk"], SEQ, 0, hmax, f"DA thickness: {a.new_label}")
    cbar(fig, im, axs[2, 1], "m")
    im = show(axs[2, 2], dn, dn["thk"] - do["thk"], DIV, -dmax, dmax, "Thickness difference (new - old)")
    cbar(fig, im, axs[2, 2], "m")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print(f"wrote {a.out}")
    print(f"old best trial {to['dir']}: {mo}")
    print(f"new best trial {tn['dir']}: {mn}")


if __name__ == "__main__":
    main()
