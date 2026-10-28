#!/usr/bin/env python3
"""
Figures and numbers for report/report_smb.tex (SMB-phase rework, 2026-10-18..20), computed from
GROUP_UNIL/ (before) and results/*/smb_G_bands_geod/ (after). Nothing in the report is typed by hand.

    python report/make_assets_smb.py      # -> report/figures_smb/*.pdf, report/tables/macros_smb.tex
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures_smb")
os.makedirs(FIG, exist_ok=True)
NEW = "G_bands_geod"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
TW = 6.3
M = {}      # macros


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    if os.environ.get("PNG"):                      # preview only (no PDF rasteriser on the node)
        fig.savefig(os.path.join(os.environ["PNG"], name.replace(".pdf", ".png")), dpi=130)
    plt.close(fig); print("fig", name)


def law(z):
    return np.minimum(0.01 * (z - 2350.0), 5.0)


def old_path(e, g):
    return os.path.join(CX, "GROUP_UNIL", e, f"{e}_{g}_method01.nc")


def new_path(e, g, tag=NEW):
    return os.path.join(CX, "results", f"{e}_{g}_cnn", f"smb_{tag}", "export", f"{e}_{g}_method01.nc")


def runs(e, g, which):
    """Family members' smb_inference output dirs: 'old' = first run, 'new' = rework."""
    base = os.path.join(CX, "results", f"{e}_{g}_cnn")
    pat = os.path.join(base, "family" if which == "old" else f"smb_{NEW}", "slid_*", "smb_inference")
    return sorted(d for d in glob.glob(pat) if "failed" not in d and os.path.exists(os.path.join(d, "loss_history.npy")))


def all_pairs():
    out = []
    for f in sorted(glob.glob(os.path.join(CX, "results", "*_cnn", f"smb_{NEW}", "export", "*_method01.nc"))):
        n = os.path.basename(f).replace("_method01.nc", "")
        e, g = n.split("_")
        if os.path.exists(old_path(e, g)):
            out.append((e, g))
    return out


def curves(e, g, which):
    L = []
    for d in runs(e, g, which):
        l = np.load(os.path.join(d, "loss_history.npy"))
        if l.size:
            L.append(l)
    return L


def rel(l):
    return l / l[0]


# ── 1. what each change does on S01 (family-level profile RMSE vs the prescribed law) ───────────────────────────
def s01_profile_rmse(path, src):
    """check_s01.py definition: profile nodes inside the 1-99 % elevation range of the glacier."""
    o, s = xr.open_dataset(path), xr.open_dataset(src)
    ice = (np.nan_to_num(s.ICEMASK.values) > 0.5) & (np.nan_to_num(s.THK.values) > 0)
    zlo, zhi = np.nanpercentile(s.DEM.values[ice], [1, 99])
    z = o.z.values
    sel = (z >= zlo) & (z <= zhi) & np.isfinite(o.SMB_PROFILE.values)
    return float(np.sqrt(np.mean((o.SMB_PROFILE.values[sel] - law(z[sel])) ** 2)))


def ablation():
    src = os.path.join(CX, "input_data", "EXP01", "EXP01_S01.nc")
    steps = [("first run", old_path("EXP01", "S01")),
             ("fixed\ntime step", new_path("EXP01", "S01", "A_rampfx")),
             ("+ border\nfix", new_path("EXP01", "S01", "v2")),
             ("+ band\nmisfit", new_path("EXP01", "S01", "E_bands_fam")),
             ("+ geodetic\nterm (final)", new_path("EXP01", "S01", "G_bands_geod"))]
    vals = [(n, s01_profile_rmse(p, src)) for n, p in steps if os.path.exists(p)]
    fig, ax = plt.subplots(figsize=(TW * 0.62, 2.1))
    cols = [MUTED] + [BLUE] * (len(vals) - 2) + [ORANGE]
    ax.bar(range(len(vals)), [v for _, v in vals], color=cols, width=0.62)
    for i, (_, v) in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([n for n, _ in vals], fontsize=6.5)
    ax.set_ylabel("S01 profile RMSE\n(m i.e. yr$^{-1}$)"); ax.grid(axis="x", visible=False)
    save(fig, "fig_ablation.pdf")
    M["ablFirst"], M["ablFinal"] = f"{vals[0][1]:.2f}", f"{vals[-1][1]:.2f}"


# ── 2. SMB-inference loss before / after ────────────────────────────────────────────────────────────────────────
def loss_figures(pairs):
    # (a) S01, the four glacier types: example loss curves, normalised to the first iteration
    sel = [("EXP01", "S01", "S01 (synthetic, steady)"), ("EXP01", "S02", "S02 (synthetic, transient)"),
           ("EXP01", "G03", "G03 (real)"), ("EXP01", "G05", "G05 Aletsch (real)")]
    fig, axs = plt.subplots(1, 4, figsize=(TW * 1.12, 1.9), sharey=True)
    for ax, (e, g, t) in zip(axs, sel):
        for which, c in (("old", MUTED), ("new", ORANGE)):
            for k, l in enumerate(curves(e, g, which)):
                ax.plot(np.arange(1, l.size + 1), rel(l), color=c, lw=1.1, alpha=0.9,
                        label=("first run" if which == "old" else "reworked") if k == 0 else None)
        ax.set_title(t, fontsize=7.5); ax.set_xlabel("L-BFGS iteration")
    axs[0].set_ylabel("loss / first iteration"); axs[0].legend(frameon=False, loc="lower right")
    save(fig, "fig_loss_examples.pdf")

    # (b) all experiments: loss reduction and number of iterations, per family member
    red = {"old": [], "new": []}; its = {"old": [], "new": []}
    for e, g in pairs:
        for which in red:
            for l in curves(e, g, which):
                red[which].append(100 * (1 - l[-1] / l[0])); its[which].append(l.size)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TW, 2.0))
    bins = np.linspace(0, 100, 26)
    for which, c, lab in (("old", MUTED, "first run"), ("new", ORANGE, "reworked")):
        a1.hist(red[which], bins=bins, color=c, alpha=0.8, label=lab)
    a1.set_xlabel("loss reduction over the run (%)"); a1.set_ylabel("family members"); a1.legend(frameon=False)
    ib = np.arange(0.5, 61.5, 2)
    for which, c in (("old", MUTED), ("new", ORANGE)):
        a2.hist(its[which], bins=ib, color=c, alpha=0.8)
    a2.set_xlabel("L-BFGS iterations until convergence"); a2.set_ylabel("family members")
    save(fig, "fig_loss_all.pdf")
    for which in red:
        M[f"redMed{which.capitalize()}"] = f"{np.median(red[which]):.0f}"
        M[f"itsMed{which.capitalize()}"] = f"{np.median(its[which]):.0f}"
        M[f"nMem{which.capitalize()}"] = f"{len(red[which])}"
    M["redBelowFiveOld"] = f"{100 * np.mean(np.array(red['old']) < 5):.0f}"
    M["redBelowFiveNew"] = f"{100 * np.mean(np.array(red['new']) < 5):.0f}"

    # (c) common yardstick: end-of-window thickness error per cell (observed > 1 m), not the optimised quantity
    fig, ax = plt.subplots(figsize=(TW * 0.5, 2.5))
    xs, ys, cs = [], [], []
    for e, g in pairs:
        r = {}
        for which in ("old", "new"):
            v = []
            for d in runs(e, g, which):
                try:
                    h, o = np.load(os.path.join(d, "thk_optimized.npy")), np.load(os.path.join(d, "thk_observed.npy"))
                except FileNotFoundError:
                    continue
                m = o > 1
                v.append(np.sqrt(np.mean((h[m] - o[m]) ** 2)))
            r[which] = np.mean(v) if v else np.nan
        xs.append(r["old"]); ys.append(r["new"]); cs.append(BLUE if g.startswith("S") else ORANGE)
    xs, ys = np.array(xs), np.array(ys)
    lim = [min(np.nanmin(xs), np.nanmin(ys)) * 0.8, max(np.nanmax(xs), np.nanmax(ys)) * 1.25]
    ax.plot(lim, lim, color=MUTED, lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(xs, ys, s=9, c=cs, lw=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("first run: thickness RMSE at t$_2$ (m)"); ax.set_ylabel("reworked (m)")
    ax.scatter([], [], s=9, c=BLUE, label="synthetic"); ax.scatter([], [], s=9, c=ORANGE, label="real")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_thk_rmse.pdf")
    ok = np.isfinite(xs) & np.isfinite(ys)
    M["thkBetter"] = f"{int(np.sum(ys[ok] < xs[ok]))}"; M["thkN"] = f"{int(ok.sum())}"
    M["thkMedOld"], M["thkMedNew"] = f"{np.median(xs[ok]):.1f}", f"{np.median(ys[ok]):.1f}"


# ── 3. S01 against the prescribed law ───────────────────────────────────────────────────────────────────────────
def s01_figures():
    zf = np.arange(1669, 2894, 5.0)
    fig, (ax, ae) = plt.subplots(1, 2, figsize=(TW * 0.85, 2.7), sharey=True)
    ax.plot(law(zf), zf, color=INK, lw=1.3, ls=(0, (4, 3)), zorder=10, label="prescribed law")
    for path, c, lab in ((old_path("EXP01", "S01"), MUTED, "first run"), (new_path("EXP01", "S01"), ORANGE, "reworked")):
        o = xr.open_dataset(path)
        p = np.interp(zf, o.z.values, o.SMB_PROFILE.values)
        ax.plot(p, zf, color=c, lw=1.8, label=lab); ae.plot(p - law(zf), zf, color=c, lw=1.8)
    ae.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m)"); ax.legend(frameon=False, loc="lower right")
    ae.set_xlabel("inferred $-$ prescribed (m i.e. yr$^{-1}$)")
    save(fig, "fig_s01_profile.pdf")

    exps = ["EXP%02d" % i for i in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]
    rows = []
    for e in exps:
        src = os.path.join(CX, "input_data", e, f"{e}_S01.nc")
        if os.path.exists(old_path(e, "S01")) and os.path.exists(new_path(e, "S01")):
            rows.append((e, s01_profile_rmse(old_path(e, "S01"), src), s01_profile_rmse(new_path(e, "S01"), src)))
    fig, axs = plt.subplots(1, 2, figsize=(TW, 2.1), gridspec_kw={"width_ratios": [4, 1]})
    x = np.arange(len(rows) - 1)
    main = [r for r in rows if r[0] != "EXP15"]
    ax = axs[0]
    ax.bar(x - 0.2, [r[1] for r in main], 0.4, color=MUTED, label="first run")
    ax.bar(x + 0.2, [r[2] for r in main], 0.4, color=ORANGE, label="reworked")
    ax.set_xticks(x); ax.set_xticklabels([r[0][3:] for r in main]); ax.set_xlabel("experiment (S01)")
    ax.set_ylabel("profile RMSE\n(m i.e. yr$^{-1}$)"); ax.set_ylim(0, 2.7); ax.legend(frameon=False, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
    r15 = [r for r in rows if r[0] == "EXP15"]
    if r15:
        ax = axs[1]
        ax.bar([-0.2], [r15[0][1]], 0.4, color=MUTED); ax.bar([0.2], [r15[0][2]], 0.4, color=ORANGE)
        ax.set_xticks([0]); ax.set_xticklabels(["15"]); ax.set_xlabel("(100 m grid)"); ax.grid(axis="x", visible=False)
        for xx, v in ((-0.2, r15[0][1]), (0.2, r15[0][2])):
            ax.text(xx, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.5)
    save(fig, "fig_s01_all.pdf")
    better = [r for r in rows if r[2] < r[1] - 0.005]
    M["sOneBetter"], M["sOneN"] = f"{len(better)}", f"{len(rows)}"
    r1 = [r for r in rows if r[0] == "EXP01"][0]
    M["sOneOld"], M["sOneNew"] = f"{r1[1]:.2f}", f"{r1[2]:.2f}"
    worse = [r for r in rows if r[2] > r[1] + 0.005]
    M["sOneWorse"] = ", ".join(r[0] for r in worse) or "none"


# ── 4. real glaciers and all experiments: how far the result moved ─────────────────────────────────────────────
def real_profiles():
    gids = ["G01", "G02", "G03", "G04", "G05", "G06"]
    fig, axs = plt.subplots(1, 6, figsize=(TW * 1.15, 2.3), sharex=False)
    for ax, g in zip(axs, gids):
        for path, c, lab in ((old_path("EXP01", g), MUTED, "first run"), (new_path("EXP01", g), ORANGE, "reworked")):
            if not os.path.exists(path):
                continue
            o = xr.open_dataset(path)
            ok = np.isfinite(o.SMB_PROFILE.values)
            ax.plot(o.SMB_PROFILE.values[ok], o.z.values[ok], color=c, lw=1.5, label=lab)
        ax.axvline(0, color=MUTED, lw=0.6); ax.set_title(g, fontsize=8)
        ax.set_xlabel("SMB (m i.e. yr$^{-1}$)", fontsize=7)
    axs[0].set_ylabel("elevation (m)"); axs[0].legend(frameon=False, loc="lower right", fontsize=6)
    save(fig, "fig_real_profiles.pdf")


def shifts(pairs):
    rows = []
    for e, g in pairs:
        o, n = xr.open_dataset(old_path(e, g)), xr.open_dataset(new_path(e, g))
        so, sn = float(np.nanmean(o.SMB.values)), float(np.nanmean(n.SMB.values))
        eo = np.mean(json.loads(o.attrs.get("ELA_admissible_m", "[]") or "[]") or [np.nan])
        en = np.mean(json.loads(n.attrs.get("ELA_admissible_m", "[]") or "[]") or [np.nan])
        uo, un = float(np.nanmean(o.UNCT_SMB.values)), float(np.nanmean(n.UNCT_SMB.values))
        rows.append((e, g, so, sn, eo, en, uo, un))
    order = ["G01", "G02", "G03", "G04", "G05", "G06", "S01", "S02"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TW, 2.3))
    rng = np.random.default_rng(0)
    for r in rows:
        x = order.index(r[1]) + rng.uniform(-0.18, 0.18)
        c = BLUE if r[1].startswith("S") else ORANGE
        a1.scatter(x, r[3] - r[2], s=8, c=c, lw=0)
        if np.isfinite(r[4]) and np.isfinite(r[5]):
            a2.scatter(x, r[5] - r[4], s=8, c=c, lw=0)
    for a, lab, lt in ((a1, "$\\Delta$ glacier-wide SMB (m i.e. yr$^{-1}$)", 0.5), (a2, "$\\Delta$ ELA (m)", 50)):
        a.set_yscale("symlog", linthresh=lt); a.axhline(0, color=MUTED, lw=0.7)
        a.set_xticks(range(len(order))); a.set_xticklabels(order, fontsize=6.5); a.set_ylabel(lab)
        a.grid(axis="x", visible=False)
    save(fig, "fig_shift.pdf")
    arr = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows], float)
    M["gwShiftMed"] = f"{np.nanmedian(np.abs(arr[:, 1] - arr[:, 0])):.2f}"
    M["gwShiftMax"] = f"{np.nanmax(np.abs(arr[:, 1] - arr[:, 0])):.2f}"
    M["elaShiftMed"] = f"{np.nanmedian(np.abs(arr[:, 3] - arr[:, 2])):.0f}"
    M["uncOld"], M["uncNew"] = f"{np.nanmedian(arr[:, 4]):.2f}", f"{np.nanmedian(arr[:, 5]):.2f}"
    M["nPairs"] = f"{len(rows)}"
    real = [r for r in rows if r[1].startswith("G")]
    M["realGwShiftMed"] = f"{np.nanmedian(np.abs(np.array([r[3] - r[2] for r in real]))):.2f}"


def main():
    pairs = all_pairs()
    print(len(pairs), "experiment-glacier pairs with both versions")
    ablation(); loss_figures(pairs); s01_figures(); real_profiles(); shifts(pairs)
    with open(os.path.join(HERE, "tables", "macros_smb.tex"), "w") as f:
        for k, v in M.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
