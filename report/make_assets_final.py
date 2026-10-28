#!/usr/bin/env python3
"""
Figures and numbers for report/report_final.tex, the short results report: CNN emulator only, data assimilation of the
first run, SMB phase of the final configuration (band misfit + geodetic term, results/*/smb_G_bands_geod/export).
The first run (GROUP_UNIL/) appears only as a reference in the loss and S01 figures. The Setup figures on the
geodetic term (geodetic()) also read S01 reruns without it (smb_E_bands_fam, smb_fx_rmse_fam, smb_H_cells_geod), the
law evaluated on every member (smb_G_truth_fam), EXP01 S02/G03 without it (smb_E_bands) and tools/geod_scan.py output. Plotting helpers and the
profile/sensitivity figures are reused from make_assets.py with its `export` pointed at the final exports.

    python report/make_assets_final.py   # -> report/figures_final/*.pdf, report/tables_final/macros_final.tex,
                                         #    report/stats_final.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_assets_smb as ms    # noqa: E402  (first: make_assets restores its own rcParams on import)
import make_assets as ma        # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import xarray as xr              # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

CX = ma.CX
FIG, TAB = os.path.join(HERE, "figures_final"), os.path.join(HERE, "tables_final")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
ma.FIG, ma.TAB = FIG, TAB                   # make_assets figures/tables of this report go to the _final folders
INK, INK2, MUTED, GRID, CAT = ma.INK, ma.INK2, ma.MUTED, ma.GRID, ma.CAT
FINAL, FIRST = CAT[0], MUTED                # final run / first run, in every figure
TW = ma.TW
M = {}                                      # LaTeX macros
STATS = ma.STATS


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    if os.environ.get("PNG"):                      # preview only (no PDF rasteriser on the node)
        fig.savefig(os.path.join(os.environ["PNG"], name.replace(".pdf", ".png")), dpi=130)
    plt.close(fig); print("fig", name)


def export_final(exp, gid, arch=False):
    p = ms.new_path(exp, gid)
    return xr.open_dataset(p) if not arch and os.path.exists(p) else None


ma.export, ma.save = export_final, save


def spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ── 1. SMB-inference loss, first run against final (absolute values) ─────────────────────────────────────────────
def loss_figure(pairs):
    sel = [("S01", "S01 (synthetic, steady)"), ("S02", "S02 (synthetic, transient)"), ("G03", "G03 (real)"),
           ("G05", "G05 (real)")]
    ticks = {"S01": [10, 20, 30, 40], "S02": [5, 10, 20, 50, 100], "G03": [10, 15, 20], "G05": [3, 5, 10]}
    fig, axs = plt.subplots(1, 4, figsize=(TW, 2.1))
    for ax, (g, t) in zip(axs, sel):
        for which, c, lab in (("old", FIRST, "first run"), ("new", FINAL, "final")):
            for k, l in enumerate(ms.curves("EXP01", g, which)):
                ax.plot(np.arange(1, l.size + 1), l, color=c, lw=1.1, label=lab if k == 0 else None)
        ax.set_yscale("log")
        ax.set_yticks(ticks[g]); ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(t, loc="left", fontsize=7.5); ax.set_xlabel("L-BFGS iteration")
        ax.grid(axis="y", color=GRID, lw=0.5, which="major"); ax.set_axisbelow(True); spines(ax)
    axs[0].set_ylabel("loss (log scale)")
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    save(fig, "fig_loss.pdf")

    red, its, end = {"old": [], "new": []}, {"old": [], "new": []}, {"old": [], "new": []}
    for e, g in pairs:
        for which in red:
            for l in ms.curves(e, g, which):
                red[which].append(100 * (1 - l[-1] / l[0])); its[which].append(l.size)
    for which, k in (("old", "First"), ("new", "Final")):
        M[f"redMed{k}"] = f"{np.median(red[which]):.0f}"
        M[f"itsMed{k}"] = f"{np.median(its[which]):.0f}"
        M[f"nMem{k}"] = f"{len(red[which])}"
    M["redBelowFiveFirst"] = f"{100 * np.mean(np.array(red['old']) < 5):.0f}"
    M["redMinFinal"] = f"{np.min(red['new']):.0f}"
    for g in ("S01", "S02", "G03", "G05"):
        for which, k in (("old", "First"), ("new", "Final")):
            L = ms.curves("EXP01", g, which)
            M[f"loss{g}{k}A"] = f"{np.mean([l[0] for l in L]):.1f}"
            M[f"loss{g}{k}B"] = f"{np.mean([l[-1] for l in L]):.1f}"


# ── 2. S01 against the prescribed law ────────────────────────────────────────────────────────────────────────────
S01_EXPS = ["EXP%02d" % i for i in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]


def s01_figure():
    src = ma.native("EXP01", "S01")
    zr = np.nanpercentile(src.DEM.values[ma.ice_of(src, "S01")], [1, 99])
    zz = np.linspace(zr[0], zr[1], 200)
    o, oo = export_final("EXP01", "S01"), xr.open_dataset(ms.old_path("EXP01", "S01"))
    fig = plt.figure(figsize=(TW, 5.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.45, 1], hspace=0.42, wspace=0.5)
    ax = fig.add_subplot(gs[0, :3])
    for k in range(o.sizes["member"]):
        ax.plot(o.MEMBER_SMB_PROFILE.values[k], o.z.values, color=FINAL, lw=0.6, alpha=0.45)
    ax.plot(oo.SMB_PROFILE.values, oo.z.values, color=FIRST, lw=1.5, ls=(0, (3, 2)), label="first run (family mean)")
    ax.plot(o.SMB_PROFILE.values, o.z.values, color=FINAL, lw=2, label="final (family mean; thin: members)")
    ax.plot(ms.law(zz), zz, color=INK, lw=1.2, label="prescribed SMB law")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("(a) EXP01", loc="left"); ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax.set_ylim(*zr); ax.set_xlim(-8, 6); spines(ax)

    ax = fig.add_subplot(gs[0, 3:], sharey=ax)
    groups = {"THK (EXP03--07)": (range(3, 8), CAT[1]), "VEL (EXP08--13)": (range(8, 14), CAT[2]),
              "RES (EXP14--15)": (range(14, 16), CAT[6])}
    for lab, (rng, c) in groups.items():
        for i, n in enumerate(rng):
            oe = export_final(f"EXP{n:02d}", "S01")
            if oe is not None:
                ax.plot(oe.SMB_PROFILE.values, oe.z.values, color=c, lw=0.9, alpha=0.85, label=lab if i == 0 else None)
    ax.plot(o.SMB_PROFILE.values, o.z.values, color=FINAL, lw=2, label="EXP01")
    ax.plot(ms.law(zz), zz, color=INK, lw=1.2, label="prescribed law")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_title("(b) all S01 experiments, final", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6); ax.set_xlim(-8, 6); ax.tick_params(labelleft=False)
    spines(ax)

    rows = []
    for e in S01_EXPS:
        s = os.path.join(CX, "input_data", e, f"{e}_S01.nc")
        if os.path.exists(ms.old_path(e, "S01")) and os.path.exists(ms.new_path(e, "S01")):
            rows.append((e, ms.s01_profile_rmse(ms.old_path(e, "S01"), s), ms.s01_profile_rmse(ms.new_path(e, "S01"), s)))
    main = [r for r in rows if r[0] != "EXP15"]
    ax = fig.add_subplot(gs[1, :4])
    x = np.arange(len(main))
    ax.bar(x - 0.2, [r[1] for r in main], 0.38, color=FIRST, label="first run")
    ax.bar(x + 0.2, [r[2] for r in main], 0.38, color=FINAL, label="final")
    ax.set_xticks(x); ax.set_xticklabels([r[0][3:] for r in main]); ax.set_xlabel("experiment")
    ax.set_ylabel("profile RMSE\n(m i.e. yr$^{-1}$)"); ax.set_ylim(0, 2.6)
    ax.set_title("(c) error of the family-mean profile against the law", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left"); spines(ax)
    r15 = [r for r in rows if r[0] == "EXP15"]
    if r15:
        ax = fig.add_subplot(gs[1, 4])
        ax.bar([-0.2], [r15[0][1]], 0.38, color=FIRST); ax.bar([0.2], [r15[0][2]], 0.38, color=FINAL)
        for xx, v in ((-0.2, r15[0][1]), (0.2, r15[0][2])):
            ax.text(xx, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks([0]); ax.set_xticklabels(["15"]); ax.set_xlabel("(100 m grid)"); ax.set_ylim(0, 31); spines(ax)
    save(fig, "fig_s01.pdf")

    d = {r[0]: r for r in rows}
    M["sOneFirst"], M["sOneFinal"] = f"{d['EXP01'][1]:.2f}", f"{d['EXP01'][2]:.2f}"
    better = [r for r in rows if r[2] < r[1] - 0.005]
    worse = [r for r in rows if r[2] > r[1] + 0.005]
    M["sOneBetter"], M["sOneN"] = f"{len(better)}", f"{len(rows)}"
    M["sOneWorse"] = ", ".join(r[0] for r in worse) or "none"
    for e in ("EXP05", "EXP09", "EXP12", "EXP13", "EXP14", "EXP15"):
        if e in d:
            k = "sOne" + e.replace("EXP", "E")
            n = 1 if e == "EXP15" else 2
            M[k + "First"], M[k + "Final"] = f"{d[e][1]:.{n}f}", f"{d[e][2]:.{n}f}"
    regular = [r[2] for r in rows if r[0] not in ("EXP05", "EXP15")]
    M["sOneRegMin"], M["sOneRegMax"] = f"{min(regular):.2f}", f"{max(regular):.2f}"
    STATS["s01_profile_rmse"] = {r[0]: {"first": r[1], "final": r[2]} for r in rows}


# ── 3. synthetic metrics, S02 thickness ──────────────────────────────────────────────────────────────────────────
SYN_EXPS = ma.SYN_EXPS


def synthetic():
    res = {g: {e: ma.synth_metrics(e, g) for e in SYN_EXPS} for g in ("S01", "S02")}
    STATS["synthetic"] = res
    first = {}
    ma.export = lambda exp, gid, arch=False: (xr.open_dataset(ms.old_path(exp, gid)) if os.path.exists(ms.old_path(exp, gid)) else None)
    try:
        first = {g: ma.synth_metrics("EXP01", g) for g in ("S01", "S02")}
    finally:
        ma.export = export_final
    STATS["synthetic_first"] = first
    a, f = res["S01"]["EXP01"], first["S01"]
    M.update(sOnePixFinal=f"{a['smb_rmse']:.2f}", sOnePixFirst=f"{f['smb_rmse']:.2f}", sOneRtwo=f"{a['smb_r2']:.2f}",
             sOneRtwoFirst=f"{f['smb_r2']:.2f}",
             sOneElaLo=f"{min(a['ela']):.0f}", sOneElaHi=f"{max(a['ela']):.0f}",
             sOneElaFirstLo=f"{min(f['ela']):.0f}", sOneElaFirstHi=f"{max(f['ela']):.0f}",
             sOneGw=f"{a['glacierwide_smb']:+.2f}", sOneGwTrue=f"{a['law_glacierwide']:+.2f}", sOneGwFirst=f"{f['glacierwide_smb']:+.2f}",
             sOneDhMod=f"{a['dhdt_model']:+.2f}", sOneDhModFirst=f"{f['dhdt_model']:+.2f}",
             sOneFdivRmse=f"{a['fdiv_rmse']:.1f}", sOneFdivRmseFirst=f"{f['fdiv_rmse']:.1f}",
             sOneThk=f"{a['thk_mean']:.0f}", sOneThkTrue=f"{a['thk_true']:.0f}")
    M["sTwoGwFirst"] = f"{first['S02']['glacierwide_smb']:+.2f}"
    b = res["S02"]["EXP01"]
    M.update(sTwoGw=f"{b['glacierwide_smb']:+.2f}", sTwoDhMod=f"{b['dhdt_model']:+.2f}", sTwoDhObs=f"{b['dhdt_obs']:+.2f}",
             sTwoThkBias=f"{b['thk_bias']:+.0f}", sTwoThkRmse=f"{b['thk_rmse']:.0f}", sTwoThkR=f"{b['thk_r']:.2f}",
             sTwoEla=f"{np.mean(b['ela']):.0f}")
    gw = {e: r["glacierwide_smb"] for e, r in res["S02"].items() if r}
    reg = [v for e, v in gw.items() if e not in ("EXP14",)]
    M["sTwoGwMin"], M["sTwoGwMax"] = f"{min(reg):+.2f}", f"{max(reg):+.2f}"
    M["sTwoGwFourteen"] = f"{gw.get('EXP14', np.nan):+.2f}"
    for e in ("EXP02", "EXP13", "EXP14"):
        r = res["S02"].get(e)
        if r and r.get("ela"):
            M["sTwoEla" + e.replace("EXP", "E")] = f"{np.mean(r['ela']):.0f}"
    if res["S02"].get("EXP02"):
        M["sTwoThkRmseRaw"] = f"{res['S02']['EXP02']['thk_rmse']:.0f}"

    # S02 thickness (DA of the first run; the final SMB phase does not change it)
    src = ma.native("EXP01", "S02"); o = export_final("EXP01", "S02"); ice = ma.ice_of(src, "S02")
    thk = ma.on_grid(o, src, "THK"); tt = src.THK.values.astype(float)
    da = xr.open_dataset(os.path.join(CX, "data", "EXP01_S02_da.nc"))
    fig = plt.figure(figsize=(TW, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.32, 1.25], wspace=0.22)
    axs = [fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4)]
    im = ma.imshow(axs[0], src, np.where(ice, tt, np.nan), ma.SEQ_ORANGE, 0, 700); axs[0].set_title("(a) true $H$", loc="left")
    ma.cbar(fig, im, axs[0], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    im = ma.imshow(axs[1], src, np.where(ice, thk, np.nan), ma.SEQ_ORANGE, 0, 700); axs[1].set_title("(b) inferred $H$", loc="left")
    ma.cbar(fig, im, axs[1], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    im = ma.imshow(axs[2], src, np.where(ice, thk - tt, np.nan), ma.DIV, -200, 200); axs[2].set_title("(c) inferred $-$ true", loc="left")
    ma.cbar(fig, im, axs[2], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    for name, c, lab in (("thkobs", INK, "GPR assimilated"), ("thkobs_val", CAT[1], "GPR withheld")):
        iy, ix = np.nonzero(np.isfinite(da[name].values))
        axs[1].scatter(da.x.values[ix], da.y.values[iy], s=3, c=c, linewidths=0, label=lab)
    axs[1].legend(frameon=False, loc="lower left", fontsize=6, markerscale=2, handletextpad=0.1, borderaxespad=0.1)
    ax = axs[3]
    m = ice & np.isfinite(thk)
    ax.scatter(tt[m], thk[m], s=1, c=FINAL, alpha=0.35, linewidths=0, rasterized=True)
    ax.plot([0, 760], [0, 760], color=INK, lw=0.8)
    ax.set_xlim(0, 760); ax.set_ylim(0, 760); ax.set_aspect("equal")
    ax.set_xlabel("true $H$ (m)"); ax.set_ylabel("inferred $H$ (m)")
    ax.set_title(f"(d) bias {b['thk_bias']:+.0f} m, RMSE {b['thk_rmse']:.0f} m", loc="left"); spines(ax)
    save(fig, "fig_s02_thickness.pdf")


# ── 4. real glaciers, EXP01 ──────────────────────────────────────────────────────────────────────────────────────
REAL = ma.REAL


def real_exp01():
    res = {g: ma.real_metrics("EXP01", g) for g in REAL}
    STATS["real_exp01"] = {g: {k: v for k, v in r.items() if k != "profile"} for g, r in res.items()}
    fig, axs = plt.subplots(2, 3, figsize=(TW, 4.6))
    for ax, g in zip(axs.ravel(), REAL):
        o, src, r = export_final("EXP01", g), ma.native("EXP01", g), res[g]
        for k in range(o.sizes["member"]):
            ax.plot(o.MEMBER_SMB_PROFILE.values[k], o.z.values, color=FINAL, lw=0.5, alpha=0.45)
        ax.plot(o.SMB_PROFILE.values, o.z.values, color=FINAL, lw=1.8, label="family mean")
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        zs = src.DEM.values[np.isfinite(ma.on_grid(o, src, "SMB"))]
        ax.set_ylim(np.nanpercentile(zs, 0.5), np.nanpercentile(zs, 99.5))
        gw = f"{r['smb']:+.2f}".replace("-", "\u2212")
        ax.set_title(f"{g}: $\\overline{{\\rm SMB}}$ {gw}, ELA {r['ela']:.0f} m", loc="left", fontsize=7.5)
        ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); spines(ax)
    axs[0, 0].set_ylabel("elevation (m a.s.l.)"); axs[1, 0].set_ylabel("elevation (m a.s.l.)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(h + [plt.Line2D([], [], color=FINAL, lw=0.5, alpha=0.6)], l + ["family members"], loc="lower center",
               ncol=2, frameon=False)
    save(fig, "fig_real_profiles.pdf")
    for g, r in res.items():
        M.update({f"gw{g}": f"{r['smb']:+.2f}", f"ela{g}": f"{r['ela']:.0f}", f"dhMod{g}": f"{r['dhdt_model']:+.2f}",
                  f"dhObs{g}": f"{r['dhdt_obs']:+.2f}", f"fdiv{g}": f"{r['fdiv']:+.2f}", f"unc{g}": f"{r['unct_smb']:.2f}",
                  f"thk{g}": f"{r['thk']:.0f}"})
        if r.get("gpr_test") is not None and np.isfinite(r["gpr_test"]):
            M[f"gprTest{g}"] = f"{r['gpr_test']:.0f}"
    M["famSpreadMax"] = f"{max(float(np.nanmean(export_final('EXP01', g).COMP_FAMILY_SPREAD_SMB.values)) for g in REAL):.2f}"
    clos = np.array([abs(r["dhdt_model"] - r["dhdt_obs"]) for r in res.values()])
    M["closMax"] = f"{clos.max():.2f}"
    M["uncMin"], M["uncMax"] = f"{min(r['unct_smb'] for r in res.values()):.1f}", f"{max(r['unct_smb'] for r in res.values()):.1f}"

    # G05 maps
    g = "G05"; o = export_final("EXP01", g); src = ma.native("EXP01", g)
    smb, fdiv, thk, un = (ma.on_grid(o, src, v) for v in ("SMB", "FDIV", "THK", "UNCT_SMB"))
    fig, axs = plt.subplots(1, 4, figsize=(TW, 2.9))
    st = 4
    for ax, a, cm, vr, t, lab in ((axs[0], thk, ma.SEQ_ORANGE, (0, 800), "(a) thickness $H$", "m"),
                                  (axs[1], smb, ma.DIV, (-8, 8), "(b) SMB", "m i.e. yr$^{-1}$"),
                                  (axs[2], fdiv, ma.DIV, (-8, 8), "(c) FDIV", "m i.e. yr$^{-1}$"),
                                  (axs[3], un, ma.SEQ_BLUE, (0, 5), "(d) $\\sigma_{\\rm SMB}$", "m i.e. yr$^{-1}$")):
        im = ma.imshow(ax, src, a, cm, *vr, step=st); ax.set_title(t, loc="left")
        ma.cbar(fig, im, ax, lab, orientation="horizontal", fraction=0.05, pad=0.03)
    ma.scalebar(axs[0], src, 5)
    save(fig, "fig_g05_maps.pdf")


# ── 5. G01 / G05 under perturbation, raw and global data ─────────────────────────────────────────────────────────
def perturbations():
    exps = ["EXP01"] + [f"EXP{i:02d}" for i in range(3, 16)]
    res = {g: {e: ma.real_metrics(e, g) for e in exps} for g in ("G01", "G05")}
    STATS["real_sensitivity"] = {g: {e: {k: v for k, v in (r or {}).items() if k != "profile"} for e, r in d.items()}
                                 for g, d in res.items()}
    for g, d in res.items():
        ref = d["EXP01"]
        dsmb = [abs(r["smb"] - ref["smb"]) for e, r in d.items() if r and e != "EXP01"]
        dela = {e: r["ela"] - ref["ela"] for e, r in d.items() if r and e != "EXP01"}
        M[f"dSmbMax{g}"] = f"{max(dsmb):.2f}"
        M[f"dElaMax{g}"] = f"{max(abs(v) for v in dela.values()):.0f}"
        M[f"dElaMaxExp{g}"] = max(dela, key=lambda e: abs(dela[e]))
        M[f"dSmbMaxNoRes{g}"] = f"{max(abs(r['smb'] - ref['smb']) for e, r in d.items() if r and e not in ('EXP14', 'EXP15')):.2f}"
        M[f"dElaMaxNoRes{g}"] = f"{max(abs(v) for e, v in dela.items() if e not in ('EXP14', 'EXP15')):.0f}"
        for e in ("EXP14", "EXP15"):
            if d.get(e):
                M[f"gw{g}{e}"], M[f"dh{g}{e}"] = f"{d[e]['smb']:+.2f}", f"{d[e]['dhdt_obs']:+.2f}"
        r2 = ma.real_metrics("EXP02", g)
        M[f"rawGw{g}"], M[f"rawEla{g}"] = f"{r2['smb']:+.2f}", f"{abs(r2['ela'] - ref['ela']):.0f}"
    ma.experiment_profiles()        # fig_s02_profiles.pdf, fig_real_exp_profiles.pdf (final exports)
    pr = STATS["profile_response"]
    for k, v in pr.items():
        g, e = k.split()
        M[f"resp{g}{e.replace('EXP', 'E')}"] = f"{v[1]:.2f}"
    for g in ("S02", "G01", "G05"):
        for lab, rng in (("Thk", range(3, 8)), ("Vel", range(8, 14))):
            v = [pr[f"{g} EXP{n:02d}"][1] for n in rng if f"{g} EXP{n:02d}" in pr]
            M[f"resp{lab}Max{g}"] = f"{max(v):.2f}"
    floor = max(ma.profile_change(f"EXP{n:02d}", "G01") for n in range(3, 8))
    M["repeatFloor"] = f"{max(floor, ma.profile_change('EXP15', 'S02')):.2f}"


GLOB_EXPS = ["EXP16", "EXP17", "EXP18", "EXP19", "EXP20"]


def global_inputs():
    ma.glob_results()               # fig_glob_profiles.pdf (+ tables_final/tab_glob.tex, not used in the report)
    G = STATS["glob"]
    for g in REAL:
        for e in ("EXP01", "EXP02", "EXP16", "EXP17", "EXP18", "EXP19", "EXP20"):
            r = G[g].get(e)
            if r:
                k = e.replace("EXP", "E")
                M[f"gl{g}{k}"] = f"{r['smb']:+.2f}"
                M[f"glEla{g}{k}"] = "--" if not np.isfinite(r["ela"]) else f"{r['ela']:.0f}"
                M[f"glDh{g}{k}"] = f"{r['dhdt_obs']:+.2f}"
                M[f"glThk{g}{k}"] = f"{r['thk']:.0f}"
    raw = [(abs(G[g]["EXP02"]["smb"] - G[g]["EXP01"]["smb"]), abs(G[g]["EXP02"]["ela"] - G[g]["EXP01"]["ela"])) for g in REAL]
    M["rawGwMax"], M["rawElaMax"] = f"{max(r[0] for r in raw):.2f}", f"{max(r[1] for r in raw):.0f}"
    # runs whose inputs are identical for this method: EXP17-19 everywhere; EXP16 = EXP20 where there is no GPR
    groups = [[G[g][e] for e in ("EXP17", "EXP18", "EXP19")] for g in REAL]
    groups += [[G[g]["EXP16"], G[g]["EXP20"]] for g in ("G01", "G04", "G06")]
    gw = [np.ptp([r["smb"] for r in grp]) for grp in groups]
    el = [np.ptp([r["ela"] for r in grp]) for grp in groups if all(np.isfinite(r["ela"]) for r in grp)]
    th = [np.ptp([r["thk"] for r in grp]) for grp in groups]
    M["idGwMax"], M["idElaMax"], M["idThkMax"] = f"{max(gw):.2f}", f"{max(el):.0f}", f"{max(th):.0f}"
    ratio = [G[g][e]["unct_smb"] / G[g]["EXP01"]["unct_smb"] for g in REAL for e in GLOB_EXPS if G[g].get(e)]
    M["uncRatioMin"], M["uncRatioMax"] = f"{min(ratio):.1f}", f"{max(ratio):.1f}"


# ── 0. Setup: the geodetic term. Mass budget of the forward model, loss along an offset, S01 with and without ───────
GEOD_W, BAND_M, PROF_DZ, PROF_REG, SMB_UPDATE_YR = 100.0, 50.0, 200.0, 0.05, 5.0    # the final configuration
GEOD_GLACIERS = ["S01", "S02", "G01", "G02", "G03", "G04", "G05", "G06"]
NOGEOD = {"S01": "E_bands_fam", "S02": "E_bands", "G03": "E_bands"}     # band misfit, no geodetic term (EXP01)
S01_CONFIGS = [("cells", "fx_rmse_fam"), ("cells + geodetic", "H_cells_geod"),
               ("bands", "E_bands_fam"), ("bands + geodetic", ms.NEW)]    # all with the final forward model
SCAN_MEMBER = "slid_1.8770"                                               # S01 best-velocity member


def _hat(z, v, z0, dz):
    t = (z - z0) / dz
    return (np.clip(1.0 - np.abs(t[..., None] - np.arange(v.size)), 0.0, None) * v).sum(-1)


def member_budget(run, exp, gid):
    """Glacier-wide budget of one member over the model ice mask (m i.e./yr): <b> on the window-start surface (as in the
    geodetic term), modelled and observed mean dH/dt over the window, and eps = <b> - modelled <dH/dt>, the ice the
    forward model loses across the outline (it melts outside the mask) or gains by clipping H at 0."""
    meta = json.load(open(os.path.join(CX, "data", f"{exp}_{gid}_da.json")))
    ig = xr.open_dataset(os.path.join(CX, "data", meta["igm_input"]))
    geo = xr.open_dataset(os.path.join(run, "geology-optimized.nc"))
    us, th = geo.usurf.values.astype(float), geo.thk.values.astype(float)
    if geo.y.values[0] > geo.y.values[-1]:
        us, th = us[::-1], th[::-1]
    topg, on, dt = us - th, ig.icemask.values > 0.5, meta["t2"] - meta["t1"]
    H1 = np.clip(ig.usurfstart.values - topg, 0, None) * on
    Hobs = ig.usurfinfer.values - topg
    sd = os.path.join(run, "smb_inference")
    H2 = np.load(os.path.join(sd, "thk_optimized.npy"))
    b = _hat(topg + H1, np.load(os.path.join(sd, "smb_vec.npy")), float(np.load(os.path.join(sd, "z_min.npy"))),
             float(np.load(os.path.join(sd, "dz.npy"))))
    r = dict(b=b[on].mean(), dh_mod=(H2 - H1)[on].mean() / dt,
             dh_obs=(np.where(np.isnan(Hobs), H1, Hobs) - H1)[on].mean() / dt)
    r["eps"] = r["b"] - r["dh_mod"]
    return r


def family_budget(exp, gid, tag):
    base = os.path.join(CX, "results", f"{exp}_{gid}_cnn", f"smb_{tag}")
    rs = [member_budget(d, exp, gid) for d in sorted(glob.glob(os.path.join(base, "slid_*")))
          if os.path.isdir(d) and "failed" not in d and os.path.exists(os.path.join(d, "smb_inference", "smb_vec.npy"))]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def s01_profile_bias(path, src):
    """Mean of (family-mean profile - law) on the nodes ms.s01_profile_rmse scores."""
    o, s = xr.open_dataset(path), xr.open_dataset(src)
    ice = (np.nan_to_num(s.ICEMASK.values) > 0.5) & (np.nan_to_num(s.THK.values) > 0)
    zlo, zhi = np.nanpercentile(s.DEM.values[ice], [1, 99])
    z = o.z.values
    sel = (z >= zlo) & (z <= zhi) & np.isfinite(o.SMB_PROFILE.values)
    return float(np.mean(o.SMB_PROFILE.values[sel] - ms.law(z[sel])))


def _parabola_min(x, y):
    k = int(np.argmin(y)); k = min(max(k, 1), len(x) - 2)
    a, b, _ = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2)
    return -b / (2 * a)


def geodetic():
    src_path = os.path.join(CX, "input_data", "EXP01", "EXP01_S01.nc")
    src = xr.open_dataset(src_path)
    law_b = family_budget("EXP01", "S01", "G_truth_fam")
    bud = {tag: family_budget("EXP01", "S01", tag) for _, tag in S01_CONFIGS}
    eps = {g: family_budget("EXP01", g, ms.NEW) for g in GEOD_GLACIERS}
    nog = {g: family_budget("EXP01", g, t) for g, t in NOGEOD.items()}
    scan = json.load(open(os.path.join(CX, "results", "EXP01_S01_cnn", "geod_scan", SCAN_MEMBER, "scan.json")))["rows"]
    dl = np.array([r["delta"] for r in scan]); J = np.array([r["data"] for r in scan]); G = np.array([r["geodetic"] for r in scan])

    # Figure A: how it works
    fig, axs = plt.subplots(1, 3, figsize=(TW, 2.35), gridspec_kw={"width_ratios": [1.05, 1, 1.25]})
    ax = axs[0]
    cases = [("law", law_b), ("$J$ only", bud["E_bands_fam"]), ("$J$ + geod.", bud[ms.NEW])]
    x = np.arange(len(cases))
    ax.bar(x - 0.19, [c["b"] for _, c in cases], 0.36, color=FINAL, label=r"$\langle b\rangle$")
    ax.bar(x + 0.19, [c["dh_mod"] for _, c in cases], 0.36, color=CAT[1], label=r"model $\langle\partial H/\partial t\rangle$")
    ax.axhline(cases[0][1]["dh_obs"], color=INK, lw=0.9, ls=(0, (3, 2)), label=r"observed $\langle\partial H/\partial t\rangle$")
    for i, (_, c) in enumerate(cases):
        top = max(c["b"], c["dh_mod"], 0) + 0.06
        ax.text(i, top, f"$\\varepsilon$={c['eps']:.2f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in cases], fontsize=6.3)
    ax.set_ylabel("m i.e. yr$^{-1}$"); ax.set_ylim(-1.0, 0.62)
    ax.set_title("(a) S01: glacier-wide budget", loc="left"); ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); spines(ax)

    ax = axs[1]
    ax.plot(dl, J, "-o", color=FINAL, ms=2.5, lw=1.2, label="band misfit $J$")
    ax.plot(dl, G, "-o", color=CAT[2], ms=2.5, lw=1.2, label="geodetic term")
    ax.plot(dl, J + G, "-", color=INK, lw=1.4, label="sum")
    dJ, dT = _parabola_min(dl, J), _parabola_min(dl, J + G)
    for d, c, t in ((dJ, FINAL, "min $J$"), (dT, INK, "min sum")):
        ax.axvline(d, color=c, lw=0.7, ls=(0, (2, 2)))
        ax.text(d - 0.015, 10.8, t, fontsize=5.8, color=c, va="bottom", ha="right", rotation=90)
    ax.set_ylim(0, 14); ax.set_xlabel(r"offset $\delta$ added to $b(z)$ (m i.e. yr$^{-1}$)"); ax.set_ylabel("loss (m)")
    ax.set_title("(b) S01: loss along an offset", loc="left"); ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); spines(ax)

    ax = axs[2]
    x = np.arange(len(GEOD_GLACIERS))
    ax.bar(x, [eps[g]["eps"] for g in GEOD_GLACIERS], 0.6, color=MUTED, label=r"$\varepsilon$ (final run)")
    for i, g in enumerate(GEOD_GLACIERS):
        if g in nog:
            ax.plot(i, nog[g]["b"] - nog[g]["dh_obs"], "D", color=CAT[1], ms=4,
                    label=r"$\langle b\rangle$ $-$ observed, band misfit only" if g == "S01" else None)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(GEOD_GLACIERS, fontsize=6.5); ax.set_ylabel("m i.e. yr$^{-1}$")
    ax.set_ylim(-0.2, 0.8)
    ax.set_title("(c) model budget error, EXP01", loc="left"); ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True); spines(ax)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_geod_mechanism.pdf")

    # Figure B: what it does to S01 (cell / band misfit, with and without the geodetic term)
    zr = np.nanpercentile(src.DEM.values[ma.ice_of(src, "S01")], [1, 99]); zz = np.linspace(zr[0], zr[1], 200)
    res = {}
    for name, tag in S01_CONFIGS:
        p = ms.new_path("EXP01", "S01", tag)
        res[tag] = dict(rmse=ms.s01_profile_rmse(p, src_path), bias=s01_profile_bias(p, src_path),
                        ela=ma.parse_ela(xr.open_dataset(p)), o=xr.open_dataset(p))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TW, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    sty = {"fx_rmse_fam": (MUTED, (0, (3, 2))), "E_bands_fam": (CAT[1], "-"), ms.NEW: (FINAL, "-")}
    for name, tag in S01_CONFIGS:
        if tag in sty:
            o = res[tag]["o"]
            a1.plot(o.SMB_PROFILE.values, o.z.values, color=sty[tag][0], ls=sty[tag][1], lw=1.6 if tag == ms.NEW else 1.2,
                    label=f"{name} (RMSE {res[tag]['rmse']:.2f})")
    a1.plot(ms.law(zz), zz, color=INK, lw=1.1, label="prescribed law")
    a1.axvline(0, color=GRID, lw=0.8, zorder=0)
    a1.set_ylim(*zr); a1.set_xlim(-8, 6); a1.set_xlabel("SMB (m i.e. yr$^{-1}$)"); a1.set_ylabel("elevation (m a.s.l.)")
    a1.set_title("(a) S01, EXP01: family-mean profiles", loc="left"); a1.legend(frameon=False, fontsize=6, loc="lower right")
    spines(a1)
    for j, (lab, pair) in enumerate((("cell misfit", ("fx_rmse_fam", "H_cells_geod")),
                                     ("band misfit", ("E_bands_fam", ms.NEW)))):
        for k, (tag, c) in enumerate(zip(pair, (CAT[1], FINAL))):
            xx = j + (k - 0.5) * 0.38
            a2.bar(xx, res[tag]["rmse"], 0.36, color=c, label=("without" if k == 0 else "with") + " geodetic term" if j == 0 else None)
            a2.text(xx, res[tag]["rmse"] + 0.03, f"{res[tag]['rmse']:.2f}\nbias {res[tag]['bias']:+.2f}", ha="center",
                    fontsize=6, color=INK2)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["cell misfit", "band misfit"]); a2.set_ylim(0, 1.5)
    a2.set_ylabel("profile RMSE (m i.e. yr$^{-1}$)"); a2.set_title("(b) error against the law", loc="left")
    a2.legend(frameon=False, fontsize=6, loc="upper right"); a2.grid(axis="y", color=GRID, lw=0.5); a2.set_axisbelow(True)
    spines(a2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_geod_effect.pdf")

    f2 = lambda v: f"{v:+.2f}"  # noqa: E731
    M.update({"geodW": f"{GEOD_W:g}", "bandM": f"{BAND_M:g}", "profDz": f"{PROF_DZ:g}", "profReg": f"{PROF_REG:g}",
              "smbUpd": f"{SMB_UPDATE_YR:g}",
              "geodLawB": f2(law_b["b"]), "geodLawDh": f2(law_b["dh_mod"]), "geodLawEps": f"{law_b['eps']:.2f}",
              "geodLawNative": f2(float(np.nanmean(ms.law(src.DEM.values)[np.nan_to_num(src.THK.values) > 0]))),
              "geodBandB": f2(bud["E_bands_fam"]["b"]), "geodBandDh": f2(bud["E_bands_fam"]["dh_mod"]),
              "geodFinalB": f2(bud[ms.NEW]["b"]), "geodFinalDh": f2(bud[ms.NEW]["dh_mod"]),
              "geodEpsMin": f"{min(c['eps'] for c in bud.values()):.2f}", "geodEpsMax": f"{max(c['eps'] for c in bud.values()):.2f}",
              "scanJmin": f"{dJ:+.2f}", "scanTmin": f"{abs(dT):.2f}" if abs(dT) < 0.005 else f"{dT:+.2f}", "scanJzero": f"{J[np.argmin(np.abs(dl))]:.2f}",
              "scanJbest": f"{np.interp(dJ, dl, J):.2f}", "scanGstep": f"{GEOD_W * 0.1 ** 2:.0f}",
              "epsRealMax": f"{max(abs(eps[g]['eps']) for g in GEOD_GLACIERS if g.startswith('G') and g != 'G03'):.2f}",
              "epsGthree": f"{eps['G03']['eps']:.2f}", "epsStwo": f"{eps['S02']['eps']:.2f}", "epsSone": f"{eps['S01']['eps']:.2f}",
              "nogGthreeB": f2(nog["G03"]["b"]), "nogGthreeObs": f2(nog["G03"]["dh_obs"]),
              "nogStwoB": f2(nog["S02"]["b"]), "nogStwoObs": f2(nog["S02"]["dh_obs"])})
    for name, tag in S01_CONFIGS:
        k = {"fx_rmse_fam": "Cells", "H_cells_geod": "CellsG", "E_bands_fam": "Bands", ms.NEW: "BandsG"}[tag]
        M[f"abl{k}Rmse"], M[f"abl{k}Bias"], M[f"abl{k}Ela"] = (f"{res[tag]['rmse']:.2f}", f2(res[tag]["bias"]),
                                                                f"{res[tag]['ela']:.0f}")
    for g, t in NOGEOD.items():
        e0, e1 = ma.parse_ela(xr.open_dataset(ms.new_path("EXP01", g, t))), ma.parse_ela(export_final("EXP01", g))
        M[f"nogEla{g}"], M[f"geodEla{g}"] = f"{e0:.0f}", f"{e1:.0f}"
    STATS["geodetic"] = {"law": law_b, "configs": {t: {k: v for k, v in res[t].items() if k != "o"} | bud[t] for _, t in S01_CONFIGS},
                         "eps_exp01": eps, "no_geodetic": nog, "scan": scan, "scan_member": SCAN_MEMBER}


# ── 6. G01, EXP01: 2-D SMB field instead of b(z) (results/EXP01_G01_cnn/smb_F_field_reg3) ──────────────────────
FIELD_TAG, FIELD_REG = "F_field_reg3", 3.0
FIELD_SCAN = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)       # smb_F_reg<r>, member slid_0.0568 only


def _member_misfit(sd):
    """End-of-window thickness misfit of one member: per-cell RMSE and RMS of the 50 m band means (m)."""
    H, obs = np.load(os.path.join(sd, "thk_optimized.npy")), np.load(os.path.join(sd, "thk_observed.npy"))
    z = np.load(os.path.join(sd, "usurf_modeled.npy"))
    on = xr.open_dataset(os.path.join(CX, "data", "EXP01_G01_da.nc")).icemask.values > 0.5
    r = (H - obs)[on]
    k = np.floor((z[on] - z[on].min()) / 50.0).astype(int)
    n = np.bincount(k); mb = np.bincount(k, r) / np.where(n > 0, n, 1)
    return float(np.sqrt(np.mean(r ** 2))), float(np.sqrt((n * mb ** 2).sum() / n.sum())), H - obs, on


def _final_roughness(log):
    for ln in open(log):
        if "final loss=" in ln and "roughness=" in ln:
            return float(ln.split("roughness=")[1])
    return np.nan


def g01_field():
    g, root = "G01", os.path.join(CX, "results", "EXP01_G01_cnn")
    prof_o, field_o = export_final("EXP01", g), xr.open_dataset(ms.new_path("EXP01", g, FIELD_TAG))
    src = ma.native("EXP01", g)
    s_prof, s_field = ma.on_grid(prof_o, src, "SMB"), ma.on_grid(field_o, src, "SMB")
    members = sorted(os.path.basename(d) for d in os.listdir(os.path.join(root, f"smb_{FIELD_TAG}"))
                     if d.startswith("slid_") and os.path.isdir(os.path.join(root, f"smb_{FIELD_TAG}", d)))
    mis = {t: [_member_misfit(os.path.join(root, f"smb_{t}", m, "smb_inference")) for m in members]
           for t in (ms.NEW, FIELD_TAG)}
    best = members[0]                                    # slid_0.0568: lowest velocity misfit of the family

    # L-curve on the best member; the profile run enters at the roughness of its b(z) on the node grid
    scan = [(r, _member_misfit(os.path.join(root, f"smb_F_reg{r:g}", best, "smb_inference"))[0],
             _final_roughness(os.path.join(root, f"smb_F_reg{r:g}", best + ".log"))) for r in FIELD_SCAN]
    prof_rough = next(float(ln.split("roughness ")[1].split()[0]) for ln in open(os.path.join(root, "smb_F_reg3", best + ".log"))
                      if "first-guess node fit" in ln)

    fig, axs = plt.subplots(2, 3, figsize=(TW, 5.0))
    st = 4
    for ax, a, cm, vr, t in ((axs[0, 0], s_prof, ma.DIV, (-5, 5), "(a) SMB, profile $b(z)$"),
                             (axs[0, 1], s_field, ma.DIV, (-5, 5), "(b) SMB, 2-D field"),
                             (axs[0, 2], s_field - s_prof, ma.DIV, (-2, 2), "(c) field $-$ profile")):
        im = ma.imshow(ax, src, a, cm, *vr, step=st); ax.set_title(t, loc="left")
        ma.cbar(fig, im, ax, "m i.e. yr$^{-1}$", orientation="horizontal", fraction=0.05, pad=0.03)
    ma.scalebar(axs[0, 0], src, 5)
    igm = xr.open_dataset(os.path.join(CX, "data", "EXP01_G01_da.nc"))
    ext = [float(igm.x[0]), float(igm.x[-1]), float(igm.y[0]), float(igm.y[-1])]
    for ax, t, lab in ((axs[1, 0], ms.NEW, "(d) $H_{\\rm sim}-H_{\\rm obs}$, profile"),
                       (axs[1, 1], FIELD_TAG, "(e) $H_{\\rm sim}-H_{\\rm obs}$, 2-D field")):
        i = members.index(best)
        res, on = mis[t][i][2], mis[t][i][3]
        im = ax.imshow(np.where(on, res, np.nan), cmap=ma.DIV, vmin=-30, vmax=30, extent=ext, origin="lower",
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(lab, loc="left"); ma.cbar(fig, im, ax, "m", orientation="horizontal", fraction=0.05, pad=0.03)
    ax = axs[1, 2]
    rr, rm, ro = (np.array(v) for v in zip(*scan))
    ax.plot(ro, rm, "-o", color=FINAL, ms=3, lw=1)
    for r, y, x in scan:
        ax.annotate(f"{r:g}", (x, y), textcoords="offset points", xytext=(4, 2), fontsize=6.5,
                    color=INK if r == FIELD_REG else INK2, fontweight="bold" if r == FIELD_REG else "normal")
    k = list(rr).index(FIELD_REG)
    ax.plot(ro[k], rm[k], "o", ms=7, mfc="none", mec=INK, mew=1)
    ax.plot(prof_rough, mis[ms.NEW][members.index(best)][0], "s", color=MUTED, ms=4)
    ax.annotate("profile", (prof_rough, mis[ms.NEW][members.index(best)][0]), textcoords="offset points",
                xytext=(4, -9), fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_xlabel("roughness ((m yr$^{-1}$ km$^{-1}$)$^2$)"); ax.set_ylabel("cell RMSE (m)")
    ax.set_title("(f) regularisation", loc="left"); ax.grid(True, color=GRID, lw=0.5); spines(ax)
    fig.tight_layout()
    save(fig, "fig_g01_field.pdf")

    rp, rf = ma.real_metrics("EXP01", g), None
    ma_export = ma.export
    ma.export = lambda e, gg, arch=False: field_o if (e, gg) == ("EXP01", g) else ma_export(e, gg, arch)
    rf = ma.real_metrics("EXP01", g)
    ma.export = ma_export
    avg = lambda t, j: float(np.mean([m[j] for m in mis[t]]))  # noqa: E731
    M.update({
        "fldReg": f"{FIELD_REG:g}", "fldNodes": "500", "fldN": f"{len(members)}",
        "fldCellProf": f"{avg(ms.NEW, 0):.1f}", "fldCellField": f"{avg(FIELD_TAG, 0):.1f}",
        "fldBandProf": f"{avg(ms.NEW, 1):.2f}", "fldBandField": f"{avg(FIELD_TAG, 1):.2f}",
        "fldGwProf": f"{rp['smb']:+.2f}", "fldGwField": f"{rf['smb']:+.2f}",
        "fldElaProf": f"{rp['ela']:.0f}", "fldElaField": f"{rf['ela']:.0f}",
        "fldUncProf": f"{rp['unct_smb']:.2f}", "fldUncField": f"{rf['unct_smb']:.2f}",
        "fldDhField": f"{rf['dhdt_model']:+.2f}", "fldDhObs": f"{rf['dhdt_obs']:+.2f}",
        "fldDiffMax": f"{np.nanpercentile(np.abs(s_field - s_prof), 95):.1f}",
    })
    STATS["g01_field"] = {"tag": FIELD_TAG, "reg": FIELD_REG, "members": members,
                          "misfit": {t: [m[:2] for m in v] for t, v in mis.items()},
                          "scan": [{"reg": r, "cell_rmse": m, "roughness": o} for r, m, o in scan],
                          "profile_roughness": prof_rough,
                          "profile": {k: v for k, v in rp.items() if k != "profile"},
                          "field": {k: v for k, v in rf.items() if k != "profile"}}


def write_macros(merge=False):
    path = os.path.join(TAB, "macros_final.tex")
    if merge:                                   # keep the values of the steps that were not rerun
        old = {}
        for ln in open(path):
            if ln.startswith("\\expandafter"):
                k = ln.split("v:", 1)[1].split("\\endcsname", 1)[0]
                old[k] = ln.split("\\endcsname{", 1)[1].rstrip().rstrip("}")
        M.update({k: v for k, v in old.items() if k not in M} | {k: v for k, v in M.items()})
    with open(path, "w") as f:     # \V{key} in the report; keys may contain digits
        f.write("\\newcommand{\\V}[1]{\\ifcsname v:#1\\endcsname\\csname v:#1\\endcsname\\else"
                "\\textbf{??#1}\\GenericWarning{}{missing value #1}\\fi}\n")
        for k, v in M.items():
            f.write("\\expandafter\\def\\csname v:%s\\endcsname{%s}\n" % (k, v))


def main():
    os.chdir(CX)
    if len(sys.argv) > 1:                       # e.g. `make_assets_final.py g01_field`: only these steps, merged
        for name in sys.argv[1:]:
            globals()[name]()
        write_macros(merge=True)
        sj = os.path.join(HERE, "stats_final.json")
        old = json.load(open(sj)) if os.path.exists(sj) else {}
        old.update(STATS)
        json.dump(old, open(sj, "w"), indent=1, default=float)
        return
    pairs = ms.all_pairs()
    print(len(pairs), "experiment-glacier pairs with both versions")
    M["nPairs"] = f"{len(pairs)}"
    loss_figure(pairs); s01_figure(); synthetic(); real_exp01(); perturbations(); global_inputs(); g01_field(); geodetic()
    write_macros()
    json.dump(STATS, open(os.path.join(HERE, "stats_final.json"), "w"), indent=1, default=float)
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
