#!/usr/bin/env python3
"""
Figures, statistics and LaTeX tables for report/report.tex, computed from input_data/, GROUP_UNIL/ (current
method01 = variant cnn), results/ and the archived SIADecompNet runs. Nothing in the report is typed by hand.

    python report/make_assets.py          # -> report/figures/*.pdf, report/tables/*.tex, report/stats.json
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
CX = os.path.dirname(HERE)
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
ARCH = os.path.join(CX, "archive", "siadecompnet_2026-10-13")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ── style (dataviz palette) ───────────────────────────────────────────────────────────────────────────────────
INK, INK2, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("o", ["#fde3d6", "#f8b896", "#f28d5e", "#eb6834", "#c94f1f", "#9a3a14", "#6b280d"])
SEQ_GRAY = LinearSegmentedColormap.from_list("g", ["#f4f3ef", "#c3c2b7", "#898781", "#52514e", "#2c2c2a"])
DIV = LinearSegmentedColormap.from_list("d", ["#1c5cab", "#86b6ef", NEUTRAL, "#ee9a9a", "#d03b3b"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                     "axes.grid": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
TW = 6.3   # text width (in)
STATS = {}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("fig", name)


def tex_table(name, header, rows, align, caption=None):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, name), "w").write("\n".join(lines) + "\n"); print("table", name)


def native(exp, gid):
    for p in (f"input_data/{exp}/{exp}_{gid}.nc", f"input_data/{exp}/{gid}/{exp}_{gid}_all.nc",
              f"input_data/{exp}/{exp}_{gid}_base_vars.nc"):
        p = os.path.join(CX, p)
        if os.path.exists(p):
            return xr.open_dataset(p)
    return None


def ice_of(ds, gid):
    if gid == "S01":
        return np.nan_to_num(ds["THK"].values) > 0
    return np.isfinite(ds["ICEMASK"].values)


def extent(ds):
    x, y = ds.x.values, ds.y.values
    return [x.min(), x.max(), y.min(), y.max()], ("lower" if y[0] < y[-1] else "upper")


def imshow(ax, ds, a, cmap, vmin, vmax, step=1):
    ext, org = extent(ds)
    im = ax.imshow(a[::step, ::step], cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, origin=org, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cbar(fig, im, ax, label, orientation="vertical", **kw):
    cb = fig.colorbar(im, ax=ax, orientation=orientation, fraction=kw.pop("fraction", 0.046), pad=kw.pop("pad", 0.02), **kw)
    cb.set_label(label, color=INK2); cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=6.5)
    return cb


def law(z):
    return np.minimum(0.01 * (z - 2350.0), 5.0)


def export(exp, gid, arch=False):
    p = (os.path.join(ARCH, "GROUP_UNIL_EXP01", f"{exp}_{gid}_method01.nc") if arch
         else os.path.join(CX, "GROUP_UNIL", exp, f"{exp}_{gid}_method01.nc"))
    return xr.open_dataset(p) if os.path.exists(p) else None


def on_grid(o, ds, var):
    """Export variable on the input's y orientation."""
    a = o[var].values.astype(float)
    return a[::-1] if o.y.values[0] != ds.y.values[0] else a


def scalebar(ax, ds, km, label=None):
    ext, _ = extent(ds)
    x0 = ext[0] + 0.05 * (ext[1] - ext[0]); y0 = ext[2] + 0.06 * (ext[3] - ext[2])
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, lw=1.5, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 0.03 * (ext[3] - ext[2]), label or f"{km:g} km", ha="center", va="bottom", fontsize=6.5)


# ── 0. glacier overview ───────────────────────────────────────────────────────────────────────────────────────
def glaciers():
    kind = {"G01": "ice cap (11 basins)", "G02": "mountain glacier", "G03": "mountain glacier", "G04": "mountain glacier", "G05": "mountain glacier",
            "G06": "mountain glacier", "S01": "synthetic, steady state", "S02": "synthetic, transient"}
    gpr = {"G01": "gridded", "G02": "profiles", "G03": "profiles", "G04": "gridded", "G05": "profiles", "G06": "gridded",
           "S01": "none", "S02": "profiles"}
    rows = []
    for gid in ["G01", "G02", "G03", "G04", "G05", "G06", "S01", "S02"]:
        m = json.load(open(os.path.join(CX, "data", f"EXP01_{gid}.json")))
        d = json.load(open(os.path.join(CX, "data", f"EXP01_{gid}_da.json")))
        ds = native("EXP01", gid)
        dxn = abs(float(ds.x[1] - ds.x[0]))
        seg = d.get("gpr_split", {}).get("n_segments")
        rows.append([gid, kind[gid], f"{dxn:g}", ("10 yr (assumed)" if gid.startswith("S") else f"{m['t1']:g}--{m['t2']:g}"), f"{m['dx_igm']:g}", f"{m['n_ice_cells_igm']:,}".replace(",", r"\,"),
                     gpr[gid] + (f" ({seg} seg.)" if seg else "")])
    tex_table("tab_glaciers.tex", ["", "type", r"native $\Delta x$ (m)", "window (yr)", r"model $\Delta x$ (m)", "ice cells", "thickness data"],
              rows, "llrrrrl")


# ── 1. synthetic glaciers: data ───────────────────────────────────────────────────────────────────────────────
def synthetic_data():
    s01, s02 = native("EXP01", "S01"), native("EXP01", "S02")
    # S01: 5 km x 1 km strip, 1 m grid
    ice = ice_of(s01, "S01")
    spd = np.where(ice, np.hypot(s01.VX.values, s01.VY.values), np.nan)
    fig, axs = plt.subplots(3, 1, figsize=(TW, 2.9))
    for ax, a, cm, lab, vr, t in (
            (axs[0], np.where(np.isfinite(s01.DEM.values), s01.DEM.values, np.nan), SEQ_GRAY, "m a.s.l.", (1600, 3000), "(a) surface elevation"),
            (axs[1], spd, SEQ_BLUE, "m yr$^{-1}$", (0, 60), "(b) surface speed"),
            (axs[2], np.where(ice, s01.THK.values, np.nan), SEQ_ORANGE, "m", (0, 130), "(c) ice thickness")):
        im = imshow(ax, s01, a, cm, *vr, step=4)
        ax.set_title(t, loc="left")
        cbar(fig, im, ax, lab, fraction=0.02, pad=0.01)
    scalebar(axs[2], s01, 1)
    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig_s01_data.pdf")

    ice = ice_of(s02, "S02")
    spd = np.where(ice, np.hypot(s02.VX.values, s02.VY.values), np.nan)
    fig, axs = plt.subplots(1, 4, figsize=(TW, 2.7))
    for ax, a, cm, lab, vr, t in (
            (axs[0], np.where(ice, s02.DEM.values, np.nan), SEQ_GRAY, "m a.s.l.", (1800, 4100), "(a) elevation"),
            (axs[1], spd, SEQ_BLUE, "m yr$^{-1}$", (0, 150), "(b) speed"),
            (axs[2], np.where(ice, s02.DHDT.values, np.nan), DIV, "m yr$^{-1}$", (-1.2, 1.2), "(c) d$h$/d$t$"),
            (axs[3], np.where(ice, s02.THK.values, np.nan), SEQ_ORANGE, "m", (0, 700), "(d) thickness")):
        im = imshow(ax, s02, a, cm, *vr)
        ax.set_title(t, loc="left")
        cbar(fig, im, ax, lab, orientation="horizontal", fraction=0.05, pad=0.03)
    scalebar(axs[0], s02, 5)
    save(fig, "fig_s02_data.pdf")

    # data table (quantities as rows)
    cols = {}
    for gid, ds in (("S01", s01), ("S02", s02)):
        ice = ice_of(ds, gid)
        dx = abs(float(ds.x[1] - ds.x[0]))
        v = np.hypot(ds.VX.values, ds.VY.values)[ice]
        d = ds.DHDT.values[ice]; t = ds.THK.values[ice]; z = ds.DEM.values[ice]
        un = lambda k: f"{np.nanmean(ds[k].values[ice]):.1f}" if k in ds else "--"  # noqa: E731
        e2 = native("EXP02", gid)
        cv = lambda k: (f"{100 * np.mean(np.isfinite(e2[k].values[ice_of(e2, gid)])):.1f}" if e2 is not None else "n/a")  # noqa: E731
        cols[gid] = [f"{dx:g}", f"{ice.sum() * dx * dx / 1e6:.2f}", f"{np.nanmin(z):.0f}--{np.nanmax(z):.0f}",
                     f"{np.nanmean(v):.0f} / {np.nanmax(v):.0f}", f"{np.nanmean(d):+.2f} [{np.nanmin(d):+.2f}, {np.nanmax(d):+.2f}]",
                     f"{np.nanmean(t):.0f} / {np.nanmax(t):.0f}", un("UNCT_VX"), un("UNCT_DHDT"), un("UNCT_THK"),
                     "100 / 100 / 100", f"{cv('VX')} / {cv('THK')}"]
    labels = [r"grid spacing (m)", r"glacier area (km$^2$)", "surface elevation (m a.s.l.)", r"surface speed mean / max (m yr$^{-1}$)",
              r"d$h$/d$t$ mean [range] (m yr$^{-1}$)", "thickness mean / max (m)", r"$\sigma_{u}$ mean (m yr$^{-1}$)",
              r"$\sigma_{{\rm d}h/{\rm d}t}$ mean (m yr$^{-1}$)", r"$\sigma_{H}$ mean (m)",
              r"EXP01 coverage of $u_s$ / d$h$/d$t$ / $H$ (\%)", r"EXP02 coverage of $u_s$ / $H$ (\%)"]
    tex_table("tab_synthetic_data.tex", ["", "S01", "S02"], [[l, cols["S01"][i], cols["S02"][i]] for i, l in enumerate(labels)], "lrr")


# ── 2. perturbation experiments ───────────────────────────────────────────────────────────────────────────────
def perturbations():
    rows = []
    for e in [f"EXP{i:02d}" for i in range(3, 16)]:
        cells = [e]
        for gid in ("S01", "S02", "G01", "G05"):
            ref, ds = native("EXP01", gid), native(e, gid)
            dx_ref, dx = abs(float(ref.x[1] - ref.x[0])), abs(float(ds.x[1] - ds.x[0]))
            if ds.THK.shape != ref.THK.shape:
                cells.append(f"resolution {dx_ref:g} $\\rightarrow$ {dx:g} m"); continue
            ice = ice_of(ref, gid)
            out = []
            mt = ice & (ref.THK.values > 20) & np.isfinite(ds.THK.values)
            rt = ds.THK.values[mt] / ref.THK.values[mt] - 1
            if np.abs(rt).max() > 1e-6:
                out.append(f"$H$: {100 * np.mean(rt):+.0f}\\,\\% $\\pm$ {100 * np.std(rt):.0f}\\,\\%")
            v0 = np.hypot(ref.VX.values, ref.VY.values); v1 = np.hypot(ds.VX.values, ds.VY.values)
            mv = ice & (v0 > 5) & np.isfinite(v1)
            rv = v1[mv] / v0[mv] - 1
            ang = np.degrees(np.angle(np.exp(1j * (np.arctan2(ds.VY.values[mv], ds.VX.values[mv]) - np.arctan2(ref.VY.values[mv], ref.VX.values[mv])))))
            if np.abs(rv).max() > 1e-6 or np.abs(ang).max() > 1e-3:
                out.append(f"$|u_s|$: {100 * np.median(rv):+.0f}\\,\\% $\\pm$ {100 * (np.percentile(rv, 84) - np.percentile(rv, 16)) / 2:.0f}\\,\\%"
                           + (f", dir. $\\pm${np.std(ang):.1f}$^\\circ$" if np.std(ang) > 0.2 else ""))
            cells.append("; ".join(out) if out else "identical to EXP01")
        rows.append(cells)
    tex_table("tab_perturbations.tex", ["", "S01", "S02", "G01", "G05"], rows, "lllll")


# ── 3. limited coverage ───────────────────────────────────────────────────────────────────────────────────────
def coverage():
    panels = [("S02", "vel+gpr"), ("G02", "gpr"), ("G03", "gpr"), ("G05", "gpr"), ("G04", "vel"), ("G06", "vel")]
    fig, axs = plt.subplots(2, 3, figsize=(TW, 4.6))
    cov = {}
    cm_mask = ListedColormap([NEUTRAL])
    for ax, (gid, kind) in zip(axs.ravel(), panels):
        ds = native("EXP02", gid)
        ice = np.isfinite(ds.ICEMASK.values)
        step = max(1, int(round(max(ds.sizes["x"], ds.sizes["y"]) / 700)))
        imshow(ax, ds, np.where(ice, 1.0, np.nan), cm_mask, 0, 1, step=step)
        vel = ice & np.isfinite(ds.VX.values)
        thk = ice & np.isfinite(ds.THK.values) & (np.nan_to_num(ds.THK.values) > 0)
        cov[gid] = {"velocity_pct": float(100 * vel.sum() / ice.sum()), "thickness_pct": float(100 * thk.sum() / ice.sum())}
        lab = []
        if "vel" in kind:
            imshow(ax, ds, np.where(vel, 1.0, np.nan), ListedColormap([CAT[0]]), 0, 1, step=step)
            lab.append(f"$u_s$ {cov[gid]['velocity_pct']:.0f}\\%")
        if "gpr" in kind:
            iy, ix = np.nonzero(thk)
            ax.scatter(ds.x.values[ix], ds.y.values[iy], s=0.6 if thk.sum() > 2000 else 2.5, c=CAT[1], linewidths=0, rasterized=True)
            lab.append(f"GPR {cov[gid]['thickness_pct']:.1f}\\%")
        ax.set_title(f"{gid}: " + ", ".join(lab).replace("\\%", "%"), loc="left")
        ext, _ = extent(ds)
        span = ext[1] - ext[0]
        km = 1 if span < 6000 else (2 if span < 15000 else 5)
        scalebar(ax, ds, km)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=NEUTRAL, label="glacier (no observation)"), Patch(color=CAT[0], label="surface velocity observed"),
                        Patch(color=CAT[1], label="GPR thickness")], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.05, hspace=0.18, bottom=0.06)
    save(fig, "fig_coverage.pdf")
    STATS["exp02_coverage"] = cov


# ── 4. synthetic results ──────────────────────────────────────────────────────────────────────────────────────
SYN_EXPS = ["EXP01", "EXP02"] + [f"EXP{i:02d}" for i in range(3, 16)]


def synth_metrics(exp, gid, arch=False):
    o = export(exp, gid, arch)
    if o is None:
        return None
    ds = native(exp, gid)
    truth = native("EXP01", gid)
    ice = ice_of(truth, gid) if truth.THK.shape == ds.THK.shape else ice_of(ds, gid)   # perturbed THK must not define the mask
    smb, fdiv, thk = (on_grid(o, ds, v) for v in ("SMB", "FDIV", "THK"))
    m = ice & np.isfinite(smb)
    r = {"glacierwide_smb": float(np.nanmean(smb[m])), "fdiv_mean": float(np.nanmean(fdiv[m])),
         "dhdt_model": float(np.nanmean((smb - fdiv)[m])), "dhdt_obs": float(np.nanmean(ds.DHDT.values[m]))}
    t_true = (truth.THK.values if truth.THK.shape == thk.shape and exp != "EXP14" else ds.THK.values).astype(float)
    mt = m & np.isfinite(thk) & np.isfinite(t_true) & (t_true > 0)
    d = thk[mt] - t_true[mt]
    r.update(thk_bias=float(d.mean()), thk_rmse=float(np.sqrt(np.mean(d ** 2))), thk_mean=float(thk[mt].mean()),
             thk_true=float(t_true[mt].mean()), thk_r=float(np.corrcoef(thk[mt], t_true[mt])[0, 1]))
    if gid == "S01":
        z = ds.DEM.values.astype(float)
        e = smb[m] - law(z[m]); ef = fdiv[m] - law(z[m])
        r.update(smb_bias=float(e.mean()), smb_rmse=float(np.sqrt(np.mean(e ** 2))),
                 smb_r2=float(1 - np.sum(e ** 2) / np.sum((law(z[m]) - law(z[m]).mean()) ** 2)),
                 fdiv_rmse=float(np.sqrt(np.mean(ef ** 2))), law_glacierwide=float(np.mean(law(z[m]))))
    ela = o.attrs.get("ELA_admissible_m")
    if ela:
        r["ela"] = [float(v) for v in ela.strip("[]").split(",")]
    st = os.path.join(CX, "results", f"{exp}_{gid}_cnn", "state.json")
    if not arch and os.path.exists(st):
        s = json.load(open(st)); r.update(tau=s["init_slidingco"], A=s["init_arrhenius"])
    g = os.path.join(CX, "results", f"{exp}_{gid}_cnn", "gpr_scores.json")
    if not arch and os.path.exists(g):
        r["gpr"] = json.load(open(g))["member_mean"]
    return r


def synthetic_results():
    res = {gid: {e: synth_metrics(e, gid) for e in SYN_EXPS} for gid in ("S01", "S02")}
    old = {gid: synth_metrics("EXP01", gid, arch=True) for gid in ("S01", "S02")}
    STATS["synthetic"] = res; STATS["synthetic_old"] = old

    # S01 profiles: truth, new family, old family, all experiments
    o = export("EXP01", "S01"); oo = export("EXP01", "S01", arch=True); src = native("EXP01", "S01")
    zr = np.nanpercentile(src.DEM.values[ice_of(src, "S01")], [1, 99])
    zz = np.linspace(zr[0], zr[1], 200)
    fig, axs = plt.subplots(1, 2, figsize=(TW, 3.0), sharey=True)
    ax = axs[0]
    for k in range(o.sizes["member"]):
        ax.plot(o.MEMBER_SMB_PROFILE.values[k], o.z.values, color=CAT[0], lw=0.6, alpha=0.5)
    ax.plot(o.SMB_PROFILE.values, o.z.values, color=CAT[0], lw=2, label="CNN emulator (family mean)")
    ax.plot(oo.SMB_PROFILE.values, oo.z.values, color=CAT[1], lw=1.6, ls="--", label="SIADecompNet (family mean)")
    ax.plot(law(zz), zz, color=INK, lw=1.2, label="true SMB law")
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("(a) S01, EXP01: inferred SMB($z$)", loc="left")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(zr[0], zr[1]); ax.set_xlim(-8, 6)
    ax = axs[1]
    groups = {"THK (EXP03--07)": (range(3, 8), CAT[1]), "VEL (EXP08--13)": (range(8, 14), CAT[2]), "RES (EXP14--15)": (range(14, 16), CAT[6])}
    for lab, (rng, c) in groups.items():
        for i, n in enumerate(rng):
            oe = export(f"EXP{n:02d}", "S01")
            if oe is not None:
                ax.plot(oe.SMB_PROFILE.values, oe.z.values, color=c, lw=0.9, alpha=0.85, label=lab if i == 0 else None)
    ax.plot(o.SMB_PROFILE.values, o.z.values, color=CAT[0], lw=2, label="EXP01")
    ax.plot(law(zz), zz, color=INK, lw=1.2, label="true SMB law")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_title("(b) S01, all perturbation experiments", loc="left")
    ax.legend(frameon=False, loc="lower right"); ax.set_xlim(-8, 6)
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_s01_profiles.pdf")

    # S02 thickness: truth, DA, difference, scatter
    src = native("EXP01", "S02"); o = export("EXP01", "S02"); ice = ice_of(src, "S02")
    thk = on_grid(o, src, "THK"); tt = src.THK.values.astype(float)
    dd = json.load(open(os.path.join(CX, "data", "EXP01_S02_da.json")))
    da = xr.open_dataset(os.path.join(CX, "data", "EXP01_S02_da.nc"))
    fig = plt.figure(figsize=(TW, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.32, 1.25], wspace=0.22)
    axs = [fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4)]
    im = imshow(axs[0], src, np.where(ice, tt, np.nan), SEQ_ORANGE, 0, 700); axs[0].set_title("(a) true $H$", loc="left")
    cbar(fig, im, axs[0], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    im = imshow(axs[1], src, np.where(ice, thk, np.nan), SEQ_ORANGE, 0, 700); axs[1].set_title("(b) inferred $H$", loc="left")
    cbar(fig, im, axs[1], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    im = imshow(axs[2], src, np.where(ice, thk - tt, np.nan), DIV, -200, 200); axs[2].set_title("(c) inferred $-$ true", loc="left")
    cbar(fig, im, axs[2], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    for name, c, lab in (("thkobs", INK, "GPR assimilated"), ("thkobs_val", CAT[1], "GPR withheld (cross-validation)")):
        a = da[name].values
        iy, ix = np.nonzero(np.isfinite(a))
        axs[1].scatter(da.x.values[ix], da.y.values[iy], s=3, c=c, linewidths=0, label=lab)
    axs[1].legend(frameon=False, loc="lower left", fontsize=6, markerscale=2, handletextpad=0.1, borderaxespad=0.1)
    ax = axs[3]
    m = ice & np.isfinite(thk)
    ax.scatter(tt[m], thk[m], s=1, c=CAT[0], alpha=0.35, linewidths=0, rasterized=True)
    ax.plot([0, 760], [0, 760], color=INK, lw=0.8)
    ax.set_xlim(0, 760); ax.set_ylim(0, 760); ax.set_aspect("equal")
    ax.set_xlabel("true $H$ (m)"); ax.set_ylabel("inferred $H$ (m)")
    r = res["S02"]["EXP01"]
    ax.set_title(f"(d) bias {r['thk_bias']:+.0f} m, RMSE {r['thk_rmse']:.0f} m", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "fig_s02_thickness.pdf")
    STATS["s02_gpr_split"] = dd.get("gpr_split")

    # sensitivity across experiments
    exps = [e for e in SYN_EXPS if res["S01"].get(e) or res["S02"].get(e)]
    fig, axs = plt.subplots(1, 3, figsize=(TW, 2.5))
    xs = np.arange(len(SYN_EXPS))
    def dots(ax, gid, key, c, lab, top=None, dy=-9):
        y = np.array([res[gid][e][key] if res[gid].get(e) and key in res[gid][e] else np.nan for e in SYN_EXPS], float)
        out = np.isfinite(y) & (y > top) if top is not None else np.zeros(y.size, bool)
        ax.plot(xs[~out], y[~out], "o", ms=4, color=c, label=lab, markeredgecolor="white", markeredgewidth=0.5)
        for i in np.nonzero(out)[0]:     # off-scale: marker at the top edge with the value
            ax.plot(xs[i], top, "^", ms=5, color=c, markeredgecolor="white", markeredgewidth=0.5, clip_on=False)
            ax.annotate(f"{y[i]:+.1f}" if key == "glacierwide_smb" else f"{y[i]:.0f}", (xs[i], top), xytext=(-3, dy),
                        textcoords="offset points", ha="right", fontsize=6, color=c)
        return y
    dots(axs[0], "S01", "smb_rmse", CAT[0], "S01 vs true law", top=2.0); axs[0].set_ylim(0, 2.0)
    axs[0].set_title("(a) S01: pixel SMB RMSE", loc="left"); axs[0].set_ylabel("m i.e. yr$^{-1}$")
    dots(axs[1], "S01", "glacierwide_smb", CAT[0], "S01", top=0.3)
    dots(axs[1], "S02", "glacierwide_smb", CAT[1], "S02", top=0.3, dy=-19); axs[1].set_ylim(-0.4, 0.3)
    axs[1].axhline(res["S01"]["EXP01"]["law_glacierwide"], color=CAT[0], lw=0.8, ls="--", label="S01 true")
    axs[1].set_title("(b) glacier-wide SMB", loc="left"); axs[1].set_ylabel("m i.e. yr$^{-1}$")
    axs[1].legend(frameon=False, fontsize=6.5, loc="best")
    dots(axs[2], "S01", "thk_rmse", CAT[0], "S01", top=110)
    dots(axs[2], "S02", "thk_rmse", CAT[1], "S02", top=110); axs[2].set_ylim(0, 110)
    axs[2].set_title("(c) thickness RMSE vs true $H$", loc="left"); axs[2].set_ylabel("m")
    axs[2].legend(frameon=False, fontsize=6.5)
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([e[3:] for e in SYN_EXPS], fontsize=5.5, rotation=90)
        ax.set_xlabel("experiment"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_synthetic_sensitivity.pdf")

    # tables: new vs old (EXP01) and per-experiment
    f = lambda v, p=2, s=False: ("--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{p}f}" if s else f"{v:.{p}f}"))  # noqa: E731
    n1, o1, n2, o2 = res["S01"]["EXP01"], old["S01"], res["S02"]["EXP01"], old["S02"]
    rows = [
        ["pixel SMB RMSE (m i.e. yr$^{-1}$)", f(n1["smb_rmse"]), f(o1["smb_rmse"]), "n/a", "n/a"],
        ["pixel SMB $R^2$", f(n1["smb_r2"], 3), f(o1["smb_r2"], 3), "n/a", "n/a"],
        ["ELA range of the family (m; true 2350)", "--".join(f"{v:.0f}" for v in n1["ela"]), "--".join(f"{v:.0f}" for v in o1["ela"]), "--", "--"],
        ["glacier-wide SMB (m i.e. yr$^{-1}$)", f(n1["glacierwide_smb"], 3, True) + f" (true {n1['law_glacierwide']:+.3f})",
         f(o1["glacierwide_smb"], 3, True), f(n2["glacierwide_smb"], 3, True), f(o2["glacierwide_smb"], 3, True)],
        ["modelled / observed d$h$/d$t$ (m yr$^{-1}$)", f"{n1['dhdt_model']:+.2f} / {n1['dhdt_obs']:+.2f}", f"{o1['dhdt_model']:+.2f} / {o1['dhdt_obs']:+.2f}",
         f"{n2['dhdt_model']:+.2f} / {n2['dhdt_obs']:+.2f}", f"{o2['dhdt_model']:+.2f} / {o2['dhdt_obs']:+.2f}"],
        ["FDIV RMSE vs true (m i.e. yr$^{-1}$)", f(n1["fdiv_rmse"]), f(o1["fdiv_rmse"]), "n/a", "n/a"],
        ["thickness bias / RMSE (m)", f"{n1['thk_bias']:+.0f} / {n1['thk_rmse']:.0f}", f"{o1['thk_bias']:+.0f} / {o1['thk_rmse']:.0f}",
         f"{n2['thk_bias']:+.0f} / {n2['thk_rmse']:.0f}", f"{o2['thk_bias']:+.0f} / {o2['thk_rmse']:.0f}"],
        ["thickness correlation", f(n1["thk_r"], 2), f(o1["thk_r"], 2), f(n2["thk_r"], 2), f(o2["thk_r"], 2)],
    ]
    tex_table("tab_synthetic_old_new.tex", ["", "S01 CNN", "S01 SIADecompNet", "S02 CNN", "S02 SIADecompNet"], rows, "lrrrr")

    rows = []
    for e in SYN_EXPS:
        a, b = res["S01"].get(e), res["S02"].get(e)
        fail = lambda r: r is not None and r.get("glacierwide_smb", 0) > 1.0   # forward model broke (resolution)  # noqa: E731
        rows.append([e + ("$^\\dagger$" if fail(a) or fail(b) else ""),
                     *(["n/a"] * 6 if a is None else [f"{a['tau']:.3f}", f"{a['A']:.0f}", f(a["smb_rmse"]), f(a["glacierwide_smb"], 2, True),
                                                     f"{a['dhdt_model']:+.2f}", f"{a['thk_rmse']:.0f}"]),
                     *(["--"] * 6 if b is None else [f"{b['tau']:.3f}", f"{b['A']:.0f}", f(b["glacierwide_smb"], 2, True),
                                                     f"{b['dhdt_model']:+.2f} / {b['dhdt_obs']:+.2f}", f"{b['thk_rmse']:.0f}",
                                                     f(b.get("gpr", {}).get("val_rmse"), 0)])])
    tex_table("tab_synthetic_all.tex",
              ["", r"$\tau_{\rm ref}$", "$A$", r"SMB RMSE", r"$\overline{\rm SMB}$", r"$\overline{\dot h}_{\rm mod}$", r"$H$ RMSE",
               r"$\tau_{\rm ref}$", "$A$", r"$\overline{\rm SMB}$", r"$\overline{\dot h}_{\rm mod}$ / obs", r"$H$ RMSE", "GPR c.-v."],
              rows, "l|rrrrrr|rrrrrr")


# ── 4b. S01 twin experiment (SMB phase rerun on one family member, tools/smb_rerun.py) ─────────────────────────
TWIN_MEMBER = "slid_0.4519"


def twin_test():
    """Original SMB phase on the real observations, the true law evaluated on them (no iteration), and the twin
    run whose target is a forward run with the true law."""
    root = os.path.join(CX, "results", "EXP01_S01_cnn")
    src = native("EXP01", "S01")
    zr = np.nanpercentile(src.DEM.values[ice_of(src, "S01")], [1, 99])
    zf = np.arange(np.ceil(zr[0] / 10) * 10, zr[1], 10.0)      # scoring grid of tools/s01_score.py

    def load(tag):
        sd = os.path.join(root, f"smb_{tag}", TWIN_MEMBER, "smb_inference")
        v = np.load(os.path.join(sd, "smb_vec.npy")).astype(float)
        z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
        p = (np.clip(1.0 - np.abs((zf - z0)[:, None] / dz - np.arange(v.size)), 0.0, None) * v).sum(-1)
        H = np.load(os.path.join(sd, "thk_optimized.npy"))
        tw = os.path.join(sd, "thk_twin_obs.npy")
        O = np.load(tw if os.path.exists(tw) else os.path.join(sd, "thk_observed.npy"))
        return dict(p=p, rmse=float(np.sqrt(np.mean((p - law(zf)) ** 2))), data=np.load(os.path.join(sd, "data_history.npy")),
                    H=H, O=O, zs=np.load(os.path.join(sd, "usurf_modeled.npy")) - H + O)

    real, truth, twin = load("base"), load("truth_ramp"), load("twin_ramp")
    # residual of the true law on the real observations: share within 50 m bands, cross-flow pattern in the trunk
    m = truth["O"] > 1
    r, zb = (truth["H"] - truth["O"])[m], np.floor(truth["zs"][m] / 50.0)
    bm = np.array([r[zb == k].mean() for k in zb])
    rr = np.where(m, truth["H"] - truth["O"], np.nan)[:, 10:70]
    row = np.nanmean(rr, 1)
    st = {"member": TWIN_MEMBER, "window_yr": 10.0,
          "real": {"rmse": real["rmse"], "data0": float(real["data"][0]), "data": float(real["data"][-1]), "it": int(real["data"].size)},
          "truth": {"rmse": truth["rmse"], "data": float(truth["data"][-1]),
                    "within_band_frac": float(np.mean((r - bm) ** 2) / np.mean(r ** 2)),
                    "trunk_row_mean_min": float(np.nanmin(row)), "trunk_row_mean_max": float(np.nanmax(row)),
                    "head_cols_residual": [float(np.nanmean(np.where(m, truth["H"] - truth["O"], np.nan)[:, k])) for k in range(3)]},
          "twin": {"rmse": twin["rmse"], "data0": float(twin["data"][0]), "data": float(twin["data"][-1]), "it": int(twin["data"].size)}}
    STATS["twin"] = st

    fig, axs = plt.subplots(1, 2, figsize=(TW, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    ax.plot(twin["p"], zf, color=CAT[4], lw=2.6, label=f"twin experiment (RMSE {twin['rmse']:.2f})")
    ax.plot(real["p"], zf, color=CAT[0], lw=2, label=f"real observations (RMSE {real['rmse']:.2f})")
    ax.plot(law(zf), zf, color=INK, lw=1.0, ls=(0, (4, 2)), label="true SMB law")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
    ax.set_title("(a) inferred SMB($z$)", loc="left")
    ax.legend(frameon=False, loc="lower right"); ax.set_ylim(zr[0], zr[1]); ax.set_xlim(-8, 6)
    ax = axs[1]
    ax.plot(np.arange(1, real["data"].size + 1), real["data"], color=CAT[0], lw=1.8, marker="o", ms=2.5, label="real observations")
    ax.axhline(truth["data"], color=INK2, lw=1.0, ls=(0, (4, 2)), label="true SMB law on the real observations")
    ax.plot(np.arange(1, twin["data"].size + 1), twin["data"], color=CAT[4], lw=1.8, marker="o", ms=2.5, label="twin experiment")
    ax.set_xlabel("L-BFGS iteration"); ax.set_ylabel("RMS end-of-window thickness misfit (m)")
    ax.set_title("(b) data misfit", loc="left")
    ax.set_ylim(0, 50); ax.legend(frameon=False, loc="center right")
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_s01_twin.pdf")


# ── 5. GPR effect ─────────────────────────────────────────────────────────────────────────────────────────────
def gpr_effect():
    cases = [("S02", "EXP01_S02", "results/EXP01_S02_cnn/family/slid_0.0733", "results/EXP01_S02_cnn/tests/da_nogpr_slid0.0733"),
             ("G05", "EXP01_G05", "results/EXP01_G05_cnn/family/slid_0.0693", "results/EXP01_G05_cnn/tests/da_nogpr_best")]
    rows, out = [], {}
    for gid, base, w, wo in cases:
        inp = xr.open_dataset(os.path.join(CX, "data", base + "_da.nc"))
        truth = xr.open_dataset(os.path.join(CX, "data", base + ".nc")) if gid == "S02" else None
        for lab, d in (("with GPR", w), ("without GPR", wo)):
            p = os.path.join(CX, d, "geology-optimized.nc")
            if not os.path.exists(p):
                continue
            o = xr.open_dataset(p); t = o.thk.values.astype(float); m = o.icemask.values > 0.5
            vel = float(np.sqrt(np.nanmean((o.velsurf_mag.values[m] - o.velsurfobs_mag.values[m]) ** 2)))
            sc = {}
            for s, k in (("train", "thkobs"), ("val", "thkobs_val"), ("test", "thkobs_test")):
                kk = np.isfinite(inp[k].values)
                sc[s] = float(np.sqrt(np.mean((t[kk] - inp[k].values[kk]) ** 2))) if kk.any() else float("nan")
            tr = float(np.sqrt(np.mean((t[m] - truth.thk.values[m]) ** 2))) if truth is not None else float("nan")
            out[f"{gid} {lab}"] = dict(sc, vel=vel, truth_rmse=tr, n=[int(np.isfinite(inp[k].values).sum()) for k in ("thkobs", "thkobs_val", "thkobs_test")])
            f = lambda v: "--" if not np.isfinite(v) else f"{v:.0f}"  # noqa: E731
            rows.append([f"{gid} (EXP01)", lab, f(sc["train"]), f(sc["val"]), f(sc["test"]), f(tr), f"{vel:.1f}"])
    tex_table("tab_gpr_effect.tex", ["glacier", "DA", "assimilated GPR", "cross-validation GPR", "independent GPR",
                                     r"whole glacier vs true $H$", r"$u_s$ misfit (m yr$^{-1}$)"], rows, "llrrrrr")
    STATS["gpr_effect"] = out


# ── 6. real glaciers ──────────────────────────────────────────────────────────────────────────────────────────
REAL = ["G01", "G02", "G03", "G04", "G05", "G06"]


def real_results():
    rows, stats = [], {}
    fig, axs = plt.subplots(2, 3, figsize=(TW, 4.6))
    for ax, gid in zip(axs.ravel(), REAL):
        o = export("EXP01", gid); oo = export("EXP01", gid, arch=True)
        if o is None:
            ax.axis("off"); continue
        src = native("EXP01", gid)
        smb, fdiv = on_grid(o, src, "SMB"), on_grid(o, src, "FDIV")
        m = np.isfinite(smb)
        dh = src.DHDT.values.astype(float)
        dh_obs = float(np.nanmean(np.where(m & (dh != 0), dh, np.nan)))
        s = json.load(open(os.path.join(CX, "results", f"EXP01_{gid}_cnn", "state.json")))
        fs = json.load(open(os.path.join(CX, "results", f"EXP01_{gid}_cnn", "family_summary_method01.json")))
        g = os.path.join(CX, "results", f"EXP01_{gid}_cnn", "gpr_scores.json")
        gp = json.load(open(g))["member_mean"] if os.path.exists(g) else None
        best = min(fs["members"], key=lambda r: r["velsurf"])
        geo = xr.open_dataset(os.path.join(best["dir"], "geology-optimized.nc")); mm = geo.icemask.values > 0.5
        vrm = float(np.sqrt(np.nanmean((geo.velsurf_mag.values[mm] - geo.velsurfobs_mag.values[mm]) ** 2)))
        vobs = float(np.nanmean(geo.velsurfobs_mag.values[mm]))
        ela = [float(v) for v in o.attrs["ELA_admissible_m"].strip("[]").split(",")]
        old_smb = float(np.nanmean(on_grid(oo, src, "SMB"))) if oo is not None else float("nan")
        r = dict(tau=s["init_slidingco"], A=s["init_arrhenius"], reg=s["reg_thk"], std=s.get("thkobs_std"), vel_rmse=vrm, vel_mean=vobs,
                 smb=float(np.nanmean(smb)), fdiv=float(np.nanmean(fdiv)), dhdt_model=float(np.nanmean((smb - fdiv)[m])),
                 dhdt_obs=dh_obs, ela=ela, n_adm=int(o.attrs["n_admissible"]), n=int(o.attrs["n_members"]),
                 thk=float(np.nanmean(on_grid(o, src, "THK"))), old_smb=old_smb, gpr=gp,
                 unct_smb=fs["mean_unct_smb"], components=fs["mean_components"])
        stats[gid] = r
        rows.append([gid, f"{r['tau']:.3f}", f"{r['A']:.0f}",
                     f"{r['vel_rmse']:.1f} ({r['vel_mean']:.0f})", f"{r['thk']:.0f}",
                     "--" if gp is None else f"{gp['test_rmse']:.0f}" if np.isfinite(gp["test_rmse"]) else "--",
                     f"{r['smb']:+.2f}", f"{r['dhdt_model']:+.2f} / {r['dhdt_obs']:+.2f}", f"{r['fdiv']:+.2f}",
                     f"{np.mean(ela):.0f}", f"{r['unct_smb']:.2f}", f"{old_smb:+.2f}"])
        # profile panel
        for k in range(o.sizes["member"]):
            ax.plot(o.MEMBER_SMB_PROFILE.values[k], o.z.values, color=CAT[0], lw=0.5, alpha=0.45)
        pm, ps = o.SMB_PROFILE.values, o.UNCT_SMB_PROFILE.values
        ax.fill_betweenx(o.z.values, pm - np.nan_to_num(ps), pm + np.nan_to_num(ps), color=CAT[0], alpha=0.15, lw=0)
        ax.plot(pm, o.z.values, color=CAT[0], lw=1.8, label="CNN family mean")
        if oo is not None:
            ax.plot(oo.SMB_PROFILE.values, oo.z.values, color=CAT[1], lw=1.2, ls="--", label="SIADecompNet")
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        zs = src.DEM.values[m]
        ax.set_ylim(np.nanpercentile(zs, 0.5), np.nanpercentile(zs, 99.5))
        ax.set_title(f"{gid}  (ELA {np.mean(ela):.0f} m)", loc="left")
        ax.set_xlabel("SMB (m i.e. yr$^{-1}$)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("elevation (m a.s.l.)"); axs[1, 0].set_ylabel("elevation (m a.s.l.)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(h + [plt.Rectangle((0, 0), 1, 1, color=CAT[0], alpha=0.15)], l + ["family spread ($\\pm 1\\sigma$)"],
               loc="lower center", ncol=3, frameon=False)
    save(fig, "fig_real_profiles.pdf")
    tex_table("tab_real.tex", ["", r"$\tau_{\rm ref}$", "$A$", r"$u_s$ RMSE (mean)", r"$\overline{H}$", "indep. GPR",
                               r"$\overline{\rm SMB}$", r"$\overline{{\rm d}h/{\rm d}t}$ mod / obs", r"$\overline{\rm FDIV}$", "ELA",
                               r"$\overline{\sigma}_{\rm SMB}$", r"$\overline{\rm SMB}$ SIA"], rows, "lrrrrrrrrrrr")
    STATS["real_exp01"] = stats

    # G05 example maps
    gid = "G05"; o = export("EXP01", gid); src = native("EXP01", gid)
    smb, fdiv, thk, un = (on_grid(o, src, v) for v in ("SMB", "FDIV", "THK", "UNCT_SMB"))
    fig, axs = plt.subplots(1, 4, figsize=(TW, 2.9))
    st = 4
    im = imshow(axs[0], src, thk, SEQ_ORANGE, 0, 800, step=st); axs[0].set_title("(a) thickness $H$", loc="left")
    cbar(fig, im, axs[0], "m", orientation="horizontal", fraction=0.05, pad=0.03)
    im = imshow(axs[1], src, smb, DIV, -8, 8, step=st); axs[1].set_title("(b) SMB", loc="left")
    cbar(fig, im, axs[1], "m i.e. yr$^{-1}$", orientation="horizontal", fraction=0.05, pad=0.03)
    im = imshow(axs[2], src, fdiv, DIV, -8, 8, step=st); axs[2].set_title("(c) FDIV", loc="left")
    cbar(fig, im, axs[2], "m i.e. yr$^{-1}$", orientation="horizontal", fraction=0.05, pad=0.03)
    im = imshow(axs[3], src, un, SEQ_BLUE, 0, 5, step=st); axs[3].set_title("(d) $\\sigma_{\\rm SMB}$", loc="left")
    cbar(fig, im, axs[3], "m i.e. yr$^{-1}$", orientation="horizontal", fraction=0.05, pad=0.03)
    scalebar(axs[0], src, 5)
    save(fig, "fig_g05_maps.pdf")


# ── 7. real glaciers: perturbations (G01, G05) and globally available inputs ────────────────────────────────
GLOB = {"EXP16": ("site", "Millan22"), "EXP17": ("Farinotti19", "site"), "EXP18": ("Millan22", "site"),
        "EXP19": ("Ma25", "site"), "EXP20": ("Farinotti19", "Millan22")}   # (THK product, velocity product); DHDT = Hugonnet21


def parse_ela(o):
    v = [float(x) for x in o.attrs.get("ELA_admissible_m", "[]").strip("[]").split(",") if x.strip()]
    return float(np.mean(v)) if v else float("nan")     # empty: the SMB profile does not cross zero on the glacier


def real_metrics(exp, gid):
    o = export(exp, gid)
    if o is None:
        return None
    src = native(exp, gid) or xr.open_dataset(os.path.join(CX, "input_data", exp, f"{exp}_{gid}.nc"))
    smb, fdiv, thk = (on_grid(o, src, v) for v in ("SMB", "FDIV", "THK"))
    m = np.isfinite(smb)
    dh = src.DHDT.values.astype(float)
    d = os.path.join(CX, "results", f"{exp}_{gid}_cnn")
    s = json.load(open(os.path.join(d, "state.json")))
    fs = json.load(open(os.path.join(d, "family_summary_method01.json")))
    best = min(fs["members"], key=lambda r: r["velsurf"])
    geo = xr.open_dataset(os.path.join(best["dir"], "geology-optimized.nc")); mm = geo.icemask.values > 0.5
    r = dict(smb=float(np.nanmean(smb)), fdiv=float(np.nanmean(fdiv)), dhdt_model=float(np.nanmean((smb - fdiv)[m])),
             dhdt_obs=float(np.nanmean(np.where(m & (dh != 0), dh, np.nan))), thk=float(np.nanmean(thk[m])), ela=parse_ela(o),
             unct_smb=float(np.nanmean(o.UNCT_SMB.values)), tau=s["init_slidingco"], A=s["init_arrhenius"], reg=s["reg_thk"],
             vel_rmse=float(np.sqrt(np.nanmean((geo.velsurf_mag.values[mm] - geo.velsurfobs_mag.values[mm]) ** 2))),
             vel_mean=float(np.nanmean(geo.velsurfobs_mag.values[mm])),
             dx=json.load(open(os.path.join(CX, "data", f"{exp}_{gid}.json")))["dx_native"])
    g = os.path.join(d, "gpr_scores.json")
    if os.path.exists(g):
        gp = json.load(open(g))["member_mean"]; r.update(gpr_val=gp.get("val_rmse"), gpr_test=gp.get("test_rmse"))
    r["profile"] = (o.z.values.tolist(), o.SMB_PROFILE.values.tolist())
    return r


def fmt(v, p=2, sign=False):
    return "--" if v is None or not np.isfinite(v) else (f"{v:+.{p}f}" if sign else f"{v:.{p}f}")


def real_sensitivity():
    exps = ["EXP01"] + [f"EXP{i:02d}" for i in range(3, 16)]
    res = {gid: {e: real_metrics(e, gid) for e in exps} for gid in ("G01", "G05")}
    STATS["real_sensitivity"] = {g: {e: {k: v for k, v in (r or {}).items() if k != "profile"} for e, r in d.items()} for g, d in res.items()}
    xs = np.arange(len(exps))
    fig, axs = plt.subplots(1, 3, figsize=(TW, 2.5))
    for gid, c in (("G01", CAT[0]), ("G05", CAT[1])):
        R = res[gid]
        y = lambda k: np.array([R[e][k] if R[e] and R[e].get(k) is not None else np.nan for e in exps], float)  # noqa: E731
        ref = R["EXP01"]
        axs[0].plot(xs, y("smb") - ref["smb"], "o", ms=4, color=c, label=gid, markeredgecolor="white", markeredgewidth=0.5)
        axs[1].plot(xs, y("ela") - ref["ela"], "o", ms=4, color=c, label=gid, markeredgecolor="white", markeredgewidth=0.5)
        axs[2].plot(xs, y("thk"), "o", ms=4, color=c, label=f"{gid} mean $H$", markeredgecolor="white", markeredgewidth=0.5)
        if gid == "G05":
            axs[2].plot(xs, y("gpr_test"), "s", ms=3.5, color=CAT[2], label="G05 misfit to independent GPR", markeredgecolor="white", markeredgewidth=0.5)
    axs[0].set_title("(a) glacier-wide SMB $-$ EXP01", loc="left"); axs[0].set_ylabel("m i.e. yr$^{-1}$"); axs[0].set_ylim(-0.15, 0.15)
    axs[1].set_title("(b) ELA $-$ EXP01", loc="left"); axs[1].set_ylabel("m")
    axs[2].set_title("(c) thickness", loc="left"); axs[2].set_ylabel("m"); axs[2].set_ylim(0, 200)
    for ax in axs:
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_xticks(xs); ax.set_xticklabels([e[3:] for e in exps], fontsize=5.5, rotation=90)
        ax.set_xlabel("experiment"); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=6.5); axs[2].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_real_sensitivity.pdf")
    rows = []
    for e in exps:
        cells = [e]
        for gid in ("G01", "G05"):
            r = res[gid][e]
            if r is None:
                cells += ["pending"] + [""] * (5 if gid == "G01" else 6); continue
            cells += [f"{r['dx']:g}", f"{r['tau']:.2f}", f"{r['vel_rmse']:.1f}", f"{r['thk']:.0f}", fmt(r["smb"], 2, True), f"{r['ela']:.0f}"]
            if gid == "G05":
                cells.append(fmt(r.get("gpr_test"), 0))
        rows.append(cells)
    tex_table("tab_real_sensitivity.tex",
              ["", r"$\Delta x$", r"$\tau_{\rm ref}$", r"$u_s$ RMSE", r"$\overline{H}$", r"$\overline{\rm SMB}$", "ELA",
               r"$\Delta x$", r"$\tau_{\rm ref}$", r"$u_s$ RMSE", r"$\overline{H}$", r"$\overline{\rm SMB}$", "ELA", "indep. GPR"],
              rows, "l|rrrrrr|rrrrrrr")


def glob_results():
    exps = ["EXP01", "EXP02"] + list(GLOB)
    res = {gid: {e: real_metrics(e, gid) for e in exps} for gid in REAL}
    exps = [e for e in exps if any(res[g][e] for g in REAL)]        # experiments not started anywhere are left out
    STATS["glob"] = {g: {e: {k: v for k, v in (r or {}).items() if k != "profile"} for e, r in d.items()} for g, d in res.items()}
    rows = []
    for gid in REAL:
        for e in exps:
            r = res[gid][e]
            lab = {"EXP01": "EXP01", "EXP02": "EXP02 (raw data)"}.get(e) or f"{e} ({GLOB[e][1]} vel.)"
            if r is None:
                rows.append([gid if e == "EXP01" else "", lab, "pending"] + [""] * 7); continue
            rows.append([gid if e == "EXP01" else "", lab, f"{r['vel_rmse']:.1f} ({r['vel_mean']:.0f})", f"{r['reg']:g}", f"{r['thk']:.0f}",
                         fmt(r.get("gpr_test"), 0), fmt(r["smb"], 2, True), f"{r['dhdt_model']:+.2f} / {r['dhdt_obs']:+.2f}",
                         fmt(r["ela"], 0), f"{r['unct_smb']:.2f}"])
        rows.append([r"\midrule"] if gid != REAL[-1] else None)
    rows = [r for r in rows if r is not None]
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r" & & $u_s$ RMSE (mean) & $\lambda$ & $\overline{H}$ & indep.\ GPR & $\overline{\rm SMB}$ & $\overline{{\rm d}h/{\rm d}t}$ mod / obs & ELA & $\overline{\sigma}_{\rm SMB}$ \\",
             r"\midrule"]
    lines += [r[0] if r == [r"\midrule"] else " & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_glob.tex"), "w").write("\n".join(lines) + "\n"); print("table tab_glob.tex")

    fig, axs = plt.subplots(2, 3, figsize=(TW, 4.6))
    sty = {"EXP01": (INK, "-", 1.8), "EXP02": (CAT[4], "-", 1.4), "EXP16": (CAT[1], "-", 1.1), "EXP17": (CAT[0], "-", 1.1), "EXP18": (CAT[2], "--", 1.1),
           "EXP19": (CAT[6], ":", 1.1), "EXP20": (CAT[3], "-.", 1.1)}
    for ax, gid in zip(axs.ravel(), REAL):
        src = native("EXP01", gid); zs = src.DEM.values[ice_of(src, gid)]
        for e in exps:
            r = res[gid][e]
            if r is None:
                continue
            c, ls, lw = sty[e]
            ax.plot(r["profile"][1], r["profile"][0], color=c, ls=ls, lw=lw,
                    label={"EXP01": "EXP01 (site data)", "EXP02": "EXP02 (raw data)"}.get(e) or f"{e}: {GLOB[e][1]} vel.")
        ax.axvline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(np.nanpercentile(zs, 0.5), np.nanpercentile(zs, 99.5))
        ax.set_title(gid, loc="left"); ax.set_xlabel("SMB (m i.e. yr$^{-1}$)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("elevation (m a.s.l.)"); axs[1, 0].set_ylabel("elevation (m a.s.l.)")
    leg = {}
    for ax in axs.ravel():                              # union over panels: some experiments are missing on some glaciers
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            leg.setdefault(ll, hh)
    l = sorted(leg); h = [leg[k] for k in l]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=min(len(h), 4), frameon=False)
    save(fig, "fig_glob_profiles.pdf")


# ── 7b. SMB profiles of every experiment, grouped by perturbation type ───────────────────────────────────────
EXP_GROUPS = [("thickness (EXP03--07)", range(3, 8)), ("velocity (EXP08--13)", range(8, 14)),
              ("resolution (EXP14--15) and raw data (EXP02)", [14, 15, 2])]


def profile_change(e, gid):
    """RMS difference (m i.e./yr) between the family-mean SMB profile of e and that of EXP01, on the EXP01 nodes
    inside the glacier's elevation range (0.5-99.5 %)."""
    o, r = export(e, gid), export("EXP01", gid)
    if o is None:
        return float("nan")
    src = native("EXP01", gid); zs = src.DEM.values[ice_of(src, gid)]
    lo, hi = np.nanpercentile(zs, [0.5, 99.5])
    z, p = o.z.values, o.SMB_PROFILE.values; ok = np.isfinite(p)
    z0, p0 = r.z.values, r.SMB_PROFILE.values
    k = np.isfinite(p0) & (z0 >= max(lo, z[ok].min())) & (z0 <= min(hi, z[ok].max()))
    return float(np.sqrt(np.mean((np.interp(z0[k], z[ok], p[ok]) - p0[k]) ** 2)))


def gpr_change(e, gid):
    """RMS relative change (%) of the GPR thickness the DA actually receives (all profile cells), against EXP01;
    None when the glacier has no GPR profiles."""
    a, b = (os.path.join(CX, "data", f"{x}_{gid}_da.nc") for x in (e, "EXP01"))
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    A, B = xr.open_dataset(a), xr.open_dataset(b)
    obs = lambda D: np.fmax(np.fmax(D.thkobs.values, D.thkobs_val.values), D.thkobs_test.values) if "thkobs" in D else None  # noqa: E731
    oa, ob = obs(A), obs(B)
    if oa is None or ob is None or oa.shape != ob.shape:
        return None
    m = np.isfinite(oa) & np.isfinite(ob) & (ob > 20)
    return float(100 * np.sqrt(np.mean((oa[m] / ob[m] - 1) ** 2))) if m.any() else None


def vel_change(e, gid):
    """RMS of |u_e - u_EXP01| / |u_EXP01| (%) over ice with |u| > 5 m/yr: magnitude and direction noise and bias together."""
    ref, ds = native("EXP01", gid), native(e, gid)
    if ds is None or ds.VX.shape != ref.VX.shape:
        return None
    u0 = np.hypot(ref.VX.values, ref.VY.values)
    du = np.hypot(ds.VX.values - ref.VX.values, ds.VY.values - ref.VY.values)
    m = ice_of(ref, gid) & (u0 > 5) & np.isfinite(du)
    return float(100 * np.sqrt(np.mean((du[m] / u0[m]) ** 2)))


def response_row(axs, gids):
    """Top row: input change of each experiment (x) against the change of the SMB profile (y, common log axis)."""
    out = {}
    for gid, c in zip(gids, (CAT[0], CAT[1], CAT[2])):
        mk = dict(ms=5, color=c, markeredgecolor="white", markeredgewidth=0.5, ls="none")
        for n in range(3, 8):
            e = f"EXP{n:02d}"; x = gpr_change(e, gid); y = profile_change(e, gid)
            x = 0.0 if x is None and gid == "G01" else x          # G01: no GPR -> pure repeats of EXP01
            if x is None or not np.isfinite(y):
                continue
            axs[0].plot(x, y, "o", **mk, label=gid if n == 3 else None)
            axs[0].annotate(f"{n:02d}", (x, y), xytext=(3, 2 if n % 2 else -7), textcoords="offset points", fontsize=5.5, color=c)
            out[(gid, e)] = (x, y)
        for n in range(8, 14):
            e = f"EXP{n:02d}"; x = vel_change(e, gid); y = profile_change(e, gid)
            if x is None or not np.isfinite(y):
                continue
            axs[1].plot(x, y, "o", **mk, label=gid if n == 8 else None)
            axs[1].annotate(f"{n:02d}", (x, y), xytext=(3, 2 if n % 2 else -7), textcoords="offset points", fontsize=5.5, color=c)
            out[(gid, e)] = (x, y)
        cats = []
        for n in (15, 14, 2):
            e = f"EXP{n:02d}"; y = profile_change(e, gid)
            if not np.isfinite(y):
                continue
            m = json.load(open(os.path.join(CX, "data", f"{e}_{gid}.json")))
            cats.append((e, m, y))
        for k, (e, m, y) in enumerate(cats):
            xk = {"EXP15": 0, "EXP14": 1, "EXP02": 2}[e] + 0.12 * (gids.index(gid) - (len(gids) - 1) / 2)
            axs[2].plot(xk, y, "o", **mk, label=gid if k == 0 else None)
            note = (f"{m['dx_igm']:g} m" if e != "EXP02" else f"$u_s$ {100 * m['n_vel_obs_igm'] / m['n_ice_cells_igm']:.0f}%")
            axs[2].annotate(note, (xk, y), xytext=(4, -1), textcoords="offset points", fontsize=5.5, color=c)
            out[(gid, e)] = (xk, y)
    # repeat-run scatter: runs whose inputs are identical to EXP01 for this method (G01 EXP03-07: no GPR; S02 EXP15)
    floor = max(profile_change(e, g) for e, g in [(f"EXP{n:02d}", "G01") for n in range(3, 8)] + [("EXP15", "S02")])
    for ax in axs:
        ax.axhspan(0.01, floor, color=NEUTRAL, zorder=0, lw=0)
    axs[2].text(2.85, floor * 0.8, "repeat-run scatter\n(identical inputs)", ha="right", va="top", fontsize=5.5, color=MUTED)
    axs[0].set_xlabel("GPR thickness change (RMS, %)", fontsize=7)
    axs[1].set_xlabel(r"velocity change $|\Delta\mathbf{u}_s|/|\mathbf{u}_s|$ (RMS, %)", fontsize=7)
    axs[2].set_xticks([0, 1, 2]); axs[2].set_xticklabels(["EXP15\n(grid)", "EXP14\n(grid)", "EXP02\n(raw data)"], fontsize=6.5)
    axs[2].set_xlim(-0.5, 2.9)
    axs[0].set_ylabel("SMB profile change vs EXP01\n(RMS, m i.e. yr$^{-1}$)")
    for ax, t in zip(axs, ("(a) thickness: GPR observations", "(b) surface velocity", "(c) resolution and raw data")):
        ax.set_yscale("log"); ax.set_ylim(0.01, 30); ax.set_title(t, loc="left", fontsize=7.5)
        ax.grid(axis="y", color=GRID, lw=0.5, which="major"); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    if len(gids) > 1:
        axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    return out


def profile_grid(gids, name, xlims=None):
    """Top row: what changed and how much the profile moved; then one row of profiles per glacier, one column per
    experiment group, EXP01 (wide grey) in every panel."""
    fig, axs = plt.subplots(len(gids) + 1, 3, figsize=(TW, 2.55 * (len(gids) + 1) + 0.3), squeeze=False,
                            gridspec_kw={"height_ratios": [0.85] + [1] * len(gids)})
    STATS.setdefault("profile_response", {}).update({f"{g} {e}": v for (g, e), v in response_row(axs[0], gids).items()})
    letters = "defghijkl"
    for i, gid in enumerate(gids):
        ref = export("EXP01", gid); src = native("EXP01", gid)
        zs = src.DEM.values[ice_of(src, gid)]
        for j, (title, rng) in enumerate(EXP_GROUPS):
            ax = axs[i + 1, j]
            m1 = json.load(open(os.path.join(CX, "data", f"EXP01_{gid}.json")))
            lab1 = "EXP01" if j < 2 else f"EXP01: {m1['dx_native']:g} m" + (f" (model {m1['dx_igm']:g} m)" if m1["dx_igm"] != m1["dx_native"] else "")
            ax.plot(ref.SMB_PROFILE.values, ref.z.values, color="#c3c2b7", lw=4.0, label=lab1, zorder=1,
                    solid_capstyle="round")         # wide, underneath: the experiments stay visible where they coincide
            k = 0
            for n in rng:
                e = f"EXP{n:02d}"; o = export(e, gid)
                if o is None:
                    continue
                lab = e
                if n in (2, 14, 15):                # grid (native -> model) and velocity coverage of the odd ones
                    m = json.load(open(os.path.join(CX, "data", f"{e}_{gid}.json")))
                    lab = (f"{e}: {m['dx_native']:g} m" + (f" (model {m['dx_igm']:g} m)" if m["dx_igm"] != m["dx_native"] else "")
                           + (f", $u_s$ on {100 * m['n_vel_obs_igm'] / m['n_ice_cells_igm']:.0f}%" if m["n_vel_obs_igm"] < 0.95 * m["n_ice_cells_igm"] else ""))
                ax.plot(o.SMB_PROFILE.values, o.z.values, color=CAT[k % len(CAT)], lw=1.1, label=lab,
                        ls="--" if n == 2 else "-")
                k += 1
            ax.axvline(0, color=GRID, lw=0.8, zorder=0)
            ax.set_ylim(np.nanpercentile(zs, 0.5), np.nanpercentile(zs, 99.5))
            if xlims and gid in xlims:
                ax.set_xlim(*xlims[gid])
            short = title.split(" (")[0] if "raw" not in title else "resolution and raw data"
            ax.set_title(f"({letters[3 * i + j]}) {gid}: {short}", loc="left", fontsize=7.5)
            ax.legend(frameon=False, fontsize=5.8, loc="best", handlelength=1.6, labelspacing=0.25)
            if i == len(gids) - 1:
                ax.set_xlabel("SMB (m i.e. yr$^{-1}$)")
            if j == 0:
                ax.set_ylabel("elevation (m a.s.l.)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.tight_layout(h_pad=1.2)
    save(fig, name)


def experiment_profiles():
    profile_grid(["S02"], "fig_s02_profiles.pdf", xlims={"S02": (-8, 4)})
    profile_grid(["G01", "G05"], "fig_real_exp_profiles.pdf")


# ── 8. progress ───────────────────────────────────────────────────────────────────────────────────────────────
def progress():
    st = json.load(open(os.path.join(CX, "results", "batch_state.json")))
    done = [k for k, v in st.items() if v.get("status") == "done"]
    hours = [st[k]["hours"] for k in done]
    q = [ln.split("#")[0].split() for ln in open(os.path.join(CX, "tools", "queue.txt"))]
    q = [f"{a[0]}_{a[1]}_{a[2]}" for a in q if a]
    real = [k for k in done if "_G0" in k]
    edge = [k for k in real if not (31 <= json.load(open(os.path.join(CX, "results", k, "state.json")))["init_arrhenius"] <= 118)]
    left = len(q) - len(done)
    rate = float(np.mean([st[k]["hours"] for k in real])) if real else float("nan")
    macros = {"NDone": len(done), "NQueue": len(q), "NLeft": left, "LaneHours": f"{np.sum(hours):.0f}",
              "NReal": len(real), "NRealEdge": len(edge),
              "HoursLeft": f"{(sum(st.get(k, {}).get('hours', 0) for k in []) + left * rate) / 2:.0f}"}
    open(os.path.join(TAB, "macros.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print("macros", macros)
    STATS["progress"] = {"done": len(done), "queue": len(q), "running": [k for k, v in st.items() if v.get("status") == "running"],
                         "lane_hours": float(np.sum(hours)),
                         "hours_by_glacier": {g: float(np.mean([st[k]["hours"] for k in done if f"_{g}_" in k] or [np.nan]))
                                              for g in ["S01", "S02"] + REAL}}


if __name__ == "__main__":
    os.chdir(CX)
    for f in (glaciers, synthetic_data, perturbations, coverage, synthetic_results, twin_test, gpr_effect, real_results,
              real_sensitivity, glob_results, experiment_profiles, progress):
        f()
    json.dump(STATS, open(os.path.join(HERE, "stats.json"), "w"), indent=1, default=float)
    print(json.dumps(STATS["progress"], indent=1))
