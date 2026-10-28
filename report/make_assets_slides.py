#!/usr/bin/env python3
"""
PNG figures for the results slides (final SMB phase only, results/*/smb_G_bands_geod), at slide size and slide fonts.
Report figures 4, 7 and 8 are redrawn with make_assets.py (export patched by make_assets_final.py), wider and with
scaled fonts; the S01 figures are drawn here. The S01 twin / true-law runs are tools/smb_rerun.py tags G_twin and
G_truth (member slid_0.4519, final settings).

    python report/make_assets_slides.py      # -> report/figures_slides/*.png, report/stats_slides.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_assets_final as mf   # noqa: E402  (patches make_assets.export to the final exports)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import xarray as xr              # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import AutoLocator, MaxNLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ma, ms = mf.ma, mf.ms
CX = ma.CX
OUT = os.path.join(HERE, "figures_slides")
os.makedirs(OUT, exist_ok=True)
BG = "#F7F6F2"                               # slide background
INK, INK2, MUTED, GRID, CAT = ma.INK, ma.INK2, ma.MUTED, ma.GRID, ma.CAT
FINAL = CAT[0]
DPI = 150
W = 12.0                                     # figure width (in) of a full-width slide figure
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "legend.facecolor": BG})
SMALL = {"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
         "legend.fontsize": 12}
ST = {}


def png(fig, name, bbox="tight"):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches=bbox)
    print("png", name)


def scale_fonts(fig, k):
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * k)


def tidy(fig, k=1.7):
    """Slide fonts (x k) and fewer numeric ticks, so the larger labels do not collide."""
    scale_fonts(fig, k)
    for ax in fig.axes:
        if ax.get_ylabel().startswith("SMB profile change"):
            ax.set_ylabel("profile change vs EXP01\n(RMS, m i.e. yr$^{-1}$)")
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear" and isinstance(axis.get_major_locator(), AutoLocator):
                axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


SIZES = {"fig_s02_profiles.pdf": (W, 6.3), "fig_real_exp_profiles.pdf": (W, 8.8), "fig_glob_profiles.pdf": (W, 6.6)}


def save_scaled(fig, name):
    """make_assets figures: slide size and fonts, then re-layout. The G01/G05 grid is cut: its first row is
    redrawn on its own (report_figures), the profile rows are saved here."""
    if name in SIZES:
        fig.set_size_inches(*SIZES[name])
    tidy(fig)
    if name == "fig_glob_profiles.pdf":
        fig.tight_layout(rect=(0, 0.13, 1, 1))
    else:
        fig.tight_layout(h_pad=1.0)
    if name == "fig_real_exp_profiles.pdf":
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        axs = [a for a in fig.axes if a.get_subplotspec().rowspan.start >= 1]
        bb = Bbox.union([a.get_tightbbox(r) for a in axs]).transformed(fig.dpi_scale_trans.inverted())
        png(fig, "fig_g01_g05_profiles.png", bbox=bb.padded(0.05))
    else:
        png(fig, name.replace(".pdf", ".png"))
    plt.close(fig)


# ── report figures 4, 7, 8 at slide size ────────────────────────────────────────────────────────────────────────
def report_figures():
    ma.TW, ma.save = W, save_scaled
    ma.profile_grid(["S02"], "fig_s02_profiles.pdf", xlims={"S02": (-8, 4)})
    ma.profile_grid(["G01", "G05"], "fig_real_exp_profiles.pdf")
    fig, axs = plt.subplots(1, 3, figsize=(W, 4.6))
    ma.response_row(axs, ["G01", "G05"])
    tidy(fig); fig.tight_layout(w_pad=0.8)
    png(fig, "fig_g01_g05_response.png"); plt.close(fig)
    ma.glob_results()


# ── S01 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def s01_profile():
    src = ma.native("EXP01", "S01")
    zr = np.nanpercentile(src.DEM.values[ma.ice_of(src, "S01")], [1, 99])
    zz = np.linspace(zr[0], zr[1], 200)
    o = mf.export_final("EXP01", "S01")
    with plt.rc_context(SMALL):
        fig, (ax, ae) = plt.subplots(1, 2, figsize=(7.4, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
        for k in range(o.sizes["member"]):
            ax.plot(o.MEMBER_SMB_PROFILE.values[k], o.z.values, color=FINAL, lw=0.8, alpha=0.45,
                    label="family members" if k == 0 else None)
        ax.plot(o.SMB_PROFILE.values, o.z.values, color=FINAL, lw=2.6, label="inferred (family mean)")
        ax.plot(ms.law(zz), zz, color=INK, lw=1.6, ls=(0, (4, 2)), label="prescribed law")
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
        ax.set_ylim(*zr); ax.set_xlim(-8, 6); ax.legend(frameon=False, loc="lower right"); spines(ax)
        zf = np.arange(np.ceil(zr[0] / 10) * 10, zr[1], 10.0)
        ae.plot(np.interp(zf, o.z.values, o.SMB_PROFILE.values) - ms.law(zf), zf, color=FINAL, lw=2.6)
        ae.axvline(0, color=INK, lw=1)
        ae.set_xlim(-2, 2); ae.set_xlabel("inferred $-$ law\n(m i.e. yr$^{-1}$)"); spines(ae)
        fig.tight_layout(w_pad=1.0)
        png(fig, "fig_s01_profile.png"); plt.close(fig)


def s01_experiments():
    rows = []
    for e in mf.S01_EXPS:
        s = os.path.join(CX, "input_data", e, f"{e}_S01.nc")
        if os.path.exists(ms.new_path(e, "S01")):
            rows.append((e, ms.s01_profile_rmse(ms.new_path(e, "S01"), s)))
    ST["s01_rmse"] = dict(rows)
    with plt.rc_context(SMALL):
        fig, (ax, a15) = plt.subplots(1, 2, figsize=(W, 4.2), gridspec_kw={"width_ratios": [6, 1]})
        main = [r for r in rows if r[0] != "EXP15"]
        x = np.arange(len(main))
        cols = [CAT[1] if r[0] == "EXP05" else FINAL for r in main]
        ax.bar(x, [r[1] for r in main], 0.62, color=cols)
        for xi, (_, v) in zip(x, main):
            ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=11, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([r[0][3:] for r in main]); ax.set_xlabel("experiment")
        ax.set_ylabel("profile RMSE against the law\n(m i.e. yr$^{-1}$)"); ax.set_ylim(0, 2.5)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); spines(ax)
        r15 = [r for r in rows if r[0] == "EXP15"]
        if r15:
            a15.bar([0], [r15[0][1]], 0.62, color=CAT[1])
            a15.text(0, r15[0][1] + 0.1, f"{r15[0][1]:.1f}", ha="center", fontsize=11, color=INK2)
            a15.set_xticks([0]); a15.set_xticklabels(["15"]); a15.set_xlabel("100 m grid")
            a15.set_xlim(-0.8, 0.8); a15.set_ylim(0, 6); a15.grid(axis="y", color=GRID, lw=0.8); a15.set_axisbelow(True)
            spines(a15)
        fig.tight_layout(w_pad=1.5)
        png(fig, "fig_s01_experiments.png"); plt.close(fig)


def s01_geometry():
    """Why S01 cannot be fitted exactly: assimilated vs true thickness, and what the true law does in our model."""
    src = ma.native("EXP01", "S01")
    o = mf.export_final("EXP01", "S01")
    thk = ma.on_grid(o, src, "THK"); tt = src.THK.values.astype(float); dem = src.DEM.values.astype(float)
    ice = ma.ice_of(src, "S01")
    x, y = src.x.values / 1000, src.y.values
    jc = int(np.argmin(np.abs(y - 500)))                            # centre line
    ic = int(np.argmin(np.abs(src.x.values - 2500)))                # mid-glacier cross section
    m = ice & np.isfinite(thk)
    ST["s01_thk"] = {"mean_da": float(np.mean(thk[m])), "mean_true": float(np.mean(tt[m])),
                     "centre_da": float(np.nanmean(thk[jc][ice[jc]])), "centre_true": float(np.nanmean(tt[jc][ice[jc]]))}
    d = os.path.join(CX, "results", "EXP01_S01_cnn", "smb_G_truth", "slid_0.4519", "smb_inference")
    res = np.load(os.path.join(d, "thk_optimized.npy")) - np.load(os.path.join(d, "thk_observed.npy"))
    obs = np.load(os.path.join(d, "thk_observed.npy"))
    res = np.where(obs > 1, res, np.nan)
    g = xr.open_dataset(os.path.join(CX, "data", "EXP01_S01.nc"))      # model grid (50 m)
    rows = np.nanmean(res[:, 10:50], 1)                             # trunk, 0.5-2.5 km: mean per row across flow
    ST["s01_truth_residual"] = {"rms": float(np.sqrt(np.nanmean(res ** 2))), "min": float(np.nanmin(res)), "max": float(np.nanmax(res)),
                                "trunk_row_means": [float(v) for v in rows if np.isfinite(v)]}
    # mass balance of the outline under the true law: mean SMB on the mask vs mean thickness change of the twin target
    mk = g.icemask.values > 0.5
    tw = os.path.join(CX, "results", "EXP01_S01_cnn", "smb_G_twin", "slid_0.4519", "smb_inference", "thk_twin_obs.npy")
    h0 = xr.open_dataset(os.path.join(CX, "results", "EXP01_S01_cnn", "family", "slid_0.4519", "geology-optimized.nc")).thk.values
    if os.path.exists(tw):
        ST["s01_outline"] = {"law_mean_smb_on_mask": float(np.load(os.path.join(d, "smb_field.npy"))[mk].mean()),
                             "model_mean_dhdt_on_mask": float((np.load(tw) - h0)[mk].mean() / 10.0)}

    with plt.rc_context(SMALL):
        fig = plt.figure(figsize=(9.2, 6.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1.25, 1], hspace=0.55, wspace=0.28)
        ax = fig.add_subplot(gs[0, 0])
        zs = np.where(ice[jc], dem[jc], np.nan)
        ax.fill_between(x, zs - np.where(ice[jc], tt[jc], np.nan), zs, color=INK, alpha=0.12, lw=0, label="true ice")
        ax.plot(x, zs, color=INK, lw=1.2, label="surface")
        ax.plot(x, zs - np.where(ice[jc], tt[jc], np.nan), color=INK, lw=1.6, ls=(0, (4, 2)), label="true bed")
        ax.plot(x, zs - np.where(ice[jc], thk[jc], np.nan), color=CAT[1], lw=2.2, label="assimilated bed")
        ax.set_xlabel("distance along flow (km)"); ax.set_ylabel("elevation (m a.s.l.)")
        ax.set_title("(a) centre line", loc="left"); ax.legend(frameon=False, loc="upper right", fontsize=11); spines(ax)
        ax = fig.add_subplot(gs[0, 1])
        zc = dem[:, ic]; ic_ice = ice[:, ic]
        ax.fill_between(y, np.where(ic_ice, zc - tt[:, ic], np.nan), np.where(ic_ice, zc, np.nan), color=INK, alpha=0.12, lw=0)
        ax.plot(y, np.where(ic_ice, zc, np.nan), color=INK, lw=1.2)
        ax.plot(y, np.where(ic_ice, zc - tt[:, ic], np.nan), color=INK, lw=1.6, ls=(0, (4, 2)))
        ax.plot(y, np.where(ic_ice, zc - thk[:, ic], np.nan), color=CAT[1], lw=2.2)
        ax.set_xlabel("across flow (m)"); ax.set_title("(b) cross section, 2.5 km", loc="left"); spines(ax)
        ax = fig.add_subplot(gs[1, :])
        ext = [float(g.x[0]) / 1000 - 0.025, float(g.x[-1]) / 1000 + 0.025, float(g.y[0]) - 25, float(g.y[-1]) + 25]
        im = ax.imshow(res, cmap=ma.DIV, vmin=-40, vmax=40, extent=ext, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xlabel("distance along flow (km)"); ax.set_ylabel("across (m)")
        ax.set_title("(c) model $-$ observed thickness after 10 yr, forced with the TRUE SMB", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("m"); cb.outline.set_visible(False)
        png(fig, "fig_s01_geometry.png"); plt.close(fig)


def s01_twin():
    root = os.path.join(CX, "results", "EXP01_S01_cnn")
    src = ma.native("EXP01", "S01")
    zr = np.nanpercentile(src.DEM.values[ma.ice_of(src, "S01")], [1, 99])
    zf = np.arange(np.ceil(zr[0] / 10) * 10, zr[1], 10.0)

    def load(tag):
        sd = os.path.join(root, f"smb_{tag}", ma.TWIN_MEMBER, "smb_inference")
        v = np.load(os.path.join(sd, "smb_vec.npy")).astype(float)
        z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
        p = (np.clip(1.0 - np.abs((zf - z0)[:, None] / dz - np.arange(v.size)), 0.0, None) * v).sum(-1)
        return dict(p=p, rmse=float(np.sqrt(np.mean((p - ms.law(zf)) ** 2))), data=np.load(os.path.join(sd, "data_history.npy")))

    real, twin, truth, twin0 = load("G_bands_geod"), load("G_twin"), load("G_truth"), load("G_twin_nogeod")
    ST["twin"] = {"member": ma.TWIN_MEMBER,
                  "twin_nogeod": {"rmse": twin0["rmse"], "data0": float(twin0["data"][0]), "data": float(twin0["data"][-1]), "it": int(twin0["data"].size)},
                  "real": {"rmse": real["rmse"], "data0": float(real["data"][0]), "data": float(real["data"][-1]), "it": int(real["data"].size)},
                  "twin": {"rmse": twin["rmse"], "data0": float(twin["data"][0]), "data": float(twin["data"][-1]), "it": int(twin["data"].size)},
                  "truth": {"rmse": truth["rmse"], "data": float(truth["data"][-1])}}
    with plt.rc_context(SMALL):
        fig, axs = plt.subplots(1, 2, figsize=(W, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
        ax = axs[0]
        ax.plot(twin0["p"], zf, color=CAT[2], lw=3.2, label=f"twin, band misfit only ({twin0['rmse']:.2f})")
        ax.plot(twin["p"], zf, color=CAT[4], lw=2.4, label=f"twin, + geodetic term ({twin['rmse']:.2f})")
        ax.plot(real["p"], zf, color=FINAL, lw=2.4, label=f"real S01 data ({real['rmse']:.2f})")
        ax.plot(ms.law(zf), zf, color=INK, lw=1.4, ls=(0, (4, 2)), label="prescribed law")
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
        ax.set_title("(a) inferred profile (RMSE vs law)", loc="left"); ax.legend(frameon=False, loc="lower right", fontsize=11)
        ax.set_ylim(*zr); ax.set_xlim(-8, 6); spines(ax)
        ax = axs[1]
        ax.plot(np.arange(1, real["data"].size + 1), real["data"], color=FINAL, lw=2.4, marker="o", ms=4, label="real S01 data")
        ax.axhline(truth["data"], color=INK2, lw=1.4, ls=(0, (4, 2)), label="true law on the real data (no fit)")
        ax.plot(np.arange(1, twin["data"].size + 1), twin["data"], color=CAT[4], lw=2.4, marker="o", ms=4, label="twin, + geodetic term")
        ax.plot(np.arange(1, twin0["data"].size + 1), twin0["data"], color=CAT[2], lw=2.4, marker="o", ms=4, label="twin, band misfit only")
        ax.set_xlabel("L-BFGS iteration"); ax.set_ylabel("band-mean thickness misfit (m)")
        ax.set_title("(b) data misfit", loc="left"); ax.set_ylim(0, None); ax.legend(frameon=False, loc="upper right", fontsize=11)
        spines(ax)
        fig.tight_layout(w_pad=2.0)
        png(fig, "fig_s01_twin.png"); plt.close(fig)


def main():
    os.chdir(CX)
    s01_profile(); s01_experiments(); s01_geometry(); s01_twin(); report_figures()
    json.dump(ST, open(os.path.join(HERE, "stats_slides.json"), "w"), indent=1, default=float)
    print(json.dumps(ST, indent=1, default=float))


if __name__ == "__main__":
    main()
