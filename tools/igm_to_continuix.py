#!/usr/bin/env python3
"""
Export a family of igm-smb-inference runs to one ContinuIX result file
(EXP##_<GID>_method##.nc) on the native ContinuIX grid.

Per family member (one tau_ref value, DA + SMB inference):
  topg       = usurf_DA - thk_DA                       (geology-optimized.nc)
  H1         = max(usurfstart - topg, 0) * icemask     (forward-run start)
  H2         = thk_optimized.npy                       (simulated end of window)
  dHdt_mod   = (H2 - H1) / dt
  SMB        = SMB(z) evaluated at the mid-window surface. On the IGM grid
               zbar = topg + (H1+H2)/2; on the native grid the observed
               DEM + DHDT ((t1+t2)/2 - t_dem). The profile is piecewise linear
               and z(t) nearly linear, so SMB(zbar) ~ the window-mean SMB.
               A member run with inversion.method=field (smb_field2d.npy) has a static
               SMB(x, y) instead: it is bilinearly interpolated to the native grid.
  FDIV       = SMB - dHdt_mod on the IGM grid (the transient, window-averaged
               flux divergence; paper Fig. 10b), bilinearly interpolated to
               the native grid.
  DHDT       = SMB - FDIV  (the modelled thinning rate: reported as "modified")

Ensemble: members whose final DA velocity misfit is <= 1.5 x the family
minimum are admissible (paper Sect. 3.3). SMB/FDIV/THK = admissible mean. ALL
members (admissible or not) are stored with their profile, tau_ref, A, final
velocity misfit, ELA and admissible flag, and a Fig. 7-style plot is written.

Uncertainty (1 sigma, per native pixel), UNCT_SMB and UNCT_FDIV:
    sqrt( family spread^2 + representation^2 + sigma_DHDT^2 )
  family spread   std over the admissible members (0 for a single member)
  representation  RMS of (H_sim - H_obs)/dt at the end of the window in the
                  pixel's 50 m elevation band, pooled over the admissible
                  members: what a single SMB(z) cannot explain
  sigma_DHDT      provided DHDT uncertainty (UNCT_DHDT field, else the scalar in
                  the DHDT `uncertainty` attribute, else 0 = not provided)
UNCT_THK = sqrt(family spread^2 + provided THK sigma^2).
The three UNCT_SMB components are also stored (COMP_*).

All mass terms in m ice equivalent (density 917 kg m-3, constant).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuix_to_igm import (fill_missing_dhdt, fill_nearest, ice_mask_native,  # noqa: E402
                              provided_sigma)

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_ICE = 917.0
CUTOFF = 1.5
REPR_BAND_M = 50.0


def final_velsurf(run_dir):
    p = os.path.join(run_dir, "costs.dat")
    if not os.path.exists(p):
        return np.nan
    lines = [ln.split() for ln in open(p) if ln.strip()]
    hdr = lines[0]
    starts = [i for i, ln in enumerate(lines) if ln[0] == hdr[0]]
    body = lines[starts[-1] + 1:]          # IGM appends: take the last block
    return float(body[-1][hdr.index("velsurf")]) if body else np.nan


def hat_profile(z, smb_vec, z_min, dz):
    """Exactly the model's basis (core/smb.py: update_smb_profile)."""
    t = (z - z_min) / dz
    k = np.arange(smb_vec.size)
    w = np.clip(1.0 - np.abs(t[..., None] - k), 0.0, None)
    return (w * smb_vec).sum(axis=-1)


def clamped_profile(z, smb_vec, z_min, dz):
    """Native-grid evaluation: linear between nodes, constant beyond the end
    nodes (the model's hat basis tapers to 0 there, which only matters for
    native pixels outside the IGM-grid elevation range)."""
    nodes = z_min + dz * np.arange(smb_vec.size)
    return np.interp(z, nodes, smb_vec)


def profile_ela(nodes, smb_vec, z_lo, z_hi, step=5.0):
    """Lowest elevation inside [z_lo, z_hi] where the profile crosses zero from
    below (same definition as the paper figures); NaN if it does not cross."""
    zf = np.arange(z_lo, z_hi + step, step)
    sf = np.interp(zf, nodes, smb_vec, left=np.nan, right=np.nan)
    ok = np.isfinite(sf)
    zf, sf = zf[ok], sf[ok]
    k = np.where((sf[:-1] < 0) & (sf[1:] >= 0))[0]
    if not k.size:
        return np.nan
    i = k[0]
    return float(zf[i] - sf[i] * (zf[i + 1] - zf[i]) / (sf[i + 1] - sf[i]))


def load_member(run_dir, igm_in, meta):
    sd = os.path.join(run_dir, "smb_inference")
    geo = xr.open_dataset(os.path.join(run_dir, "geology-optimized.nc"))
    usurf_da, thk_da = geo.usurf.values.astype(float), geo.thk.values.astype(float)
    if geo.y.values[0] > geo.y.values[-1]:
        usurf_da, thk_da = usurf_da[::-1], thk_da[::-1]
    topg = usurf_da - thk_da
    icem = igm_in.icemask.values > 0.5
    H1 = np.clip(igm_in.usurfstart.values - topg, 0, None) * icem
    H2 = np.load(os.path.join(sd, "thk_optimized.npy")).astype(float)
    smb_vec = np.load(os.path.join(sd, "smb_vec.npy")).astype(float)
    z_min, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
    dt = meta["dt"]
    dhdt_mod = (H2 - H1) / dt
    zbar = topg + 0.5 * (H1 + H2)
    f2d = os.path.join(sd, "smb_field2d.npy")
    field = np.load(f2d).astype(float) if os.path.exists(f2d) else None     # method 'field': static SMB(x, y)
    smb_c = field if field is not None else hat_profile(zbar, smb_vec, z_min, dz)
    fdiv_c = np.where(icem, smb_c - dhdt_mod, np.nan)
    resid = np.where(icem, H2 - (igm_in.usurfinfer.values - topg), np.nan)
    nodes = z_min + dz * np.arange(smb_vec.size)
    zi = zbar[icem]
    return {
        "dir": run_dir, "velsurf": final_velsurf(run_dir),
        "slidingco": float(np.nanmean(geo.slidingco.values)) if "slidingco" in geo else np.nan,
        "arrhenius": float(np.nanmean(geo.arrhenius.values[geo.icemask.values > 0.5])) if "arrhenius" in geo else np.nan,
        "smb_vec": smb_vec, "z_min": z_min, "dz": dz, "smb_c": np.where(icem, smb_c, np.nan) if field is not None else None,
        "fdiv_c": fdiv_c, "thk_da": np.where(icem, thk_da, np.nan),
        "rms_geom_misfit": float(np.sqrt(np.nanmean(resid ** 2))),
        "smb_glacierwide_igm": float(np.nanmean(np.where(icem, smb_c, np.nan))),
        "ela": profile_ela(nodes, smb_vec, float(zi.min()), float(zi.max())),
        "z_cells": zi, "r_cells": (resid[icem] / dt),          # (H_sim - H_obs)/dt, m i.e./yr
    }


def to_native(field_c, xc, yc, xn, yn):
    """Bilinear interpolation of an IGM-grid field onto the native grid; NaNs
    are nearest-filled first so the ice margin keeps values."""
    f = fill_nearest(np.where(np.isfinite(field_c), field_c, np.nan))
    it = RegularGridInterpolator((yc, xc), f, method="linear", bounds_error=False, fill_value=None)
    Y, X = np.meshgrid(yn, xn, indexing="ij")
    return it(np.stack([Y, X], axis=-1))


def band_rms_curve(z, r, band=REPR_BAND_M, min_n=5):
    """RMS of r in `band`-m elevation bands (bands with < min_n cells take the
    nearest populated band). Returns (band centres, rms)."""
    b = np.floor(z / band).astype(int)
    ub = np.unique(b)
    n = np.array([(b == k).sum() for k in ub])
    rms = np.array([np.sqrt(np.mean(r[b == k] ** 2)) for k in ub])
    use = n >= min_n if (n >= min_n).any() else n > 0
    src = np.array([np.argmin(np.where(use, np.abs(ub - k), np.inf)) for k in ub])
    return (ub + 0.5) * band, rms[src]


def sigma_field(src, var, ice_n, window_yr, per_year):
    """Provided 1-sigma of `var` on the native (ascending-y) grid; zeros if not provided."""
    sig, how = provided_sigma(src, var, window_yr, per_year)
    if sig is None:
        return np.zeros(ice_n.shape), how
    if np.isscalar(sig):
        return np.full(ice_n.shape, float(sig)), how
    pos = np.isfinite(sig) & (sig > 0) & ice_n
    fill = float(np.median(sig[pos])) if pos.any() else 0.0
    return np.where(pos, sig, fill), how


def fig7(path, title, meta, members, adm_flags, zp, prof, z_ice, hyps):
    """Fig. 7-style family plot: SMB(z) of every member (admissible solid, coloured by
    tau_ref; rejected dashed grey; best-velocity member bold), admissible-mean profile,
    ELA band of the admissible members, hypsometry on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LogNorm
    taus = np.array([m["slidingco"] for m in members])
    vs = np.array([m["velsurf"] for m in members])
    ibest = int(np.nanargmin(vs))
    adm = np.array(adm_flags, bool)
    norm = LogNorm(vmin=max(taus.min() * 0.9, 1e-3), vmax=taus.max() * 1.1 if taus.max() > taus.min() else taus.min() * 1.2)
    fig, (ax, ah) = plt.subplots(1, 2, figsize=(5.6, 4.4), gridspec_kw={"width_ratios": [4, 1], "wspace": 0.04}, sharey=True)
    elas = np.array([m["ela"] for m in members])
    if adm.any() and np.isfinite(elas[adm]).any():
        ax.axhspan(np.nanmin(elas[adm]), np.nanmax(elas[adm]), color="0.85", zorder=0, label="ELA range (admissible)")
    ax.axvline(0, color="0.4", lw=0.8, ls=":", zorder=1)
    for i, m in enumerate(members):
        lab = f"τref={m['slidingco']:.3f}, vs={m['velsurf']:.1f}"
        if adm[i]:
            ax.plot(prof[i], zp, "-", color=cm.viridis(norm(m["slidingco"])), lw=2.4 if i == ibest else 1.4,
                    zorder=4 if i == ibest else 3, label=lab + (" (best)" if i == ibest else ""))
        else:
            ax.plot(prof[i], zp, "--", color="0.6", lw=1.1, zorder=2, label=lab + " (rejected)")
    if adm.sum() > 1:
        ax.plot(np.nanmean(prof[adm], 0), zp, "-", color="k", lw=0.9, zorder=5, label="admissible mean")
    ax.set_ylim(z_ice[0] - 50, z_ice[1] + 50)
    ax.set_xlabel("SMB (m i.e. yr$^{-1}$)"); ax.set_ylabel("elevation (m a.s.l.)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=6, loc="best", framealpha=0.85)
    ax.set_title(title, fontsize=9)
    ah.barh(hyps[0], hyps[1], height=hyps[0][1] - hyps[0][0] if len(hyps[0]) > 1 else 50, color="0.7")
    ah.set_xlabel("ice cells", fontsize=8); ah.tick_params(labelleft=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="results/<EXP>_<GID>_<variant>")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--method", default="01")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plot-dir", default=None, help="Fig. 7 plot + family_summary.json (default: run root)")
    ap.add_argument("--members", nargs="*", default=None, help="explicit run dirs (default: family/slid_*)")
    ap.add_argument("--thk-sigma", type=float, default=None,
                    help="scalar thickness error (m) for UNCT_THK, e.g. the validation-GPR RMSE (default: provided THK sigma, "
                         "or none for an input built by gpr_input.py, whose DA never sees a provided thickness)")
    ap.add_argument("--thk-sigma-source", default="", help="description of --thk-sigma")
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    igm_in = xr.open_dataset(os.path.join(CX_DIR, "data", meta["igm_input"]))
    run_dirs = a.members or sorted(d for d in glob.glob(os.path.join(a.run_root, "family", "slid_*"))
                                   if os.path.isdir(d))
    members = []
    for d in run_dirs:
        if not os.path.exists(os.path.join(d, "smb_inference", "smb_vec.npy")):
            print(f"[export] skip {d}: no smb_vec.npy"); continue
        members.append(load_member(d, igm_in, meta))
    if not members:
        raise SystemExit("no completed family members")
    best = np.nanmin([m["velsurf"] for m in members])
    adm_flags = [bool(m["velsurf"] <= CUTOFF * best) for m in members]
    if not any(adm_flags):
        adm_flags = [True] * len(members)
    adm = [m for m, f in zip(members, adm_flags) if f]
    print(f"[export] {len(adm)}/{len(members)} admissible (velsurf <= {CUTOFF} x {best:.3f})")

    # native grid, worked in ascending y then restored to native orientation
    src = xr.open_dataset(meta["source_file"])
    ydesc = bool(src.y.values[1] < src.y.values[0])
    srcA = src.isel(y=slice(None, None, -1)) if ydesc else src
    xn, yn = srcA.x.values.astype(float), srcA.y.values.astype(float)
    ice_n = ice_mask_native(srcA, meta["gid"])
    dem_n = srcA.DEM.values.astype(float)
    dem_n = fill_nearest(np.where(np.isfinite(dem_n), dem_n, srcA.BED.values.astype(float)))
    dhdt_n, _ = fill_missing_dhdt(srcA.DHDT.values.astype(float), ice_n, dem_n)   # same gap fill as the converter
    zbar_n = dem_n + dhdt_n * (0.5 * (meta["t1"] + meta["t2"]) - meta["t_dem"])
    xc, yc = igm_in.x.values.astype(float), igm_in.y.values.astype(float)

    smb_s, fdiv_s, thk_s = [], [], []
    for m in adm:
        smb_s.append(to_native(m["smb_c"], xc, yc, xn, yn) if m["smb_c"] is not None
                     else clamped_profile(zbar_n, m["smb_vec"], m["z_min"], m["dz"]))
        fdiv_s.append(to_native(m["fdiv_c"], xc, yc, xn, yn))
        thk_s.append(np.clip(to_native(m["thk_da"], xc, yc, xn, yn), 0, None))
    smb_s, fdiv_s, thk_s = map(np.array, (smb_s, fdiv_s, thk_s))
    nadm = len(adm)
    spread = (lambda a_: a_.std(0, ddof=1)) if nadm > 1 else (lambda a_: np.zeros(a_.shape[1:]))
    smb, fdiv, thk = smb_s.mean(0), fdiv_s.mean(0), thk_s.mean(0)
    dhdt_mod = smb - fdiv
    sp_smb, sp_fdiv, sp_thk = spread(smb_s), spread(fdiv_s), spread(thk_s)

    # band representation error: pooled over admissible members, evaluated at the native mid-window surface
    zc = np.concatenate([m["z_cells"] for m in adm]); rc = np.concatenate([m["r_cells"] for m in adm])
    zb, rb = band_rms_curve(zc, rc)
    rep = np.interp(zbar_n, zb, rb)
    sig_dhdt, how_dhdt = sigma_field(srcA, "DHDT", ice_n, meta["dt"], per_year=True)
    if a.thk_sigma is not None:
        sig_thk, how_thk = np.where(ice_n, a.thk_sigma, 0.0), f"{a.thk_sigma:.1f} m: {a.thk_sigma_source}"
    elif "has_gpr" in meta:      # gpr_input.py input: the DA thickness never sees a provided THK, so its sigma is not ours
        sig_thk, how_thk = np.zeros(ice_n.shape), "none (no held-out thickness observation): family spread only"
    else:
        sig_thk, how_thk = sigma_field(srcA, "THK", ice_n, None, per_year=False)
    u_smb = np.sqrt(sp_smb ** 2 + rep ** 2 + sig_dhdt ** 2)
    u_fdiv = np.sqrt(sp_fdiv ** 2 + rep ** 2 + sig_dhdt ** 2)
    u_thk = np.sqrt(sp_thk ** 2 + sig_thk ** 2)
    nan = lambda a_: np.where(ice_n, a_, np.nan).astype(np.float32)  # noqa: E731

    # 1-D profiles of ALL members on a common node set; NaN outside a member's own nodes
    z0 = min(m["z_min"] for m in members); dz = members[0]["dz"]
    z1 = max(m["z_min"] + dz * (m["smb_vec"].size - 1) for m in members)
    zp = np.arange(z0, z1 + 0.5 * dz, dz)
    prof = np.array([np.where((zp >= m["z_min"] - 1e-6) & (zp <= m["z_min"] + dz * (m["smb_vec"].size - 1) + 1e-6),
                              clamped_profile(zp, m["smb_vec"], m["z_min"], m["dz"]), np.nan) for m in members])
    prof_adm = prof[np.array(adm_flags)]
    prof_sd = prof_adm.std(0, ddof=1) if nadm > 1 else np.full(zp.size, np.nan)

    def var(a_, long, units, extra=None):
        a_ = a_[::-1] if ydesc else a_
        at = {"long_name": long, "units": units, "grid_mapping": "spatial_ref"}
        at.update(extra or {})
        return (("y", "x"), a_, at)

    mi = "m i.e. yr-1"
    ens = f"admissible tau_ref family mean (n={nadm})"
    unct = "sqrt(family spread^2 + elevation-band representation error^2 + sigma_DHDT^2)"
    ds = xr.Dataset(
        {
            "SMB": var(nan(smb), "surface mass balance", mi, {"description": ("static 2-D SMB(x, y) inverted on a coarse node grid, bilinear to this grid; " if adm[0]["smb_c"] is not None
                                                                         else "window-mean SMB(z) evaluated at the mid-window surface; ") + ens}),
            "FDIV": var(nan(fdiv), "flux divergence", mi, {"description": "window-averaged transient flux divergence, SMB - dH/dt_model; " + ens}),
            "DHDT": var(nan(dhdt_mod), "elevation change (modified: modelled)", mi, {"description": "SMB - FDIV; the modelled thinning rate over the window"}),
            "THK": var(nan(thk), "ice thickness (modified: data assimilation)", "m i.e.", {"timestamp": f"{meta['t_da']:.2f}"}),
            "DENSITY": var(nan(np.full_like(smb, RHO_ICE)), "density", "kg m-3", {"description": "ice density applied throughout (no firn correction)"}),
            "UNCT_SMB": var(nan(u_smb), "uncertainty of SMB", mi, {"description": "1-sigma: " + unct}),
            "UNCT_FDIV": var(nan(u_fdiv), "uncertainty of FDIV", mi, {"description": "1-sigma: " + unct}),
            "UNCT_THK": var(nan(u_thk), "uncertainty of THK", "m i.e.", {"description": "1-sigma: sqrt(family spread^2 + thickness error^2); thickness error = " + how_thk}),
            "COMP_FAMILY_SPREAD_SMB": var(nan(sp_smb), "UNCT_SMB component: admissible-family spread", mi),
            "COMP_FAMILY_SPREAD_FDIV": var(nan(sp_fdiv), "UNCT_FDIV component: admissible-family spread", mi),
            "COMP_REPRESENTATION_ERROR": var(nan(rep), "UNCT component: RMS end-of-window geometry residual / dt in the elevation band", mi),
            "COMP_SIGMA_DHDT": var(nan(sig_dhdt), "UNCT component: provided DHDT uncertainty (0 = not provided)", mi),
            "SMB_PROFILE": (("z",), np.nanmean(prof_adm, 0).astype(np.float32), {"long_name": "inferred SMB(z) profile, admissible-family mean", "units": mi}),
            "UNCT_SMB_PROFILE": (("z",), prof_sd.astype(np.float32), {"long_name": "family spread of SMB(z)", "units": mi}),
            "MEMBER_SMB_PROFILE": (("member", "z"), prof.astype(np.float32), {"long_name": "SMB(z) of every family member (NaN outside its node range)", "units": mi}),
            "MEMBER_TAU_REF": (("member",), np.array([m["slidingco"] for m in members], np.float32), {"long_name": "Weertman reference basal shear stress tau_ref", "units": "MPa"}),
            "MEMBER_ARRHENIUS": (("member",), np.array([m["arrhenius"] for m in members], np.float32), {"long_name": "Arrhenius factor A (rate factor)", "units": "MPa-3 yr-1"}),
            "MEMBER_VELSURF": (("member",), np.array([m["velsurf"] for m in members], np.float32), {"long_name": "final DA surface-velocity misfit cost"}),
            "MEMBER_ELA": (("member",), np.array([m["ela"] for m in members], np.float32), {"long_name": "ELA: lowest zero crossing of the profile inside the glacier elevation range", "units": "m a.s.l."}),
            "MEMBER_ADMISSIBLE": (("member",), np.array(adm_flags, np.int8), {"long_name": f"1 if velocity misfit <= {CUTOFF} x best"}),
            "MEMBER_RMS_GEOMETRY_MISFIT": (("member",), np.array([m["rms_geom_misfit"] for m in members], np.float32), {"long_name": "RMS end-of-window thickness residual (model grid)", "units": "m"}),
            "MEMBER_GLACIERWIDE_SMB": (("member",), np.array([m["smb_glacierwide_igm"] for m in members], np.float32), {"long_name": "glacier-wide mean SMB on the model grid", "units": mi}),
        },
        coords={"x": src.x, "y": src.y, "z": ("z", zp.astype(np.float32), {"units": "m a.s.l.", "long_name": "profile node elevation"}),
                "member": ("member", np.arange(len(members)))},
    )
    if "spatial_ref" in src:
        ds["spatial_ref"] = src["spatial_ref"]
    adm_ela = [m["ela"] for m in adm if np.isfinite(m["ela"])]
    ds.attrs = {
        "title": f"ContinuIX {meta['exp']} {meta['gid']} — GROUP_UNIL method{a.method}",
        "method": "Transient differentiable SMB(z) inversion through IGM (Lleshi et al.)",
        "window": f"{meta['t1']:.2f}-{meta['t2']:.2f}",
        "dem_epoch_assumed": f"{meta['t_dem']:.2f}", "da_epoch": f"{meta['t_da']:.2f}",
        "model_resolution_m": meta["dx_igm"],
        "n_members": len(members), "n_admissible": nadm,
        "ELA_admissible_m": json.dumps([round(min(adm_ela), 1), round(max(adm_ela), 1)]) if adm_ela else "[]",
        "sigma_DHDT_source": how_dhdt, "sigma_THK_source": how_thk,
        "representation_band_m": REPR_BAND_M,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{meta['exp']}_{meta['gid']}_method{a.method}.nc")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars if v != "spatial_ref" and ds[v].ndim >= 2}
    ds.to_netcdf(out, encoding=enc)

    plot_dir = a.plot_dir or a.run_root
    os.makedirs(plot_dir, exist_ok=True)
    zi = np.concatenate([m["z_cells"] for m in members])
    hist_b = np.arange(np.floor(zi.min() / 50) * 50, zi.max() + 50, 50)
    cnt = np.histogram(zc, bins=hist_b)[0] / nadm
    fig7_path = os.path.join(plot_dir, f"fig7_{meta['exp']}_{meta['gid']}_method{a.method}.png")
    fig7(fig7_path, f"{meta['exp']} {meta['gid']} method{a.method}: {nadm}/{len(members)} admissible",
         meta, members, adm_flags, zp, prof, (float(zi.min()), float(zi.max())), (hist_b[:-1] + 25, cnt))
    gw = np.nanmean(np.where(ice_n, smb, np.nan))
    summary = {
        "n_members": len(members), "n_admissible": nadm, "glacierwide_smb_mie": float(gw),
        "sigma_DHDT_source": how_dhdt, "sigma_THK_source": how_thk,
        "mean_unct_smb": float(np.nanmean(nan(u_smb))),
        "mean_components": {"family": float(np.nanmean(nan(sp_smb))), "representation": float(np.nanmean(nan(rep))),
                            "sigma_dhdt": float(np.nanmean(nan(sig_dhdt)))},
        "members": [{"tau_ref": m["slidingco"], "A": m["arrhenius"], "velsurf": m["velsurf"], "ela": m["ela"],
                     "admissible": f, "rms_geom_misfit_m": m["rms_geom_misfit"],
                     "glacierwide_smb_mie": m["smb_glacierwide_igm"], "dir": m["dir"]}
                    for m, f in zip(members, adm_flags)],
    }
    with open(os.path.join(plot_dir, f"family_summary_method{a.method}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[export] wrote {out}\n  fig7 {fig7_path}\n"
          f"  glacier-wide SMB {gw:+.3f} m i.e./yr ({gw * RHO_ICE / 1000:+.3f} m w.e./yr); mean UNCT_SMB {summary['mean_unct_smb']:.3f} "
          f"= family {summary['mean_components']['family']:.3f} + repr {summary['mean_components']['representation']:.3f} "
          f"+ sigma_DHDT {summary['mean_components']['sigma_dhdt']:.3f} (quadrature; sigma_DHDT: {how_dhdt})\n  members: "
          + ", ".join(f"tau={m['slidingco']:.3f} vs={m['velsurf']:.2f} ELA={m['ela']:.0f}{'' if f else ' (rej)'}"
                      for m, f in zip(members, adm_flags)))


if __name__ == "__main__":
    main()
