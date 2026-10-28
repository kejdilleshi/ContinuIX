#!/usr/bin/env python3
"""
Diagnostics for one IGM DA + SMB-inference run directory (IGM grid).

    python tools/diag_run.py <run_dir> --meta data/EXP01_G03.json [--stakes stakes.csv]

Reports: final geometry misfit, mean/RMS end-of-window thickness residual on
the fitted cells, cells with negative thickness, modelled vs observed mean
thinning rate on the ice mask, mass that left the mask, glacier-wide SMB and
the mean transient flux divergence (should be ~0 when mass stays on the mask).
With --stakes (CSV with X, Y and an SMB column in m w.e./yr, same CRS as the
ContinuIX grid) the SMB field (profile at the mid-window surface) is sampled
at the stake XY on the IGM grid — consistent datum, unlike the profile at the
stake altitude.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hat(z, v, z0, dz):
    t = (z - z0) / dz
    w = np.clip(1.0 - np.abs(t[..., None] - np.arange(v.size)), 0, None)
    return (w * v).sum(-1)


def final_costs(run_dir):
    """Last row of IGM's costs.dat (it appends: take the block after the last header)."""
    p = os.path.join(run_dir, "costs.dat")
    if not os.path.exists(p):
        return {}
    lines = [ln.split() for ln in open(p) if ln.strip()]
    hdr = lines[0]
    starts = [i for i, ln in enumerate(lines) if ln[0] == hdr[0]]
    body = lines[starts[-1] + 1:]
    return {f"DA_{k}": float(v) for k, v in zip(hdr, body[-1])} if body else {}


def diag(run_dir, meta, stakes=None):
    ig = xr.open_dataset(os.path.join(CX_DIR, "data", meta["igm_input"]))
    m = ig.icemask.values > 0.5
    geo = xr.open_dataset(os.path.join(run_dir, "geology-optimized.nc"))
    topg = geo.usurf.values - geo.thk.values
    sd = os.path.join(run_dir, "smb_inference")
    H1 = np.clip(ig.usurfstart.values - topg, 0, None) * m
    H2 = np.load(os.path.join(sd, "thk_optimized.npy"))
    Ho = np.load(os.path.join(sd, "thk_observed.npy"))
    v = np.load(os.path.join(sd, "smb_vec.npy"))
    z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
    data = np.load(os.path.join(sd, "data_history.npy"))
    dt = meta["dt"]
    fit = m & (Ho > 1)
    r = (H2 - Ho)[fit]
    smb = hat(topg + 0.5 * (H1 + H2), v, z0, dz)
    dmod = (H2 - H1) / dt
    out = {
        "misfit_final": float(data[-1]), "n_evals": int(len(data)),
        "fit_mean_resid_m": float(r.mean()), "fit_rms_resid_m": float(np.sqrt((r ** 2).mean())),
        "H<0_cells": int((H2 < 0).sum()), "H2_min": float(H2.min()),
        "dHdt_mod_on_mask": float(dmod[m].mean()), "dHdt_obs_on_mask": float(ig.dhdt.values[m].mean()),
        "mass_off_mask_m_per_yr": float(((H2 - H1) / dt)[~m].sum() / m.sum()),
        "smb_glacierwide_mie": float(smb[m].mean()),
        "fdiv_mean_on_mask": float((smb - dmod)[m].mean()),
        "glacierwide_smb_minus_dhdt_obs": float(smb[m].mean() - ig.dhdt.values[m].mean()),
    }
    out.update({k: v for k, v in final_costs(run_dir).items() if k in ("DA_velsurf", "DA_thk", "DA_divflux", "DA_thk_regu")})
    # DA thickness vs the provided (GPR-based) thickness at the DA epoch, where it exists (not stakes)
    tko = ig.thkobs.values
    okt = m & np.isfinite(tko)
    thk_da = geo.thk.values
    out["thk_da_mean_m"] = float(thk_da[m].mean())
    out["thk_da_minus_provided_rms_m"] = float(np.sqrt(np.mean((thk_da[okt] - tko[okt]) ** 2))) if okt.any() else float("nan")
    out["thk_da_minus_provided_mean_m"] = float(np.mean(thk_da[okt] - tko[okt])) if okt.any() else float("nan")
    if stakes:
        st = pd.read_csv(stakes)
        col = [c for c in st.columns if "smb" in c.lower()][0]
        f = xr.DataArray(np.where(m, smb, np.nan), coords={"y": ig.y, "x": ig.x}, dims=("y", "x"))
        s = f.sel(x=xr.DataArray(st.X.values), y=xr.DataArray(st.Y.values), method="nearest").values * 0.917
        ok = np.isfinite(s)
        out["stake_rmse_xy_mwe"] = float(np.sqrt(np.mean((s[ok] - st[col].values[ok]) ** 2)))
        out["stake_bias_xy_mwe"] = float(np.mean(s[ok] - st[col].values[ok]))
        out["stake_n"] = int(ok.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stakes", default=None)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    rows = {"/".join(os.path.normpath(d).split(os.sep)[-2:]): diag(d, meta, a.stakes) for d in a.run_dirs}
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df)


if __name__ == "__main__":
    main()
