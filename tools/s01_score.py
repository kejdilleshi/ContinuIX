#!/usr/bin/env python3
"""
Score SMB-phase reruns of S01 (tools/smb_rerun.py) against the synthetic law. Diagnostic only.

For every member in results/EXP01_S01_cnn/smb_<tag>/slid_*: final data misfit, iterations/evaluations,
the profile (model hat basis) against SMB(z) = min(da_ds (z - ela), max_a) on a 10 m grid over the glacier's
elevation range (1-99 % of the native DEM on ice, as tools/check_s01.py), ELA and ablation gradient,
and the end-of-window thickness misfit split into the 5 border columns and the interior.

    python tools/s01_score.py base dz100 ...       # tags
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(CX_DIR, "results", "EXP01_S01_cnn")


def hat(z, v, z0, dz):
    t = (z - z0) / dz
    k = np.arange(v.size)
    return (np.clip(1.0 - np.abs(t[..., None] - k), 0.0, None) * v).sum(-1)


def main():
    src = xr.open_dataset(os.path.join(CX_DIR, "input_data", "EXP01", "EXP01_S01.nc"))
    ela, g, cap = (float(src.attrs[k]) for k in ("ela_m", "da_ds", "max_a_m_yr"))
    ice = (np.nan_to_num(src.ICEMASK.values) > 0.5) & (np.nan_to_num(src.THK.values) > 0)
    zs = src.DEM.values[ice]
    zlo, zhi = np.nanpercentile(zs, 1), np.nanpercentile(zs, 99)
    zf = np.arange(np.ceil(zlo / 10) * 10, zhi, 10.0)
    law = np.minimum(g * (zf - ela), cap)
    # hypsometric weights (native DEM) for an area-weighted profile error
    hw = np.histogram(zs, bins=np.r_[zf - 5, zf[-1] + 5])[0].astype(float)
    hw /= hw.sum()
    print(f"law: ELA {ela:g}, gradient {g:g}, cap {cap:g}; scored over {zlo:.0f}-{zhi:.0f} m")
    print(f"{'tag/member':34s} {'data':>8s} {'it':>3s} {'ev':>3s} {'RMSE':>6s} {'aRMSE':>6s} {'bias':>6s} "
          f"{'ELA':>6s} {'grad':>7s} {'Hrms_in':>7s} {'Hrms_bd':>7s}")
    for tag in sys.argv[1:]:
        rows = []
        for d in sorted(glob.glob(os.path.join(ROOT, f"smb_{tag}", "slid_*"))):
            sd = os.path.join(d, "smb_inference")
            if not os.path.isdir(d) or not os.path.exists(os.path.join(sd, "smb_vec.npy")):
                continue
            v = np.load(os.path.join(sd, "smb_vec.npy")).astype(float)
            z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
            p = hat(zf, v, z0, dz)
            e = p - law
            k = np.where((p[:-1] < 0) & (p[1:] >= 0))[0]
            pela = zf[k[0]] - p[k[0]] * 10 / (p[k[0] + 1] - p[k[0]]) if k.size else np.nan
            ab = (zf > zlo) & (zf < ela - 200)
            slope = np.polyfit(zf[ab], p[ab], 1)[0]
            dh = np.load(os.path.join(sd, "data_history.npy"))
            ev = np.load(os.path.join(sd, "eval_log.npy")) if os.path.exists(os.path.join(sd, "eval_log.npy")) else None
            twin = os.path.join(sd, "thk_twin_obs.npy")      # twin test: the target was a model run
            H = np.load(os.path.join(sd, "thk_optimized.npy"))
            O = np.load(twin if os.path.exists(twin) else os.path.join(sd, "thk_observed.npy"))
            m = O > 1
            bd = np.zeros_like(m); bd[:5, :] = bd[-5:, :] = bd[:, :5] = bd[:, -5:] = True
            r = H - O
            rows.append((f"{tag}/{os.path.basename(d)[5:]}", dh[-1], dh.size, 0 if ev is None else ev.shape[0],
                         np.sqrt(np.mean(e ** 2)), np.sqrt(np.sum(hw * e ** 2)), np.mean(e), pela, slope,
                         np.sqrt(np.mean(r[m & ~bd] ** 2)), np.sqrt(np.mean(r[m & bd] ** 2)) if (m & bd).any() else np.nan, p))
        for rw in rows:
            print(f"{rw[0]:34s} {rw[1]:8.3f} {rw[2]:3d} {rw[3]:3d} {rw[4]:6.3f} {rw[5]:6.3f} {rw[6]:+6.3f} "
                  f"{rw[7]:6.0f} {rw[8]:7.4f} {rw[9]:7.2f} {rw[10]:7.2f}")
        if len(rows) > 1:
            pm = np.mean([rw[-1] for rw in rows], 0)
            e = pm - law
            print(f"{tag + '/MEAN':34s} {'':8s} {'':3s} {'':3s} {np.sqrt(np.mean(e ** 2)):6.3f} "
                  f"{np.sqrt(np.sum(hw * e ** 2)):6.3f} {np.mean(e):+6.3f}")


if __name__ == "__main__":
    main()
