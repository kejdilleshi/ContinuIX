#!/usr/bin/env python3
"""
Compare SMB reruns of one member: 2-D field inversions (inversion.method=field) against a profile run.

For each smb_<tag>/<member>/smb_inference: per-cell thickness RMSE at the window end, the RMS of the
50 m elevation-band mean residuals (the G data term), mean |residual|, glacier-wide SMB, final
roughness of the node field. One figure: SMB maps (row 1) and end-of-window residual maps (row 2).

  python tools/smb_field_compare.py --exp EXP01 --gid G01 --member slid_0.0568 \
      --tags G_bands_geod F_reg0.1 F_reg1 F_reg10 --out results/EXP01_G01_cnn/smb_field_scan.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def band_rms(r, z, on, band=50.0):
    k = np.floor((z[on] - z[on].min()) / band).astype(int)
    n = np.bincount(k)
    m = np.bincount(k, r[on]) / np.where(n > 0, n, 1)
    return float(np.sqrt((n * m ** 2).sum() / n.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gid", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = os.path.join(CX_DIR, "results", f"{a.exp}_{a.gid}_cnn")
    rows = []
    for tag in a.tags:
        sd = os.path.join(root, f"smb_{tag}", a.member, "smb_inference")
        H = np.load(os.path.join(sd, "thk_optimized.npy"))
        obs = np.load(os.path.join(sd, "thk_observed.npy"))
        smb = np.load(os.path.join(sd, "smb_field.npy"))
        z = np.load(os.path.join(sd, "usurf_modeled.npy"))
        on = smb != 0
        r = np.where(on, H - obs, np.nan)
        rough = np.nan
        log = os.path.join(root, f"smb_{tag}", a.member + ".log")
        for ln in open(log):
            if "final loss=" in ln and "roughness=" in ln:
                rough = float(ln.split("roughness=")[1])
        rows.append(dict(tag=tag, smb=np.where(on, smb, np.nan), r=r,
                         rmse=float(np.sqrt(np.nanmean(r ** 2))), band=band_rms(np.nan_to_num(r), z, on),
                         mae=float(np.nanmean(np.abs(r))), gw=float(smb[on].mean()), rough=rough))

    print(f"{'tag':<16}{'cell RMSE m':>12}{'band RMS m':>12}{'MAE m':>8}{'SMB m/yr':>10}{'rough':>8}")
    for d in rows:
        print(f"{d['tag']:<16}{d['rmse']:>12.2f}{d['band']:>12.2f}{d['mae']:>8.2f}{d['gw']:>+10.3f}{d['rough']:>8.3f}")

    n = len(rows)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6.4), squeeze=False)
    vs = np.nanpercentile(np.abs(np.concatenate([d["smb"].ravel() for d in rows])), 99)
    vr = 40.0
    for j, d in enumerate(rows):
        im0 = ax[0, j].imshow(d["smb"], origin="lower", cmap="RdBu", vmin=-vs, vmax=vs)
        ax[0, j].set_title(f"{d['tag']}\nSMB, mean {d['gw']:+.2f} m/yr", fontsize=8)
        im1 = ax[1, j].imshow(d["r"], origin="lower", cmap="PuOr_r", vmin=-vr, vmax=vr)
        ax[1, j].set_title(f"H_sim - H_obs: RMSE {d['rmse']:.1f} m", fontsize=8)
        for k in (0, 1):
            ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
    fig.colorbar(im0, ax=ax[0, :], shrink=0.8, label="SMB (m i.e./yr)")
    fig.colorbar(im1, ax=ax[1, :], shrink=0.8, label="end-of-window thickness residual (m)")
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"[smb_field_compare] {a.out}")


if __name__ == "__main__":
    main()
