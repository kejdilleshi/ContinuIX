#!/usr/bin/env python3
"""
Compare the SMB-phase reruns (results/<EXP>_<GID>_cnn/smb_<tag>/export/) with the current submission
(GROUP_UNIL/<EXP>/<EXP>_<GID>_method01.nc): glacier-wide SMB, admissible ELA range, mean end-of-window
geometry misfit of the members, mean UNCT_SMB, and the RMS difference of the two SMB(z) profiles.

    python tools/smb_compare.py v2 [--csv out.csv]
"""
import argparse
import glob
import json
import os

import numpy as np
import xarray as xr

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summary(path):
    d = xr.open_dataset(path)
    ice = np.isfinite(d.SMB.values)
    ela = json.loads(d.attrs.get("ELA_admissible_m", "[]") or "[]")
    return {"smb": float(np.nanmean(d.SMB.values)), "ela": float(np.mean(ela)) if ela else np.nan,
            "geom": float(np.nanmean(d.MEMBER_RMS_GEOMETRY_MISFIT.values)), "unct": float(np.nanmean(d.UNCT_SMB.values)),
            "z": d.z.values.astype(float), "prof": d.SMB_PROFILE.values.astype(float), "n": int(ice.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for new in sorted(glob.glob(os.path.join(CX_DIR, "results", "*_cnn", f"smb_{a.tag}", "export", "*_method01.nc"))):
        name = os.path.basename(new).replace("_method01.nc", "")
        old = os.path.join(CX_DIR, "GROUP_UNIL", name[:5], os.path.basename(new))
        if not os.path.exists(old):
            continue
        o, n = summary(old), summary(new)
        zz = np.intersect1d(np.round(o["z"], 1), np.round(n["z"], 1))
        po = np.interp(zz, o["z"], o["prof"]); pn = np.interp(zz, n["z"], n["prof"])
        ok = np.isfinite(po) & np.isfinite(pn)
        dprof = float(np.sqrt(np.mean((pn[ok] - po[ok]) ** 2))) if ok.any() else np.nan
        rows.append((name, o["smb"], n["smb"], o["ela"], n["ela"], o["geom"], n["geom"], o["unct"], n["unct"], dprof))
    hdr = ("experiment", "SMB_old", "SMB_new", "ELA_old", "ELA_new", "geom_old", "geom_new", "UNCT_old", "UNCT_new", "dprof_rms")
    print(" ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        print(f"{r[0]:>10s} " + " ".join(f"{v:10.3f}" if abs(v) < 100 else f"{v:10.0f}" for v in r[1:]))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write(",".join(hdr) + "\n")
            for r in rows:
                f.write(",".join([r[0]] + [f"{v:.4f}" for v in r[1:]]) + "\n")


if __name__ == "__main__":
    main()
