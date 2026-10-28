#!/usr/bin/env python3
"""
Score DA thickness against the held-out GPR of the with/without-GPR comparison (tools/gpr_pair_input.py).

  tune    <hydra multirun dir>  gpr variant: one DA per (reg.thk, thkobs_std); writes gpr_tune_result.json with
          the choice. Rule (anti-overfitting): among runs whose velocity RMSE is <= 1.5 x the best, take those
          whose VALIDATION RMSE is within --tol of the best, then the largest thkobs_std (least weight on the
          GPR), then the largest reg.thk (smoothest). The test profiles are not looked at.
  family  <family dir>          any variant: train / val / test RMSE and bias of each member (and of the
          member mean); writes gpr_scores.json next to it.

    python tools/gpr_eval.py tune results/EXP01_G05_gpr/gpr_tune --input EXP01_G05_dagpr.nc
    python tools/gpr_eval.py family results/EXP01_G05_nogpr/family --input EXP01_G05_dagpr.nc
"""
import argparse
import glob
import json
import os

import numpy as np
import xarray as xr

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = (("train", "thkobs"), ("val", "thkobs_val"), ("test", "thkobs_test"))


def overrides(run_dir):
    p = os.path.join(run_dir, ".hydra", "overrides.yaml")
    out = {}
    if os.path.exists(p):
        for ln in open(p):
            ln = ln.strip().lstrip("- ")
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k.split(".")[-1]] = v
    return out


def score(nc, inp):
    with xr.open_dataset(nc) as ds:
        thk = ds.thk.values.astype(float)
        m = ds.icemask.values > 0.5
        vel = float(np.sqrt(np.nanmean((ds.velsurf_mag.values[m] - ds.velsurfobs_mag.values[m]) ** 2)))
        ref = inp.sel(x=ds.x, y=ds.y, method="nearest")
        dx = float(ds.x[1] - ds.x[0])
    r = {"vel_rmse": vel, "volume_km3": float(np.sum(thk[m]) * dx * dx / 1e9)}
    for name, var in SETS:
        o = ref[var].values
        k = np.isfinite(o)
        d = thk[k] - o[k]
        r[f"{name}_n"] = int(k.sum())          # a set can be empty (few profiles -> no test set)
        r[f"{name}_rmse"], r[f"{name}_bias"] = ((float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))) if k.any()
                                                else (float("nan"), float("nan")))
    return r, thk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "family"])
    ap.add_argument("dir")
    ap.add_argument("--input", required=True, help="IGM input name in data/ (with thkobs, thkobs_val, thkobs_test)")
    ap.add_argument("--tol", type=float, default=1.05)
    ap.add_argument("--vel-cutoff", type=float, default=1.5)
    a = ap.parse_args()
    inp = xr.open_dataset(os.path.join(CX_DIR, "data", a.input)).load()
    rows = []
    for nc in sorted(glob.glob(os.path.join(a.dir, "*", "geology-optimized.nc"))):
        r, thk = score(nc, inp)
        ov = overrides(os.path.dirname(nc))
        r.update(run=os.path.basename(os.path.dirname(nc)),
                 reg_thk=float(ov.get("thk", "nan")), thkobs_std=float(ov.get("thkobs_std", "nan")),
                 slidingco=float(ov.get("init_slidingco", ov.get("slidingco", "nan"))))   # paper / dev override name
        r["_thk"] = thk
        rows.append(r)
    if not rows:
        raise SystemExit(f"no geology-optimized.nc under {a.dir}")
    for r in rows:
        print(f"  {r['run']:>14s} reg={r['reg_thk']:<6g} std={r['thkobs_std']:<4g} vel {r['vel_rmse']:6.2f}  "
              + "  ".join(f"{n} {r[n + '_rmse']:6.1f} ({r[n + '_bias']:+6.1f})" for n, _ in SETS)
              + f"  V {r['volume_km3']:.2f} km3")

    if a.mode == "tune":
        vbest = min(r["vel_rmse"] for r in rows)
        ok = [r for r in rows if r["vel_rmse"] <= a.vel_cutoff * vbest]
        best_val = min(r["val_rmse"] for r in ok)
        near = [r for r in ok if r["val_rmse"] <= a.tol * best_val]
        pick = sorted(near, key=lambda r: (r["thkobs_std"], r["reg_thk"]))[-1]
        res = {"rule": f"vel <= {a.vel_cutoff} x best; val RMSE <= {a.tol} x best; then max thkobs_std, max reg.thk",
               "best": {k: v for k, v in pick.items() if k != "_thk"},
               "runs": [{k: v for k, v in r.items() if k != "_thk"} for r in rows]}
        out = os.path.join(a.dir, "gpr_tune_result.json")
        print(f"pick: reg.thk={pick['reg_thk']:g} thkobs_std={pick['thkobs_std']:g} val {pick['val_rmse']:.1f} m "
              f"(best val {best_val:.1f} m)")
    else:
        mean_thk = np.mean([r["_thk"] for r in rows], axis=0)
        ref = inp.sel(x=xr.open_dataset(glob.glob(os.path.join(a.dir, "*", "geology-optimized.nc"))[0]).x,
                      y=xr.open_dataset(glob.glob(os.path.join(a.dir, "*", "geology-optimized.nc"))[0]).y,
                      method="nearest")
        mean = {}
        for name, var in SETS:
            o = ref[var].values; k = np.isfinite(o); d = mean_thk[k] - o[k]
            mean[f"{name}_rmse"], mean[f"{name}_bias"] = ((float(np.sqrt(np.mean(d ** 2))), float(np.mean(d))) if k.any()
                                                          else (float("nan"), float("nan")))
        print("  member mean: " + "  ".join(f"{n} {mean[n + '_rmse']:6.1f} ({mean[n + '_bias']:+6.1f})" for n, _ in SETS))
        res = {"members": [{k: v for k, v in r.items() if k != "_thk"} for r in rows], "member_mean": mean}
        out = os.path.join(os.path.dirname(os.path.abspath(a.dir)), "gpr_scores.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
