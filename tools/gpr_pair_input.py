#!/usr/bin/env python3
"""
IGM input for the with/without-GPR comparison (variants nogpr / gpr of run_continuix.py).

Neither run sees a provided thickness or bed raster (THK, BED, Farinotti, ...): the thickness and bed come
from the DA only. Starting from the regular conversion data/<EXP>_<GID>.nc (surfaces, velocities, mask; these
do not depend on THK), this writes data/<EXP>_<GID>_dagpr.nc with

  thk, thkinit  surface-slope (perfect-plasticity) first guess from the DA surface only
  thkobs        GPR thickness on the TRAIN profiles (used by the gpr variant's DA cost; ignored by nogpr)
  thkobs_val    GPR thickness on the VALIDATION profiles (DA tuning of the gpr variant: thkobs_std, reg.thk)
  thkobs_test   GPR thickness on the TEST profiles (final comparison only; never used for any choice)

GPR: the raw point shapefile (EXP02). The split is by profile (prf_name), so a held-out cell is not just a
neighbour of a training point on the same line; a cell crossed by several sets is given to the set used most
for fitting (train > val > test), so val/test cells contain no train point. The points carry z_bed, which does
not change with time, so the observed thickness at the DA epoch is usurf(t_DA) - mean z_bed of the cell
(the surveys are 2009-2011; their z_surf is ~17 m above the 2017 DEM on Aletsch).

    python tools/gpr_pair_input.py EXP01 G05
"""
import argparse
import json
import os
import sys

import geopandas as gpd
import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from continuix_to_igm import plastic_thickness   # noqa: E402

GPR = {"G05": ("EXP02/G05/EXP02_G05_THK.shp", "EPSG:2056")}
SPLIT = (0.6, 0.2, 0.2)
SEED = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    ap.add_argument("gid")
    a = ap.parse_args()
    base = f"{a.exp}_{a.gid}"
    src_nc, src_js = (os.path.join(CX_DIR, "data", base + e) for e in (".nc", ".json"))
    if not os.path.exists(src_nc):
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "continuix_to_igm.py"),
                        os.path.join(CX_DIR, "input_data", a.exp, base + ".nc")], check=True)
    ds = xr.open_dataset(src_nc).load()
    meta = json.load(open(src_js))
    x, y = ds.x.values.astype(float), ds.y.values.astype(float)
    dx = float(x[1] - x[0])
    ice = ds.icemask.values > 0.5
    usurf = ds.usurf.values.astype(float)

    # first guess: surface slope only
    h0 = np.where(ice, plastic_thickness(usurf, ice, dx), 0.0).astype(np.float32)

    shp, crs = GPR[a.gid]
    g = gpd.read_file(os.path.join(CX_DIR, "input_data", shp)).to_crs(crs)
    prof = np.array(sorted(g.prf_name.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(prof))
    n_tr, n_va = int(round(SPLIT[0] * len(prof))), int(round(SPLIT[1] * len(prof)))
    which = {p: 0 for p in prof[perm[:n_tr]]}
    which.update({p: 1 for p in prof[perm[n_tr:n_tr + n_va]]})
    which.update({p: 2 for p in prof[perm[n_tr + n_va:]]})
    s = g.prf_name.map(which).values
    ix = np.round((g.geometry.x.values - x[0]) / dx).astype(int)
    iy = np.round((g.geometry.y.values - y[0]) / (y[1] - y[0])).astype(int)
    ok = (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
    ix, iy, s, zb = ix[ok], iy[ok], s[ok], g.z_bed.values[ok]
    ok = ice[iy, ix]
    ix, iy, s, zb = ix[ok], iy[ok], s[ok], zb[ok]

    cell_set = np.full(ice.shape, 9)
    np.minimum.at(cell_set, (iy, ix), s)           # train(0) > val(1) > test(2)
    out = {}
    for k, name in enumerate(("thkobs", "thkobs_val", "thkobs_test")):
        m = s == k
        sb = np.zeros(ice.shape); n = np.zeros(ice.shape)
        np.add.at(sb, (iy[m], ix[m]), zb[m]); np.add.at(n, (iy[m], ix[m]), 1)
        sel = (cell_set == k) & (n > 0)
        h = np.full(ice.shape, np.nan)
        h[sel] = np.clip(usurf[sel] - sb[sel] / n[sel], 0, None)
        out[name] = h.astype(np.float32)

    ds["thk"] = (("y", "x"), h0, {"long_name": "DA first guess: perfect-plasticity thickness from the surface slope", "units": "m"})
    ds["thkinit"] = (("y", "x"), h0, ds["thk"].attrs)
    for name, lab in (("thkobs", "train"), ("thkobs_val", "validation"), ("thkobs_test", "test")):
        ds[name] = (("y", "x"), out[name], {"long_name": f"GPR thickness at DA epoch ({lab} profiles; usurf - z_bed)", "units": "m"})
    ds.attrs["history"] = ds.attrs.get("history", "") + "; tools/gpr_pair_input.py (no provided THK/BED)"

    dst = base + "_dagpr"
    ds.to_netcdf(os.path.join(CX_DIR, "data", dst + ".nc"))
    cnt = {k: int(np.isfinite(v).sum()) for k, v in out.items()}
    meta.update({"igm_input": dst + ".nc", "first_guess": "perfect plasticity (tau=100 kPa, f=0.8, 500 m slope smoothing)",
                 "gpr_split": {"seed": SEED, "fractions": SPLIT, "n_profiles": len(prof),
                               "n_profiles_train_val_test": [n_tr, n_va, len(prof) - n_tr - n_va], "cells": cnt},
                 "n_thkobs_igm": cnt["thkobs"]})
    json.dump(meta, open(os.path.join(CX_DIR, "data", dst + ".json"), "w"), indent=2)
    hv = {k: float(np.nanmean(v)) for k, v in out.items()}
    print(f"{dst}: first guess mean {h0[ice].mean():.0f} m (max {h0[ice].max():.0f});  GPR cells {cnt}  mean H {hv}")


if __name__ == "__main__":
    main()
