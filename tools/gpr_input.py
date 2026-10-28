#!/usr/bin/env python3
"""
DA input of the current procedure (variant `cnn` of run_continuix.py) for any experiment x glacier.

No provided thickness or bed enters the DA as a field (user decision 2026-10-13): starting from the regular
conversion data/<EXP>_<GID>.nc (surfaces, velocities, mask), this writes data/<EXP>_<GID>_da.nc with

  thk, thkinit  first guess: perfect-plasticity thickness from the DA surface slope only
  thkobs        GPR thickness on the TRAIN profiles      (in the DA cost)
  thkobs_val    GPR thickness on the VALIDATION profiles (tuning of thkobs_std / reg.thk only)
  thkobs_test   GPR thickness on the TEST profiles       (reported, never used for a choice)

GPR profiles = the raw thickness of the EXP02 package where it is point/profile data, i.e. covers <= 25 % of the
glacier (G02, G03, G05, S02). G01, G04, G06 ship gridded (97-99 % coverage) thickness products and S01 none: no
GPR, velocity-only DA. EXP17-20 replace THK by model products (Farinotti, Millan, ICEBOOST): no GPR there either.

Values: the experiment's OWN THK sampled at the GPR pixels and moved to the DA epoch with DHDT
(H = THK + DHDT (t_da - t_thk)), so the thickness perturbations of EXP03-07 reach the observations; in EXP02 this
is the raw GPR itself. Native pixels are averaged per model cell.

Split (anti-overfitting): by profile segment = 8-connected component of GPR pixels on the EXP02 grid, so a held-out
cell is never the neighbour of a training cell on the same line. Components are shuffled (seed 0) and assigned
greedily to reach the target cell fractions: 60/20/20 with >= 10 components, else 70/30/0 (no test set; at least
one validation component). A model cell crossed by several sets goes to the set used most for fitting
(train > val > test), so val/test cells never contain a training pixel.

    python tools/gpr_input.py EXP01 S02
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import xarray as xr
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from continuix_to_igm import plastic_thickness   # noqa: E402

MAX_COVER = 0.25                       # raw thickness on more of the glacier than this = gridded product, not GPR
NO_GPR_EXPS = {"EXP17", "EXP18", "EXP19", "EXP20"}   # THK replaced by model products
SEED = 0
SPLITS = {True: (0.6, 0.2, 0.2), False: (0.7, 0.3, 0.0)}   # key: >= MIN_COMP_TEST components
MIN_COMP_TEST = 10


def gpr_pixels(gid):
    """Native GPR pixel centres and their profile-segment label from the EXP02 package (None if no GPR)."""
    p = os.path.join(CX_DIR, "input_data", "EXP02", f"EXP02_{gid}.nc")
    if not os.path.exists(p):
        return None
    with xr.open_dataset(p) as r:
        thk = r["THK"].values
        ice = np.isfinite(r["ICEMASK"].values) if "ICEMASK" in r else np.ones(thk.shape, bool)
        k = np.isfinite(thk) & (thk > 0) & ice
        if k.sum() == 0 or k.sum() > MAX_COVER * ice.sum():
            return None
        lab, n = ndimage.label(k, structure=np.ones((3, 3)))
        iy, ix = np.nonzero(k)
        return {"x": r.x.values[ix].astype(float), "y": r.y.values[iy].astype(float), "comp": lab[iy, ix], "n_comp": n}


def native_input(exp, gid):
    for name in (f"{exp}_{gid}.nc", f"{exp}_{gid}_base_vars.nc"):
        p = os.path.join(CX_DIR, "input_data", exp, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no native input for {exp} {gid}")


def split_components(comp_cells, rng):
    """comp_cells: {component: n model cells}. Returns {component: 0 train / 1 val / 2 test}."""
    comps = np.array(sorted(comp_cells))
    frac = SPLITS[len(comps) >= MIN_COMP_TEST]
    total = sum(comp_cells.values())
    order = comps[rng.permutation(len(comps))]
    which, filled = {}, [0, 0, 0]
    # one component to each needed set first, then fill towards the target fractions
    needed = [s for s in range(3) if frac[s] > 0]
    for s, c in zip(needed, order):
        which[c] = s; filled[s] += comp_cells[c]
    for c in order[len(needed):]:
        s = max(needed, key=lambda s: frac[s] * total - filled[s])
        which[c] = s; filled[s] += comp_cells[c]
    return which, frac


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp")
    ap.add_argument("gid")
    a = ap.parse_args()
    base = f"{a.exp}_{a.gid}"
    src_nc, src_js = (os.path.join(CX_DIR, "data", base + e) for e in (".nc", ".json"))
    if not os.path.exists(src_nc):
        subprocess.run([sys.executable, os.path.join(HERE, "continuix_to_igm.py"),
                        os.path.join(CX_DIR, "input_data", a.exp, base + ".nc")], check=True)
    ds = xr.open_dataset(src_nc).load()
    meta = json.load(open(src_js))
    x, y = ds.x.values.astype(float), ds.y.values.astype(float)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    ice = ds.icemask.values > 0.5
    usurf = ds.usurf.values.astype(float)

    h0 = np.where(ice, plastic_thickness(usurf, ice, abs(dx)), 0.0).astype(np.float32)
    obs = {k: np.full(ice.shape, np.nan, np.float32) for k in ("thkobs", "thkobs_val", "thkobs_test")}
    gpr_meta = {"has_gpr": False}

    g = None if a.exp in NO_GPR_EXPS else gpr_pixels(a.gid)
    if g is not None:
        with xr.open_dataset(native_input(a.exp, a.gid)) as nat:
            pts = dict(x=xr.DataArray(g["x"], dims="p"), y=xr.DataArray(g["y"], dims="p"))
            thk = nat["THK"].sel(**pts, method="nearest").values.astype(float)
            dhdt = nat["DHDT"].sel(**pts, method="nearest").values.astype(float) if "DHDT" in nat else 0 * thk
        h = thk + np.nan_to_num(dhdt) * (meta["t_da"] - meta.get("t_thk", meta["t_da"]))
        ix = np.round((g["x"] - x[0]) / dx).astype(int)
        iy = np.round((g["y"] - y[0]) / dy).astype(int)
        ok = np.isfinite(h) & (thk > 0) & (ix >= 0) & (ix < x.size) & (iy >= 0) & (iy < y.size)
        ix, iy, h, comp = ix[ok], iy[ok], np.clip(h[ok], 0, None), g["comp"][ok]
        on = ice[iy, ix]
        ix, iy, h, comp = ix[on], iy[on], h[on], comp[on]
        cells = {}
        for c in np.unique(comp):
            cells[int(c)] = len(set(zip(iy[comp == c].tolist(), ix[comp == c].tolist())))
        which, frac = split_components(cells, np.random.default_rng(SEED))
        s = np.array([which[int(c)] for c in comp])
        cell_set = np.full(ice.shape, 9)
        np.minimum.at(cell_set, (iy, ix), s)            # train(0) > val(1) > test(2)
        hs, n = np.zeros(ice.shape), np.zeros(ice.shape)
        np.add.at(hs, (iy, ix), h); np.add.at(n, (iy, ix), 1)
        for k, name in enumerate(("thkobs", "thkobs_val", "thkobs_test")):
            sel = (cell_set == k) & (n > 0)
            obs[name][sel] = (hs[sel] / n[sel]).astype(np.float32)
        cnt = {name: int(np.isfinite(v).sum()) for name, v in obs.items()}
        gpr_meta = {"has_gpr": cnt["thkobs"] > 0 and cnt["thkobs_val"] > 0,
                    "gpr_source": f"EXP02_{a.gid} raw thickness pixels; values = {a.exp} THK at t_da",
                    "gpr_split": {"seed": SEED, "fractions": list(frac), "unit": "8-connected profile segment",
                                  "n_segments": int(len(cells)),
                                  "n_segments_train_val_test": [sum(1 for v in which.values() if v == k) for k in range(3)],
                                  "cells": cnt}}

    ds["thk"] = (("y", "x"), h0, {"long_name": "DA first guess: perfect-plasticity thickness from the surface slope", "units": "m"})
    ds["thkinit"] = (("y", "x"), h0, ds["thk"].attrs)
    for name, lab in (("thkobs", "train"), ("thkobs_val", "validation"), ("thkobs_test", "test")):
        ds[name] = (("y", "x"), obs[name], {"long_name": f"GPR thickness at DA epoch ({lab} profiles)", "units": "m"})
    ds.attrs["history"] = ds.attrs.get("history", "") + "; tools/gpr_input.py (no provided THK/BED field)"

    dst = base + "_da"
    ds.to_netcdf(os.path.join(CX_DIR, "data", dst + ".nc"))
    meta.update({"igm_input": dst + ".nc", "first_guess": "perfect plasticity (tau=100 kPa, f=0.8, 500 m slope smoothing)",
                 "n_thkobs_igm": int(np.isfinite(obs["thkobs"]).sum()), **gpr_meta})
    json.dump(meta, open(os.path.join(CX_DIR, "data", dst + ".json"), "w"), indent=2)
    msg = gpr_meta.get("gpr_split", "no GPR -> velocity-only DA")
    print(f"{dst}: first guess mean {h0[ice].mean():.0f} m (max {h0[ice].max():.0f});  {msg}")


if __name__ == "__main__":
    main()
