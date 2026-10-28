#!/usr/bin/env python3
"""
Assemble the ContinuIX "globally available inputs" experiments (EXP16-20) into
one EXP01-style NetCDF per glacier, so continuix_to_igm.py can read them.

Each EXP16-20 glacier comes as four files on (possibly) different grids:
    EXP<n>_<G>_base_vars.nc   BED, DEM, ELEVBINS, ICEMASK (+BASINMASK)  [reference grid]
    EXP<n>_<G>_glob_THK.nc    THK            (site, Farinotti19, Millan22 or Ma25)
    EXP<n>_<G>_glob_VX-VY.nc  VX, VY, UNCT_* (site or Millan22)
    EXP<n>_<G>_glob_dhdt.nc   DHDT, UNCT_DHDT (Hugonnet21, 2000-2020 mean rate, 100 m)
The replacement fields are interpolated onto the base_vars grid (bilinear;
missing values -> 0 as in the ContinuIX packages). The inversion window is
kept at the glacier's EXP01 window (DHDT timestamp of EXP01_<G>.nc): the
Hugonnet rate is applied over that window, as in the Hugonnet test of the
paper (Appendix A1); its native period is kept in the `source_period` attribute.

Writes input_data/EXP<n>/EXP<n>_<G>.nc.
    python tools/assemble_glob.py EXP16 EXP17 EXP18 EXP19 EXP20
"""
import glob
import os
import sys

import numpy as np
import xarray as xr

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(CX_DIR, "input_data")


def onto(da, ref):
    """Interpolate da onto ref's x/y (exact copy when the grids coincide)."""
    if da.sizes == ref.sizes and np.allclose(da.x, ref.x) and np.allclose(da.y, ref.y):
        out = da.copy()
    else:
        out = da.interp(x=ref.x, y=ref.y, method="linear")
    out = out.assign_coords(x=ref.x, y=ref.y)
    out.attrs = dict(da.attrs)
    return out.fillna(0.0)


def assemble(exp, gid):
    d = os.path.join(IN, exp)
    base = xr.open_dataset(os.path.join(d, f"{exp}_{gid}_base_vars.nc"))
    ref01 = xr.open_dataset(os.path.join(IN, "EXP01", f"EXP01_{gid}.nc"))
    out = base.copy()
    for part in ("glob_THK", "glob_VX-VY", "glob_dhdt"):
        src = xr.open_dataset(os.path.join(d, f"{exp}_{gid}_{part}.nc"))
        for v in src.data_vars:
            if v == "spatial_ref":
                continue
            name = "DHDT" if v.upper() == "DHDT" else ("UNCT_DHDT" if v.upper() == "UNCT_DHDT" else v)
            out[name] = onto(src[v], base)
            out[name].attrs["source_file"] = os.path.basename(src.encoding.get("source", part))
    # keep the reference window; record the product's own period
    out["DHDT"].attrs["source_period"] = out["DHDT"].attrs.get("timestamp", "")
    out["DHDT"].attrs["timestamp"] = ref01["DHDT"].attrs.get("timestamp", "")
    out.attrs["assembled_by"] = "tools/assemble_glob.py"
    path = os.path.join(d, f"{exp}_{gid}.nc")
    out.to_netcdf(path)
    return path


def main():
    for exp in sys.argv[1:]:
        gids = sorted({os.path.basename(p).split("_")[1]
                       for p in glob.glob(os.path.join(IN, exp, f"{exp}_*_base_vars.nc"))})
        for g in gids:
            print("wrote", assemble(exp, g))


if __name__ == "__main__":
    main()
