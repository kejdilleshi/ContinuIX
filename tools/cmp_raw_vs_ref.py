#!/usr/bin/env python3
"""Sanity check of an assembled EXP02 file against EXP01 (on the ice mask): coverage and agreement where both exist."""
import sys
import numpy as np, xarray as xr
for g in sys.argv[1:]:
    a = xr.open_dataset(f"input_data/EXP02/EXP02_{g}.nc"); b = xr.open_dataset(f"input_data/EXP01/EXP01_{g}.nc")
    ice = np.nan_to_num(b.ICEMASK.values) > 0
    print(f"== {g}: ice cells {ice.sum()}")
    for v in ["DEM", "DHDT", "VX", "VY", "THK", "BED"]:
        x, y = a[v].values[ice], b[v].values[ice]
        okb = np.isfinite(y) & (y != 0) if v != "BED" else np.isfinite(y)
        both = np.isfinite(x) & okb
        d = x[both] - y[both]
        print(f"   {v:5s} finite on ice {100*np.isfinite(x).mean():5.1f}% (EXP01 non-zero {100*okb.mean():5.1f}%)  "
              f"both {both.sum():6d}  diff mean {d.mean() if d.size else np.nan:+8.3f}  rms {np.sqrt((d**2).mean()) if d.size else np.nan:8.3f}  (EXP01 rms {np.sqrt(np.mean(y[both]**2)) if d.size else np.nan:8.2f})")
