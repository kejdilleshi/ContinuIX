#!/usr/bin/env python3
"""
Convert a ContinuIX experiment NetCDF (EXP01-style homogenised grid) into an
IGM-ready input file for the igm-smb-inference pipeline.

What it does
------------
1. Reads epochs from the per-variable ``timestamp`` attributes:
     t1, t2  <- DHDT timestamp (inversion window)
     t_dem   <- DEM timestamp (a range is read as its midpoint)
     t_vel   <- VX timestamp midpoint
   Synthetic cases carry no timestamps; they come from GLACIER_OVERRIDES below.
2. Block-averages every field onto a coarser grid (100 m, or 50 m for glaciers
   < 5 km2), the resolution range the SIADecompNet emulator was trained on.
   Velocity and dh/dt are averaged over valid on-ice pixels only, so the
   zero-fill ContinuIX uses outside the outline does not dilute the margins.
3. Builds the three surfaces the method needs from the single DEM + DHDT:
     usurfstart = DEM + DHDT (t1    - t_dem)   forward-run start (t1)
     usurfinfer = DEM + DHDT (t2    - t_dem)   SMB-inversion target (t2)
     usurf(obs) = DEM + DHDT (t_da  - t_dem)   DA surface
   with t_da = clip(t_vel, t1, t2): the paper's rule of assimilating at the
   epoch closest to the velocity observations.
4. Writes ``ContinuIX/data/<EXP>_<GID>.nc`` and a JSON sidecar with the
   epochs, grids and coarsening factor, which the exporter
   (igm_to_continuix.py) uses to map the results back onto the native grid.

The provided thickness THK (epoch t_thk from its timestamp) is moved to the DA
epoch with DHDT: H = THK + DHDT (t_da - t_thk), clipped at 0; DEM - BED is the
fallback where THK is missing. It is the DA initial guess (variant vel) or the
thickness observation (variant velthk).

ContinuIX zero-fills gaps ("Missing/NaN values were filled with 0"), so inside
the outline an exact 0 is a gap, not a measurement:
  DHDT == 0 (or NaN) on ice -> median DHDT of the pixel's 50 m elevation band
                               (fill_missing_dhdt; also used by the exporter)
  THK  == 0 on ice          -> missing for the thickness observation thkobs
                               (the initial guess keeps the provided value)

Usage
-----
    python tools/continuix_to_igm.py input_data/EXP01/EXP01_G03.nc
    python tools/continuix_to_igm.py input_data/EXP01/EXP01_G03.nc --dx 100
"""
import argparse
import json
import os
import re

import numpy as np
import xarray as xr
from scipy import ndimage

CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CX_DIR, "data")   # IGM core.folder_data, relative to the run cwd

# Synthetic glaciers have no timestamps. S01 is a steady state (DHDT ~ 0), so
# the window length is arbitrary; S02 is a 10-yr perturbation whose DEM epoch
# is not documented -> assume mid-window (see README, open question).
GLACIER_OVERRIDES = {
    "S01": {"t1": 0.0, "t2": 10.0, "t_dem": 5.0, "t_vel": 5.0},
    "S02": {"t1": 0.0, "t2": 10.0, "t_dem": 5.0, "t_vel": 5.0},
}

SMALL_GLACIER_KM2 = 5.0


# ── epochs ───────────────────────────────────────────────────────────────────

def _decimal_year(tok):
    if len(tok) == 8:  # YYYYMMDD
        y, m, d = int(tok[:4]), int(tok[4:6]), int(tok[6:])
        return y + (m - 1) / 12.0 + (d - 1) / 365.25
    return float(tok)


def parse_epochs(ts):
    """'2012-2021' -> (2012, 2021); '"2018"' -> (2018, 2018); '20170901' -> (2017.67, ...)."""
    toks = re.findall(r"\d{8}|\d{4}", str(ts))
    if not toks:
        return None
    yrs = [_decimal_year(t) for t in toks]
    return min(yrs), max(yrs)


def get_epochs(ds, gid):
    if gid in GLACIER_OVERRIDES:
        return dict(GLACIER_OVERRIDES[gid])
    w = parse_epochs(ds["DHDT"].attrs.get("timestamp", ""))
    d = parse_epochs(ds["DEM"].attrs.get("timestamp", ""))
    v = parse_epochs(ds["VX"].attrs.get("timestamp", ""))
    if w is None or d is None:
        raise SystemExit(f"{gid}: cannot parse DHDT/DEM timestamps; add a GLACIER_OVERRIDES entry")
    t1, t2 = w
    t_dem = 0.5 * (d[0] + d[1])
    t_vel = 0.5 * (v[0] + v[1]) if v else t_dem
    h = parse_epochs(ds["THK"].attrs.get("timestamp", ""))
    t_thk = 0.5 * (h[0] + h[1]) if h else t_dem
    return {"t1": t1, "t2": t2, "t_dem": t_dem, "t_vel": t_vel, "t_thk": t_thk}


# ── grid helpers ─────────────────────────────────────────────────────────────

def ice_mask_native(ds, gid):
    im = ds["ICEMASK"].values.astype(float)
    if gid.startswith("S"):
        # S01's ICEMASK spans the whole half-pipe including ice-free rims at
        # glacier elevations; a profile SMB(z) cannot hold H=0 there and
        # accumulation on the centreline at once, so keep only cells with ice.
        m = np.isfinite(im) & (im > 0.5)
        # EXP02 S02 provides thickness in a few cells only: the outline alone defines the glacier there
        thk = ds["THK"].values
        return m & (np.nan_to_num(thk) > 0) if np.isfinite(thk[m]).mean() > 0.5 else m
    return np.isfinite(im) & (im > 0)


def fill_nearest(a):
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


DHDT_BAND_M = 50.0


def fill_missing_dhdt(dhdt, ice, dem, band=DHDT_BAND_M, min_n=5):
    """On-ice DHDT that is exactly 0 or NaN is a ContinuIX zero-fill: replace it
    with the median of the valid (non-zero) on-ice DHDT in the same `band`-m
    elevation band (nearest band with >= min_n valid pixels if its own has too
    few). Off-ice returns 0. Returns (filled, n_filled)."""
    d = np.where(ice, np.asarray(dhdt, float), 0.0)
    gap = ice & (~np.isfinite(d) | (d == 0))
    out = np.where(ice & ~gap, d, 0.0)
    if not gap.any() or not (ice & ~gap).any():
        return out, 0
    zb = np.floor(np.where(np.isfinite(dem), dem, 0.0) / band).astype(int)
    ok = ice & ~gap
    bands = np.unique(zb[ice])
    cnt = np.array([(ok & (zb == b)).sum() for b in bands])
    med = np.array([np.median(d[ok & (zb == b)]) if c else np.nan for b, c in zip(bands, cnt)])
    usable = cnt >= min_n if (cnt >= min_n).any() else cnt > 0
    src = np.array([np.argmin(np.where(usable, np.abs(bands - b), np.inf)) for b in bands])
    out[gap] = med[src][np.searchsorted(bands, zb[gap])]
    return out, int(gap.sum())


def parse_scalar_sigma(txt, window_yr=None, per_year=True):
    """'+- 0.07 m/year' -> 0.07; '+- 2 m' -> 2 / window_yr when per_year (an
    elevation-change error over the window); 'unknown' -> None;
    'unknown, assumed to be <= 0.5 m' -> 0.5 (the stated bound)."""
    t = str(txt or "")
    m = re.search(r"(\d+\.?\d*)\s*(m\s*/\s*(?:yr|year|a)|m)?", t)
    if not m:
        return None
    val, unit = float(m.group(1)), (m.group(2) or "m").replace(" ", "")
    if per_year and unit == "m":
        return val / window_yr if window_yr else None
    return val


def provided_sigma(ds, var, window_yr=None, per_year=True):
    """Provided 1-sigma of `var` on the native grid: the UNCT_<var> field if it
    exists, else the scalar in the variable's `uncertainty` attribute, else None.
    Returns (array_or_float_or_None, source_string)."""
    u = "UNCT_" + var
    if u in ds:
        return ds[u].values.astype(float), f"field {u}"
    sc = parse_scalar_sigma(ds[var].attrs.get("uncertainty"), window_yr, per_year)
    if sc is not None:
        return float(sc), f"attribute '{ds[var].attrs.get('uncertainty')}'"
    return None, "not provided"


def plastic_thickness(dem, ice, dx, tau=1.0e5, shape=0.8, smooth_m=500.0, hmin=10.0, hmax=600.0):
    """Surface-slope thickness H = tau / (f rho g sin(alpha)) (perfect-plasticity, Haeberli & Hoelzle 1995 form),
    slope from the DEM smoothed over ~smooth_m. Used ONLY as the DA initial guess where neither THK nor BED is
    provided (EXP02 S02); the data assimilation then fits the velocity-observed part."""
    sig = max(smooth_m / dx, 1.0)
    w = ndimage.gaussian_filter(ice.astype(float), sig)
    z = ndimage.gaussian_filter(np.where(ice, dem, 0.0), sig) / np.where(w > 1e-6, w, 1.0)
    z = np.where(w > 1e-6, z, dem)
    gy, gx = np.gradient(z, dx)
    sin_a = np.hypot(gx, gy) / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    h = tau / (shape * 917.0 * 9.81 * np.maximum(sin_a, 1e-3))
    return np.clip(h, hmin, hmax)


def block_mean(a, w, f):
    """Weighted mean over f x f blocks (trimmed); NaN where total weight is 0."""
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    a = np.where(w > 0, a, 0.0)[:ny, :nx].reshape(ny // f, f, nx // f, f)
    w = w[:ny, :nx].reshape(ny // f, f, nx // f, f)
    sw = w.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, (a * w).sum(axis=(1, 3)) / sw, np.nan), sw / (f * f)


def block_coord(c, f):
    n = (c.size // f) * f
    return c[:n].reshape(-1, f).mean(axis=1)


# ── main conversion ──────────────────────────────────────────────────────────

def convert(path, dx_target=None, margin_cells=5, suffix=""):
    base = os.path.splitext(os.path.basename(path))[0]          # EXP01_G03
    exp, gid = base.split("_")[:2]
    out_base = base + suffix                                    # e.g. EXP01_G01_dx200 (dev/gate inputs)
    ds = xr.open_dataset(path)

    # Work with ascending y (IGM convention); remember native orientation.
    y_desc = bool(ds.y.values[1] < ds.y.values[0])
    if y_desc:
        ds = ds.isel(y=slice(None, None, -1))
    x, y = ds.x.values.astype(float), ds.y.values.astype(float)
    dx_nat = abs(float(x[1] - x[0]))

    ice = ice_mask_native(ds, gid)
    area_km2 = ice.sum() * dx_nat ** 2 / 1e6
    if dx_target is None:
        dx_target = 50.0 if area_km2 < SMALL_GLACIER_KM2 else 100.0
    # nearest integer block factor; keep the native grid when it is already
    # coarser than the target (EXP14/15 resolution experiments)
    f = max(1, int(round(dx_target / dx_nat)))
    dx_target = f * dx_nat

    ep = get_epochs(ds, gid)
    t1, t2, t_dem, t_vel = ep["t1"], ep["t2"], ep["t_dem"], ep["t_vel"]
    t_thk = ep.get("t_thk", t_dem)
    t_da = float(np.clip(t_vel, t1, t2))

    dem = ds["DEM"].values.astype(float)
    bed = ds["BED"].values.astype(float)
    thk = ds["THK"].values.astype(float)
    dhdt_raw = ds["DHDT"].values.astype(float)
    vx = ds["VX"].values.astype(float)
    vy = ds["VY"].values.astype(float)

    # Surface everywhere: DEM, else BED (off-ice BED == DEM), else nearest.
    dem = fill_nearest(np.where(np.isfinite(dem), dem, bed))
    # On-ice DHDT == 0 / NaN is a zero-filled gap -> median of its elevation band.
    dhdt, n_dhdt_fill = fill_missing_dhdt(dhdt_raw, ice, dem)
    # Provided thickness, moved from its own epoch to the DEM epoch with DHDT
    # (THK and DEM can differ, e.g. G06: 2006 vs 2017). THK, not DEM-BED, is
    # used because the thickness-perturbation experiments (EXP03-07) perturb
    # THK only. DEM-BED is the fallback where THK is missing.
    h_dem = np.where(np.isfinite(thk), thk + dhdt * (t_dem - t_thk), np.nan)
    h_dem = np.where(np.isfinite(h_dem), h_dem, dem - bed)
    n_plastic = int((ice & ~np.isfinite(h_dem)).sum())
    if n_plastic:           # neither THK nor BED there (EXP02 S02): surface-slope thickness as the DA initial guess
        h_dem = np.where(ice & ~np.isfinite(h_dem), plastic_thickness(dem, ice, dx_nat), h_dem)
    # On-ice THK == 0 is a zero-filled gap: kept in the initial guess, but not a
    # thickness observation.
    thk_ok = ice & np.isfinite(thk) & (thk > 0)
    h_obs_dem = np.where(thk_ok, thk + dhdt * (t_dem - t_thk), 0.0)
    h_dem = np.where(ice, np.clip(np.nan_to_num(h_dem), 0, None), 0.0)
    # ContinuIX zero-fills missing velocities: treat exact (0, 0) on ice as missing.
    vel_ok = ice & np.isfinite(vx) & np.isfinite(vy) & ~((vx == 0) & (vy == 0))

    one = np.ones_like(dem)
    icef = ice.astype(float)
    dem_c, _ = block_mean(dem, one, f)
    dhdt_c, frac = block_mean(dhdt, icef, f)
    h_c, _ = block_mean(h_dem, one, f)            # area mean (0 off-ice)
    hobs_c, tfrac = block_mean(h_obs_dem, thk_ok.astype(float), f)   # valid-THK pixels only
    vx_c, vfrac = block_mean(vx, vel_ok.astype(float), f)
    vy_c, _ = block_mean(vy, vel_ok.astype(float), f)
    xc, yc = block_coord(x, f), block_coord(y, f)

    icemask = (frac >= 0.5).astype(np.float32)
    dhdt_c = np.where(icemask > 0, np.nan_to_num(dhdt_c), 0.0)
    # velocity obs only where >= 25 % of the block had a valid measurement
    vobs = (icemask > 0) & (vfrac >= 0.25)
    vx_c = np.where(vobs, vx_c, np.nan)
    vy_c = np.where(vobs, vy_c, np.nan)

    z_start = dem_c + dhdt_c * (t1 - t_dem)
    z_end = dem_c + dhdt_c * (t2 - t_dem)
    z_da = dem_c + dhdt_c * (t_da - t_dem)
    h_da = np.where(icemask > 0, np.clip(h_c + dhdt_c * (t_da - t_dem), 0, None), 0.0)
    # thickness observation: only where >= 25 % of the block has a provided (non-zero) THK
    h_obs_da = np.clip(hobs_c + dhdt_c * (t_da - t_dem), 0, None)
    thkobs = np.where((icemask > 0) & (tfrac >= 0.25) & np.isfinite(h_obs_da), h_obs_da, np.nan)

    # Crop to the ice bounding box + margin.
    rows = np.where(icemask.any(axis=1))[0]
    cols = np.where(icemask.any(axis=0))[0]
    r0, r1 = max(rows[0] - margin_cells, 0), min(rows[-1] + margin_cells + 1, icemask.shape[0])
    c0, c1 = max(cols[0] - margin_cells, 0), min(cols[-1] + margin_cells + 1, icemask.shape[1])
    sl = (slice(r0, r1), slice(c0, c1))

    def da(a, long_name, units):
        return (("y", "x"), np.asarray(a[sl], dtype=np.float32), {"long_name": long_name, "units": units})

    out = xr.Dataset(
        {
            "usurf": da(z_da, f"Surface at DA epoch {t_da:.2f} (DEM + DHDT shift)", "m"),
            "usurfobs": da(z_da, f"Observed surface at DA epoch {t_da:.2f}", "m"),
            "usurfstart": da(z_start, f"Surface at t1={t1:.2f} (forward-run start)", "m"),
            "usurfinfer": da(z_end, f"Surface at t2={t2:.2f} (SMB-inversion target)", "m"),
            "dhdt": da(dhdt_c, f"Observed dh/dt {t1:.2f}-{t2:.2f}", "m/y"),
            "thkinit": da(h_da, f"Provided thickness at DA epoch (DEM-BED shifted)", "m"),
            "thk": da(h_da, "Ice thickness (DA initial state)", "m"),
            "thkobs": da(thkobs, "Provided thickness at DA epoch, zero-filled gaps excluded (used only by thk-aware DA)", "m"),
            "uvelsurfobs": da(vx_c, "x surface velocity", "m/y"),
            "vvelsurfobs": da(vy_c, "y surface velocity", "m/y"),
            "icemask": da(icemask, "Ice mask (any year in window)", ""),
            "icemaskobs": da(icemask, "Ice mask for DA icemask cost", ""),
        },
        coords={"x": ("x", xc[sl[1]].astype(np.float32)), "y": ("y", yc[sl[0]].astype(np.float32))},
    )
    if "BASINMASK" in ds:
        # basin index of each block's centre pixel (categorical: no averaging)
        bm = np.nan_to_num(ds["BASINMASK"].values.astype(float))
        bm_c = bm[f // 2::f, f // 2::f][: icemask.shape[0], : icemask.shape[1]]
        out["basinmask"] = da(np.where(icemask > 0, bm_c, 0), "Basin index (nearest)", "")
    out.attrs = {
        "title": f"IGM input from ContinuIX {exp} {gid}",
        "source_file": os.path.abspath(path),
        "history": "tools/continuix_to_igm.py",
    }

    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    nc_path = os.path.join(out_dir, f"{out_base}.nc")
    out.to_netcdf(nc_path)

    meta = {
        "exp": exp, "gid": gid, "source_file": os.path.abspath(path),
        "igm_input": f"{out_base}.nc",
        "dx_native": dx_nat, "dx_igm": float(dx_target), "factor": f,
        "y_descending_native": y_desc, "area_km2": float(area_km2),
        "t1": t1, "t2": t2, "t_dem": t_dem, "t_vel": t_vel, "t_da": t_da, "t_thk": t_thk,
        "dt": t2 - t1, "epochs_from_override": gid in GLACIER_OVERRIDES,
        "igm_shape": list(out.icemask.shape),
        "n_ice_cells_igm": int(out.icemask.values.sum()),
        "n_vel_obs_igm": int(np.isfinite(out.uvelsurfobs.values).sum()),
        "n_thkobs_igm": int(np.isfinite(out.thkobs.values).sum()),
        "n_init_thickness_plastic_native": n_plastic,
        "dhdt_gap_pixels_filled": int(n_dhdt_fill), "dhdt_gap_fraction_on_ice": float(n_dhdt_fill / max(ice.sum(), 1)),
        "thk_zero_fraction_on_ice": float(((ice & np.isfinite(thk) & (thk == 0)).sum()) / max(ice.sum(), 1)),
        "z_range_on_ice": [float(np.nanmin(np.where(icemask > 0, z_start, np.nan))),
                           float(np.nanmax(np.where(icemask > 0, z_start, np.nan)))],
    }
    with open(os.path.join(out_dir, f"{out_base}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return nc_path, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dx", type=float, default=None, help="target resolution (m); default 100, or 50 for <5 km2")
    ap.add_argument("--suffix", default="", help="append to the output name (gate/dev inputs), e.g. _dx200")
    args = ap.parse_args()
    for p in args.files:
        nc_path, m = convert(p, args.dx, suffix=args.suffix)
        print(f"{m['exp']}_{m['gid']}: {m['area_km2']:7.1f} km2  dx {m['dx_native']:g}->{m['dx_igm']:g} m  "
              f"grid {m['igm_shape']}  ice {m['n_ice_cells_igm']}  vel {m['n_vel_obs_igm']}  thkobs {m['n_thkobs_igm']}  "
              f"dhdt-filled {100 * m['dhdt_gap_fraction_on_ice']:.1f}%  "
              f"t1={m['t1']:.2f} t2={m['t2']:.2f} t_dem={m['t_dem']:.2f} t_da={m['t_da']:.2f}  "
              f"z {m['z_range_on_ice'][0]:.0f}-{m['z_range_on_ice'][1]:.0f}  -> {nc_path}")


if __name__ == "__main__":
    main()
