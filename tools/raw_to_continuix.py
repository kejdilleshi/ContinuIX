#!/usr/bin/env python3
"""
Assemble the raw EXP02 data of one real glacier (GeoTIFFs, GPR points, outline shapefiles) into an
input-format NetCDF `input_data/EXP02/EXP02_<GID>.nc` that the same pipeline (continuix_to_igm.py ->
run_continuix.py) consumes like EXP01_<GID>.nc. S02's raw file is already a NetCDF on the EXP01 grid
(EXP02_S02_all.nc); it is only linked (see main()).

Grid: the EXP01 grid of the glacier (the protocol asks for results "on the same georeferenced grid as the
input data"; the raw rasters sit on different grids from 2 to 200 m, the EXP01 grid is the one common,
well-defined choice and is what the exporter maps back to).

What is done (all of it is documented in the method README):
  * every raster is brought to the EXP01 grid with GDAL resampling: area-average when it is finer than
    the grid (DEM, DHDT), bilinear when it is coarser (velocity, BED); no-data stays NaN;
  * on-ice DEM gaps (the model needs a complete surface) are filled by inverse-distance interpolation (<= 500 m);
    every other gap stays NaN -- nothing is zero-filled and DHDT / velocity / thickness are NOT interpolated here
    (gap handling for those is in continuix_to_igm.py);
  * several DEM epochs: each DEM is shifted to the mean epoch with the observed DHDT and the valid ones
    averaged (a pixel seen at only some epochs is not biased); several DHDT windows: time-weighted mean
    over the valid windows, window = first start .. last end (a provided full-window DHDT is used as is);
  * several velocity periods: mean of the valid ones (as the provided EXP01 velocity, an average);
  * thickness: the provided raster, or GPR points averaged per cell (cells without GPR stay NaN); for G05, whose points
    mix surveys from 2009-2020, the per-cell mean bed elevation z_bed is subtracted from the combined DEM instead;
  * ICEMASK from the outlines: for each pixel the last outline year containing it (EXP01 definition).
  * vertical datum: G01 only (Iceland: ellipsoidal DEM vs orthometric BED/THK): DEM shifted by the median of DEM - BED - THK (thinning-corrected) when > 20 m;
Output variables: DEM, DHDT, VX, VY, THK, BED, ICEMASK with EXP01-style `timestamp` attributes.

    python tools/raw_to_continuix.py G01 G02 G03 G04 G05 G06
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)
RAW = os.path.join(CX_DIR, "input_data", "EXP02")
E01 = os.path.join(CX_DIR, "input_data", "EXP01")


def dec_year(tok):
    """'201310' -> 2013.79 ; '2013' -> 2013.0 (EXP01 convention: a bare year is its start) ; '20170901' -> 2017.67"""
    tok = str(tok)
    if len(tok) == 8:
        return int(tok[:4]) + (int(tok[4:6]) - 1) / 12 + (int(tok[6:]) - 1) / 365.25
    if len(tok) == 6:
        return int(tok[:4]) + (int(tok[4:6]) - 0.5) / 12
    return float(int(tok))


def to_token(y):
    """decimal year -> YYYYMMDD token that continuix_to_igm.parse_epochs decodes back to ~y."""
    yr = int(np.floor(y)); f = y - yr
    m = min(int(f * 12), 11); d = min(int(round((f * 12 - m) * 30.4)), 28) + 1
    return f"{yr:04d}{m + 1:02d}{d:02d}"


def ts_attr(y0, y1=None):
    return f"{to_token(y0)}-{to_token(y1)}" if y1 is not None and y1 != y0 else to_token(y0)


def files(gid, kind):
    return sorted(f for f in glob.glob(os.path.join(RAW, gid, f"EXP02_{gid}_{kind}*.tif"))
                  if not re.search(rf"{kind}[-_]?(UNCT|FLAGGED)", os.path.basename(f)))


def epochs_in(name, kind, gid):
    """decimal-year epoch(s) from a file name: the dates after `kind` (one for a DEM, two for a window)"""
    stem = os.path.basename(name)[:-4].split(f"{gid}_{kind}", 1)[1]
    toks = re.findall(r"\d{8}|\d{6}|\d{4}", stem)
    return [dec_year(t) for t in toks]


def warp(path, grid, resampling="average"):
    """Read `path` resampled onto the target grid (NaN where invalid)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    ny, nx, x0, y1, dx = grid["ny"], grid["nx"], grid["x0"], grid["y1"], grid["dx"]
    from rasterio.transform import from_origin
    dst_tf = from_origin(x0, y1, dx, dx)
    with rasterio.open(path) as src:
        nd = src.nodata if src.nodata is not None else np.nan
        # float vrt with explicit NaN no-data; zero is NOT no-data
        vrt = WarpedVRT(src, crs=grid["crs"], transform=dst_tf, width=nx, height=ny,
                        resampling=getattr(Resampling, resampling), src_nodata=nd, nodata=np.nan, dtype="float32")
        a = vrt.read(1, masked=False).astype(np.float64)
        vrt.close()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e20] = np.nan
    return a


def points_to_grid(shp, col, grid):
    """Mean of GPR point values per cell; NaN where no point."""
    import geopandas as gpd
    p = gpd.read_file(shp).to_crs(grid["crs"])
    p = p[np.isfinite(p[col].astype(float))]
    ix = np.floor((p.geometry.x.values - grid["x0"]) / grid["dx"]).astype(int)
    iy = np.floor((grid["y1"] - p.geometry.y.values) / grid["dx"]).astype(int)
    ok = (ix >= 0) & (ix < grid["nx"]) & (iy >= 0) & (iy < grid["ny"])
    s = np.zeros((grid["ny"], grid["nx"])); n = np.zeros_like(s)
    np.add.at(s, (iy[ok], ix[ok]), p[col].astype(float).values[ok]); np.add.at(n, (iy[ok], ix[ok]), 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), int(ok.sum()), len(p)


def icemask_from_outlines(gid, grid):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    tf = from_origin(grid["x0"], grid["y1"], grid["dx"], grid["dx"])
    out = np.zeros((grid["ny"], grid["nx"]), np.float32)
    for f in sorted(glob.glob(os.path.join(RAW, gid, f"EXP02_{gid}_outline_*.shp"))):
        yr = int(re.search(r"outline_(\d{4})", f).group(1))
        g = gpd.read_file(f).to_crs(grid["crs"])
        m = rasterize(((geom, 1) for geom in g.geometry if geom is not None), out_shape=out.shape, transform=tf,
                      fill=0, dtype="uint8", all_touched=False).astype(bool)
        out[m] = np.maximum(out[m], yr)
    return out


# per-glacier choices that cannot be read from file names ----------------------------------------------------
# thickness: ('raster', kind) -> raster(s) with that name; ('points', shapefile stem, column)
CFG = {
    "G01": {"thk": ("raster", "THK"), "thk_pick": "2020", "datum_check": True},   # 2013/2020/2023 rasters: middle epoch; Iceland DEM is ellipsoidal
    "G02": {"thk": ("points", "THK_20120809", "thickness"), "thk_epoch": 2012.69},
    "G03": {"thk": ("points", "THK_20170215", "thickness"), "thk_epoch": 2017.12},
    "G04": {"thk": ("raster", "THK"), "thk_pick": "2021"},
    # G05: the points mix surveys of 2009, 2011 (VAW airborne GPR) and 2020 (GLAMOS); their thickness refers to each
    # survey's own surface, so the time-invariant z_bed is used instead: THK = DEM(t_mean) - cell-mean z_bed
    "G05": {"thk": ("points_bed", "THK", "z_bed")},
    "G06": {"thk": ("raster", "THK"), "thk_pick": "2006"},
}


def assemble(gid):
    ref = xr.open_dataset(os.path.join(E01, f"EXP01_{gid}.nc"))
    ydesc = bool(ref.y.values[1] < ref.y.values[0])
    x, y = ref.x.values.astype(float), ref.y.values.astype(float)
    dx = abs(float(x[1] - x[0]))
    grid = {"nx": x.size, "ny": y.size, "dx": dx, "x0": x.min() - dx / 2, "y1": y.max() + dx / 2,
            "crs": ref.spatial_ref.attrs.get("crs_wkt") or ref.spatial_ref.attrs.get("spatial_ref")}
    assert ydesc, "EXP01 grids are north-up (descending y)"
    print(f"[{gid}] EXP01 grid {grid['nx']}x{grid['ny']} dx {dx:g}")
    cfg = CFG[gid]
    rep = {}
    ice = icemask_from_outlines(gid, grid)

    # ---- DHDT: windows --------------------------------------------------------------------------------
    dh_files = files(gid, "DHDT")
    wins = [(f, *epochs_in(f, "DHDT", gid)[:2]) for f in dh_files]
    full = max(wins, key=lambda w: w[2] - w[1])            # a provided full-window DHDT (e.g. G04 2021-2025)
    t0, t1 = min(w[1] for w in wins), max(w[2] for w in wins)
    if full[1] <= t0 + 1e-6 and full[2] >= t1 - 1e-6:
        use = [full]
    else:
        use = wins
    num = np.zeros((grid["ny"], grid["nx"])); den = np.zeros_like(num)
    for f, a, b in use:
        d = warp(f, grid, "average")
        ok = np.isfinite(d); num[ok] += d[ok] * (b - a); den[ok] += (b - a)
    dhdt = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    w0, w1 = min(w[1] for w in use), max(w[2] for w in use)
    rep["DHDT"] = f"windows {[(round(a, 2), round(b, 2)) for _, a, b in use]} -> {w0:.2f}-{w1:.2f}; finite {100 * np.isfinite(dhdt).mean():.0f}% of grid"

    # ---- DEM: epochs shifted to their mean epoch -----------------------------------------------------
    dem_files = files(gid, "DEM")
    ep = [epochs_in(f, "DEM", gid)[0] for f in dem_files]
    t_mean = float(np.mean(ep))
    stack = []
    for f, t in zip(dem_files, ep):
        z = warp(f, grid, "average")
        z[z == 0] = np.nan                      # raw DEMs carry zeros as gaps (e.g. G02 2006: 56 %)
        stack.append(z + np.where(np.isfinite(dhdt), dhdt, 0.0) * (t_mean - t))
    dem = np.nanmean(np.array(stack), axis=0) if len(stack) > 1 else stack[0]
    n_gap = int(((ice > 0) & ~np.isfinite(dem)).sum())
    if n_gap:      # the model needs a complete surface: inverse-distance fill of on-ice DEM gaps (<= 500 m reach)
        from rasterio.fill import fillnodata
        valid = np.isfinite(dem)
        filled = fillnodata(np.where(valid, dem, 0.0).astype("float32"), mask=valid.astype("uint8"),
                            max_search_distance=max(5, int(500 / dx)), smoothing_iterations=0).astype(float)
        fill_ok = (ice > 0) & ~valid & (filled != 0)
        dem = np.where(fill_ok, filled, dem)
        rep["DEM gap fill"] = f"{n_gap} on-ice cells without DEM; {int(fill_ok.sum())} filled by inverse-distance (<=500 m), rest left for the converter"
    rep["DEM"] = f"{len(dem_files)} epoch(s) {[round(t, 2) for t in ep]} -> {t_mean:.2f}; finite {100 * np.isfinite(dem).mean():.0f}%"

    # ---- BED, velocity, thickness ------------------------------------------------------------------------
    bed = warp(files(gid, "BED")[0], grid, "bilinear")
    vx_f = [f for f in glob.glob(os.path.join(RAW, gid, f"EXP02_{gid}_VX*.tif")) if "UNCT" not in f]
    vy_f = [f.replace("_VX", "_VY") for f in vx_f]
    vx = np.nanmean(np.array([warp(f, grid, "bilinear") for f in sorted(vx_f)]), axis=0)
    vy = np.nanmean(np.array([warp(f, grid, "bilinear") for f in sorted(vy_f)]), axis=0)
    vper = [epochs_in(f, "VX", gid)[:2] for f in sorted(vx_f)]
    vt = [min(p[0] for p in vper), max(p[-1] for p in vper)] if vper and all(vper) else None
    rep["VX/VY"] = f"{len(vx_f)} period(s); finite {100 * np.isfinite(vx).mean():.0f}% of grid"

    if cfg["thk"][0] == "raster":
        tf = [f for f in files(gid, "THK") if cfg["thk_pick"] in os.path.basename(f)][0]
        thk = warp(tf, grid, "average"); t_thk = epochs_in(tf, "THK", gid)[0]
        rep["THK"] = f"raster {os.path.basename(tf)}; finite {100 * np.isfinite(thk).mean():.0f}%"
    elif cfg["thk"][0] == "points_bed":
        zb, nin, nall = points_to_grid(os.path.join(RAW, gid, f"EXP02_{gid}_{cfg['thk'][1]}.shp"), cfg["thk"][2], grid)
        thk = dem - zb; t_thk = t_mean
        rep["THK"] = (f"GPR bed points {nin}/{nall} in grid -> {np.isfinite(thk).sum()} cells; THK = DEM({t_mean:.2f}) - z_bed, "
                      f"{int((thk <= 0).sum())} cells <= 0")
    else:
        thk, nin, nall = points_to_grid(os.path.join(RAW, gid, f"EXP02_{gid}_{cfg['thk'][1]}.shp"), cfg["thk"][2], grid)
        t_thk = cfg["thk_epoch"]
        rep["THK"] = f"GPR points {nin}/{nall} in grid -> {np.isfinite(thk).sum()} cells (NaN elsewhere)"

    if cfg.get("datum_check"):
        # vertical-datum consistency of DEM vs BED/THK from the raw data alone (G01: DEM is ellipsoidal, BED/THK
        # orthometric, ~67 m apart): DEM - BED - THK(thinning-corrected to the DEM epoch) should be ~0 on ice
        ok = (ice > 0) & np.isfinite(dem) & np.isfinite(bed) & np.isfinite(thk)
        if ok.sum() > 100:
            expect = thk + np.where(np.isfinite(dhdt), dhdt, 0.0) * (t_mean - t_thk)
            off = float(np.median((dem - bed - expect)[ok]))
            if abs(off) > 20.0:
                dem = dem - off
                rep["DEM datum"] = f"DEM - BED - THK median {off:+.1f} m -> DEM shifted by {-off:+.1f} m"
            else:
                rep["DEM datum"] = f"DEM - BED - THK median {off:+.1f} m (consistent, no shift)"
    rep["ICEMASK"] = f"{int((ice > 0).sum())} cells, years {sorted(set(np.unique(ice[ice > 0]).astype(int)))}"
    for k, v in rep.items():
        print(f"   {k:8s} {v}")

    # ---- write (north-up like EXP01), gaps as NaN -------------------------------------------------------
    def v(a, long, units, ts=None, grid_mapping=True):
        at = {"long_name": long, "units": units, "grid_mapping": "spatial_ref"}
        if ts:
            at["timestamp"] = ts
        return (("y", "x"), a.astype(np.float32), at)

    ds = xr.Dataset(
        {"DEM": v(dem, "Surface elevation (EXP02 raw, epochs combined)", "m", ts_attr(t_mean)),
         "DHDT": v(dhdt, "Surface elevation change (EXP02 raw, gaps = NaN)", "m/year", ts_attr(w0, w1)),
         "VX": v(vx, "x surface velocity (EXP02 raw, gaps = NaN)", "m/year", ts_attr(*([vt[0], vt[1]] if vt else [t_mean]))),
         "VY": v(vy, "y surface velocity (EXP02 raw, gaps = NaN)", "m/year", ts_attr(*([vt[0], vt[1]] if vt else [t_mean]))),
         "THK": v(thk, "Ice thickness (EXP02 raw: provided raster or GPR points, gaps = NaN)", "m", ts_attr(t_thk)),
         "BED": v(bed, "Bed elevation (EXP02 raw)", "m"),
         "ICEMASK": v(np.where(ice > 0, ice, np.nan), "Ice mask: last outline year", "yr")},
        coords={"x": ref.x, "y": ref.y})
    ds["spatial_ref"] = ref["spatial_ref"]
    ds.attrs = {"title": f"EXP02 raw data assembled on the EXP01 grid, {gid}", "history": "tools/raw_to_continuix.py",
                "gaps": "NaN (not zero-filled); no interpolation in this file"}
    out = os.path.join(RAW, f"EXP02_{gid}.nc")
    ds.to_netcdf(out)
    print(f"[{gid}] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gids", nargs="+")
    a = ap.parse_args()
    for g in a.gids:
        if g == "S02":
            link = os.path.join(RAW, "EXP02_S02.nc")
            if not os.path.lexists(link):
                os.symlink(os.path.join("S02", "EXP02_S02_all.nc"), link)
            print(f"[S02] linked {link}")
        else:
            assemble(g)


if __name__ == "__main__":
    main()
