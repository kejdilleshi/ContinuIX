#!/usr/bin/env python3
"""
Gate G2 check: does the pipeline recover the SMB law of the steady-state synthetic S01?

The law is in the file attributes (ela_m, da_ds, max_a_m_yr):  SMB(z) = min(da_ds (z - ela), max_a).
Check only — nothing here feeds back into the method.

    python tools/check_s01.py GROUP_UNIL/EXP01/EXP01_S01_method01.nc [--src input_data/EXP01/EXP01_S01.nc]

Reports for the exported file (family mean, every member, and the stored profile):
  * ELA of every member vs 2350 m, ablation-zone gradient vs da_ds, accumulation plateau vs max_a
  * SMB(z) profile error (bias, RMSE) over the glacier's elevation range, per member and for the mean
  * pixel-level SMB and FDIV (= SMB for a steady state) against the law at the mid-window surface
  * whether the truth lies inside SMB +- UNCT_SMB (fraction of pixels)
"""
import argparse

import numpy as np
import xarray as xr


def law(z, ela, g, cap):
    return np.minimum(g * (z - ela), cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--src", default="input_data/EXP01/EXP01_S01.nc")
    a = ap.parse_args()
    src, out = xr.open_dataset(a.src), xr.open_dataset(a.export)
    ela, g, cap = float(src.attrs["ela_m"]), float(src.attrs["da_ds"]), float(src.attrs["max_a_m_yr"])
    print(f"truth (file attrs): ELA {ela:g} m, gradient {g:g} yr-1, cap {cap:g} m/yr")

    ice = (np.nan_to_num(src.ICEMASK.values) > 0.5) & (np.nan_to_num(src.THK.values) > 0)
    zs = src.DEM.values[ice]
    zlo, zhi = np.nanpercentile(zs, 1), np.nanpercentile(zs, 99)
    z = out.z.values
    sel = (z >= zlo) & (z <= zhi)
    print(f"glacier elevation range (1-99 %): {zlo:.0f}-{zhi:.0f} m;  {sel.sum()} profile nodes inside")

    truth_p = law(z, ela, g, cap)
    rows = []
    for i in range(out.sizes["member"]):
        p = out.MEMBER_SMB_PROFILE.values[i]
        ok = sel & np.isfinite(p)
        abl = ok & (z < ela - 200) & (z > zlo)
        slope = np.polyfit(z[abl], p[abl], 1)[0] if abl.sum() >= 2 else np.nan
        acc = ok & (z > ela + 400)
        rows.append((i, float(out.MEMBER_TAU_REF[i]), float(out.MEMBER_VELSURF[i]), bool(out.MEMBER_ADMISSIBLE[i]),
                     float(out.MEMBER_ELA[i]), slope, float(np.nanmean(p[acc])) if acc.any() else np.nan,
                     float(np.mean(p[ok] - truth_p[ok])), float(np.sqrt(np.mean((p[ok] - truth_p[ok]) ** 2)))))
    print("\nmember  tau_ref  velsurf  adm   ELA(m)  ablation-slope  plateau  bias   RMSE   (profile vs law, m i.e./yr)")
    for r in rows:
        print(f"  {r[0]:3d}   {r[1]:6.3f}  {r[2]:7.2f}  {int(r[3])}   {r[4]:7.1f}   {r[5]:10.4f}   {r[6]:7.2f}  {r[7]:+.2f}  {r[8]:.2f}")
    pm = out.SMB_PROFILE.values
    ok = sel & np.isfinite(pm)
    print(f"admissible mean profile: bias {np.mean(pm[ok] - truth_p[ok]):+.3f}, RMSE {np.sqrt(np.mean((pm[ok] - truth_p[ok]) ** 2)):.3f} m i.e./yr; "
          f"ELA range of admissible members {out.attrs.get('ELA_admissible_m')}")

    # pixel level
    dt = 10.0
    zbar = src.DEM.values + np.nan_to_num(src.DHDT.values) * (0.5 * (0 + dt) - 5.0)   # S01: window 0-10, DEM at 5
    tr = law(zbar, ela, g, cap)
    smb, fdiv, unc = out.SMB.values, out.FDIV.values, out.UNCT_SMB.values
    m = ice & np.isfinite(smb) & np.isfinite(tr)
    e = smb[m] - tr[m]
    print(f"\npixel SMB vs law: n={m.sum()}  bias {e.mean():+.3f}  RMSE {np.sqrt((e ** 2).mean()):.3f}  "
          f"R2 {1 - (e ** 2).sum() / ((tr[m] - tr[m].mean()) ** 2).sum():.4f}  "
          f"|err|<UNCT_SMB: {np.mean(np.abs(e) <= unc[m]):.2f}   median UNCT_SMB {np.median(unc[m]):.3f}")
    mf = m & np.isfinite(fdiv)
    ef = fdiv[mf] - tr[mf]
    print(f"pixel FDIV vs law (steady state: FDIV = SMB): bias {ef.mean():+.3f}  RMSE {np.sqrt((ef ** 2).mean()):.3f}  "
          f"|err|<UNCT_FDIV: {np.mean(np.abs(ef) <= out.UNCT_FDIV.values[mf]):.2f}")
    mean_dhdt_in = float(np.nanmean(np.where(ice, src.DHDT.values, np.nan)))
    print(f"input DHDT mean on ice {mean_dhdt_in:+.4f} m/yr; exported mean(SMB-FDIV) {np.nanmean(smb[mf] - fdiv[mf]):+.4f}; "
          f"glacier-wide SMB {np.nanmean(smb[m]):+.3f} (law {np.nanmean(tr[m]):+.3f})")
    print("components (mean): family %.3f  repr %.3f  sigma_dhdt %.3f   sources: DHDT %s" % (
        np.nanmean(out.COMP_FAMILY_SPREAD_SMB.values), np.nanmean(out.COMP_REPRESENTATION_ERROR.values),
        np.nanmean(out.COMP_SIGMA_DHDT.values), out.attrs.get("sigma_DHDT_source")))


if __name__ == "__main__":
    main()
