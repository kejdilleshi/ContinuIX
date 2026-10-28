#!/usr/bin/env python3
"""
Loss terms of the SMB inference along a uniform offset of a finished profile: b(z) + delta.

For one member of results/<EXP>_<GID>_cnn/smb_<tag>/, every delta is one evaluation-only run (tools/smb_rerun.py,
optimization.nbitmax=0, regularisation 0) with the settings of that run, so the logged loss minus the logged data
term is the geodetic term alone. Shows where the band misfit and the geodetic term each put the glacier-wide SMB.

  python tools/geod_scan.py --exp EXP01 --gid S01 --member slid_1.8770 --deltas -0.3 0 0.3 0.6
  -> results/<EXP>_<GID>_cnn/geod_scan/<member>/scan.json (+ profile_<delta>.csv, runs smb_geodscan_<member>_<delta>/)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

os.environ["MPLBACKEND"] = "Agg"          # see run_continuix.py: never let a detached run probe X
os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gid", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--tag", default="G_bands_geod", help="run whose profile and settings are scanned")
    ap.add_argument("--deltas", nargs="+", type=float, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--parallel", type=int, default=4)
    a = ap.parse_args()

    root = os.path.join(CX_DIR, "results", f"{a.exp}_{a.gid}_cnn")
    src = os.path.join(root, f"smb_{a.tag}")
    sd = os.path.join(src, a.member, "smb_inference")
    vec = np.load(os.path.join(sd, "smb_vec.npy")).astype(float)
    z0, dz = float(np.load(os.path.join(sd, "z_min.npy"))), float(np.load(os.path.join(sd, "dz.npy")))
    nodes = z0 + dz * np.arange(vec.size)
    sets = [s for s in json.load(open(os.path.join(src, "settings.json")))["set"]
            if not s.startswith(("optimization.nbitmax", "optimization.regularisation", "inversion.init_profile"))]
    out = os.path.join(root, "geod_scan", a.member)
    os.makedirs(out, exist_ok=True)

    gpus = [g for g in a.gpus.split(",") if g]
    pending, running, tags = list(a.deltas), [], {}
    while pending or running:
        while pending and len(running) < a.parallel:
            d = pending.pop(0)
            csv = os.path.join(out, f"profile_{d:+.2f}.csv")
            np.savetxt(csv, np.c_[nodes, vec + d], delimiter=",", fmt="%.6f",
                       header=f"z,smb  {a.tag}/{a.member} profile {d:+.2f} m i.e./yr")
            tag = f"geodscan_{a.member}_{d:+.2f}"
            tags[d] = tag
            cmd = [sys.executable, os.path.join(HERE, "smb_rerun.py"), "--exp", a.exp, "--gid", a.gid, "--tag", tag,
                   "--members", a.member, "--gpus", gpus[len(tags) % len(gpus)], "--jobs", "1", "--set", *sets,
                   f"inversion.init_profile={csv}", "optimization.nbitmax=0", "optimization.regularisation=0"]
            running.append(subprocess.Popen(cmd, cwd=CX_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        time.sleep(5)
        running = [p for p in running if p.poll() is None]

    rows = []
    for d in a.deltas:
        log = open(os.path.join(root, f"smb_{tags[d]}", a.member + ".log")).read()
        m = re.search(r"Evaluation only: loss=(\S+), data=(\S+)", log)
        g = re.search(r"geodetic constraint: glacier-wide SMB -> (\S+) m/yr \(weight (\S+)\)", log)
        if not m:
            print(f"[geod_scan] {d:+.2f}: no result, see {tags[d]}"); continue
        loss, data = float(m.group(1)), float(m.group(2))
        rows.append({"delta": d, "loss": loss, "data": data, "geodetic": loss - data,
                     "target": float(g.group(1)) if g else None, "weight": float(g.group(2)) if g else None})
        print(f"[geod_scan] delta {d:+.2f}: data {data:.4f}, geodetic {loss - data:.4f}")
    json.dump({"exp": a.exp, "gid": a.gid, "member": a.member, "tag": a.tag, "rows": rows},
              open(os.path.join(out, "scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
