#!/usr/bin/env python3
"""
SMB-phase rerun of every finished `cnn` experiment (tools/smb_rerun.py), one lane per GPU.

The experiments of tools/queue.txt (variant cnn, with a finished family/) are sorted by grid size (smallest
first) and dealt to the lanes alternately; each lane reruns its experiments one after the other, the members of one experiment
concurrently (5 for small grids, 2 at ~23k cells, 1 for G01), and exports each family to
results/<EXP>_<GID>_cnn/smb_<tag>/export/ (GROUP_UNIL/ is not touched). An experiment whose export exists is
skipped, so a lane can be relaunched.

    setsid nohup python tools/smb_batch.py --tag v2 --lane 0 --gpu 0 --set ... > logs/smb_v2_lane0.log 2>&1 &
    setsid nohup python tools/smb_batch.py --tag v2 --lane 1 --gpu 1 --set ... > logs/smb_v2_lane1.log 2>&1 &
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

os.environ["MPLBACKEND"] = "Agg"
os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)


def experiments():
    out = []
    for ln in open(os.path.join(HERE, "queue.txt")):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        exp, gid, variant = ln.split()[:3]
        if variant != "cnn":
            continue
        fam = os.path.join(CX_DIR, "results", f"{exp}_{gid}_cnn", "family")
        if not glob.glob(os.path.join(fam, "slid_*", "smb_inference", "smb_vec.npy")):
            continue
        meta = json.load(open(os.path.join(CX_DIR, "data", f"{exp}_{gid}_da.json")))
        out.append((meta["igm_shape"][0] * meta["igm_shape"][1], exp, gid))
    return sorted(out)


def jobs_for(ncell):
    return 5 if ncell <= 10000 else 2 if ncell <= 40000 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these EXP_GID names")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()

    todo = experiments()
    if a.only:
        todo = [t for t in todo if f"{t[1]}_{t[2]}" in a.only]
    # smallest first (quick feedback on S01/S02 and the small glaciers); alternate dealing keeps the lanes balanced
    mine = [t for i, t in enumerate(todo) if i % a.lanes == a.lane]
    print(f"[smb_batch] lane {a.lane} (GPU {a.gpu}): {len(mine)} experiments, set={a.set}", flush=True)
    for ncell, exp, gid in mine:
        root = os.path.join(CX_DIR, "results", f"{exp}_{gid}_cnn", f"smb_{a.tag}")
        if os.path.exists(os.path.join(root, "export", f"{exp}_{gid}_method01.nc")):
            print(f"[smb_batch] {exp} {gid}: done, skip", flush=True)
            continue
        t0 = time.time()
        print(f"[smb_batch] {exp} {gid} ({ncell} cells) start {time.strftime('%H:%M:%S')}", flush=True)
        rc = subprocess.run([sys.executable, os.path.join(HERE, "smb_rerun.py"), "--exp", exp, "--gid", gid,
                             "--tag", a.tag, "--gpus", a.gpu, "--jobs", str(jobs_for(ncell)), "--export",
                             "--set", *a.set], cwd=CX_DIR).returncode
        print(f"[smb_batch] {exp} {gid} exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
