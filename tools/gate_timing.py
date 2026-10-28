#!/usr/bin/env python3
"""
Timing summary of tools/gate_run.sh logs (every line prefixed with epoch seconds).

    python tools/gate_timing.py results/_dev/G1/*.log

DA     = START -> first '[smb_inference] Profile inversion' line (includes ~15 s TF/emulator start-up)
SMB    = that line -> END; per-evaluation time from the 'L-BFGS eval N' lines
"""
import re
import sys


def parse(path):
    t = {"start": None, "smb_start": None, "end": None, "evals": [], "rc": None, "last": None}
    for ln in open(path, errors="replace"):
        m = re.match(r"^(\d{10})(?:\.\d+)?\s(.*)", ln.replace("\r", "\n").split("\n")[0])
        if not m:
            continue
        ts, rest = int(m.group(1)), m.group(2)
        t["last"] = ts
        if rest.startswith("START"):
            t["start"] = ts
        elif "Profile inversion" in rest and t["smb_start"] is None:
            t["smb_start"] = ts
        elif re.search(r"L-BFGS eval (\d+)", rest):
            t["evals"].append(ts)
        elif rest.startswith("END"):
            t["end"] = ts
            t["rc"] = rest.split("rc=")[-1]
    return t


def main():
    print(f"{'run':34s} {'DA (min)':>9s} {'SMB (min)':>10s} {'evals':>6s} {'s/eval':>7s} {'total (min)':>12s}  status")
    for p in sys.argv[1:]:
        t = parse(p)
        name = p.split("/")[-1].replace(".log", "")
        if t["start"] is None:
            continue
        end = t["end"] or t["last"]
        da = (t["smb_start"] - t["start"]) / 60 if t["smb_start"] else (end - t["start"]) / 60
        smb = (end - t["smb_start"]) / 60 if t["smb_start"] else float("nan")
        n = len(t["evals"])
        per = (t["evals"][-1] - t["smb_start"]) / n if n and t["smb_start"] else float("nan")
        print(f"{name:34s} {da:9.1f} {smb:10.1f} {n:6d} {per:7.1f} {(end - t['start']) / 60:12.1f}  "
              f"{'done rc=' + t['rc'] if t['end'] else 'running/killed'}")


if __name__ == "__main__":
    main()
