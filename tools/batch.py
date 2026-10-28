#!/usr/bin/env python3
"""
Background batch runner for the ContinuIX runs on one node.

Each lane owns one GPU and repeatedly takes the next pending job from
tools/queue.txt ("EXP01 G03 vel" per line, '#' comments allowed), runs
tools/run_continuix.py --resume on it, and records the outcome in
results/batch_state.json (shared between lanes under a file lock). The queue
file is re-read before every pick, so jobs can be appended while it runs.
Re-launching a lane resumes: a job left "running" by a dead lane is retried,
and run_continuix skips the stages it already completed.

Launch (survives closing the Claude/ssh session):
    cd ~/Documents/ContinuIX
    setsid nohup python tools/batch.py --gpu 0 > logs/lane0.log 2>&1 < /dev/null &
    setsid nohup python tools/batch.py --gpu 1 > logs/lane1.log 2>&1 < /dev/null &
Status:
    python tools/batch.py --status
Stop a lane: kill its batch.py PID (the current igm_run finishes or is killed
with it); the job is retried on the next launch.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time

import run_continuix as cx   # gpu_free_mib / member_mem_mib

# Detached runs inherit the login session's DISPLAY (ssh X11 forward). matplotlib's backend probe then
# blocks forever in XOpenDisplay when that forward stalls or the session closes -> hung trials. Never use X.
os.environ["MPLBACKEND"] = "Agg"
os.environ.pop("DISPLAY", None)
HERE = os.path.dirname(os.path.abspath(__file__))
CX_DIR = os.path.dirname(HERE)
QUEUE = os.path.join(HERE, "queue.txt")
STATE = os.path.join(CX_DIR, "results", "batch_state.json")
LOCK = STATE + ".lock"
LOGS = os.path.join(CX_DIR, "logs")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ONLY = None     # set from --only: restrict the lane to these EXP names (queue.txt itself is untouched)


def read_queue():
    jobs = []
    for ln in open(QUEUE):
        ln = ln.split("#")[0].strip()
        if ln:
            exp, gid, variant = ln.split()[:3]
            if ONLY is None or exp in ONLY:
                jobs.append(f"{exp}_{gid}_{variant}")
    return jobs


class Locked:
    def __enter__(self):
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        self.fh = open(LOCK, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        return self.st

    def __exit__(self, *exc):
        with open(STATE + ".tmp", "w") as f:
            json.dump(self.st, f, indent=2)
        os.replace(STATE + ".tmp", STATE)
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def job_need(job):
    """GPU memory (MiB) a job needs to start: one family member / DA process plus margin (0 if not converted yet)."""
    exp, gid, _ = job.split("_")
    mp = os.path.join(CX_DIR, "data", f"{exp}_{gid}.json")
    if not os.path.exists(mp):
        return 0
    meta = json.load(open(mp))
    return max(cx.member_mem_mib(meta), cx.da_mem_mib(meta)) + 1000


def pick(gpu):
    """Next job that is not done/failed/owned by a live lane AND fits in the GPU's free memory now (the cluster
    is shared). Returns "WAIT" if pending jobs exist but none fits, None if the queue is exhausted."""
    blocked = False
    with Locked() as st:
        for job in read_queue():
            j = st.get(job, {})
            status = j.get("status")
            if status in ("done", "failed"):
                continue
            if status == "running" and pid_alive(j.get("pid")):
                continue                      # owned by a live lane
            if cx.gpu_free_mib(gpu) < job_need(job):
                blocked = True
                continue
            st[job] = {"status": "running", "gpu": gpu, "pid": os.getpid(),
                       "start": now(), "attempts": j.get("attempts", 0) + 1}
            return job
    return "WAIT" if blocked else None


def finish(job, rc, t0):
    with Locked() as st:
        st[job].update({"status": "done" if rc == 0 else "failed", "rc": rc,
                        "end": now(), "hours": round((time.time() - t0) / 3600, 2)})


def concurrency_for(exp, gid):
    """(Optuna workers, L-curve jobs, family members) run concurrently on the lane's GPU.
    The runs are CPU/latency-bound (GPU ~idle, 12 cores for 2 lanes), so small grids share the card;
    the limit is GPU memory (24 GB): the SMB reverse pass needs ~3 GB (G03), ~9 GB (G05/S02 size),
    ~17 GB (G01 at 100 m), measured in gates G1/G3."""
    meta = os.path.join(CX_DIR, "data", f"{exp}_{gid}.json")
    n = json.load(open(meta))["n_ice_cells_igm"] if os.path.exists(meta) else 0
    if n > 20000:       # G01 (83k cells): DA ~6 GB, SMB ~17 GB -> one SMB at a time
        return "2", "2", "1"
    if n > 3000:        # G05 / S02 (8k cells)
        return "3", "3", "2"
    return "3", "3", "5"


def room_for(job, gpu, fh):
    """The cluster is shared: do not start (or over-subscribe) a job on a GPU other users have filled.
    Waits until one family member and one DA process fit, and cuts the Optuna workers / L-curve jobs to what
    fits in the free memory now. Returns (workers, lcurve_jobs, family_parallel); run_continuix then sizes its
    family waves from the free memory at that stage."""
    exp, gid, _ = job.split("_")
    w, j, fp = (int(x) for x in concurrency_for(exp, gid))
    meta = json.load(open(os.path.join(CX_DIR, "data", f"{exp}_{gid}.json")))
    dam, mem = cx.da_mem_mib(meta), cx.member_mem_mib(meta)
    need, waited = max(mem, dam) + 1000, False
    while cx.gpu_free_mib(gpu) < need:
        if not waited:
            fh.write(f"##### {now()} waiting for GPU{gpu} memory: need {need} MiB, free {cx.gpu_free_mib(gpu)} MiB\n"); fh.flush()
            print(f"[{now()}] lane gpu{gpu}: {job} waits for GPU memory (need {need} MiB, free {cx.gpu_free_mib(gpu)})", flush=True)
            waited = True
        time.sleep(300)
    fit = max(1, (cx.gpu_free_mib(gpu) - 1000) // dam)
    w, j = min(w, fit), min(j, fit)
    fh.write(f"##### {now()} {'GPU free after wait; ' if waited else ''}workers {w}, L-curve jobs {j}, family<= {fp}\n"); fh.flush()
    return str(w), str(j), str(fp)


def run_job(job, gpu):
    exp, gid, variant = job.split("_")
    log = os.path.join(LOGS, f"{job}.log")
    base = [sys.executable, os.path.join(HERE, "run_continuix.py"), "--exp", exp, "--gid", gid,
            "--variant", variant, "--gpus", str(gpu), "--resume"]
    with open(log, "a") as fh:
        fh.write(f"\n##### {now()} lane gpu{gpu} start {job}\n"); fh.flush()
        # convert first so the grid size (-> parallelism) is known
        rc = subprocess.call(base + ["--to-stage", "yaml"], cwd=CX_DIR, stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0:
            w, j, fp = room_for(job, gpu, fh)
            rc = subprocess.call(base + ["--workers-per-gpu", w, "--lcurve-jobs", j, "--family-parallel", fp],
                                 cwd=CX_DIR, stdout=fh, stderr=subprocess.STDOUT)
        fh.write(f"##### {now()} exit {rc}\n")
    return rc


def status():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    q = read_queue()
    counts = {}
    for job in q:
        s = st.get(job, {}).get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        if s != "pending":
            j = st[job]
            print(f"{job:22s} {s:8s} gpu{j.get('gpu')} start {j.get('start')}  "
                  f"{('end ' + j['end'] + '  ' + str(j['hours']) + ' h') if 'end' in j else ''}")
    print("summary:", counts, f"(queue {len(q)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of EXP names to run, e.g. EXP01 (default: whole queue)")
    ap.add_argument("--retry-failed", action="store_true", help="reset failed jobs to pending")
    a = ap.parse_args()
    global ONLY
    ONLY = set(a.only.split(",")) if a.only else None
    if a.status:
        return status()
    if a.retry_failed:
        with Locked() as st:
            for k, v in st.items():
                if v.get("status") == "failed":
                    v["status"] = "pending"
        return
    os.makedirs(LOGS, exist_ok=True)
    print(f"[{now()}] lane gpu{a.gpu} pid {os.getpid()} up", flush=True)
    while True:
        job = pick(a.gpu)
        if job == "WAIT":
            print(f"[{now()}] lane gpu{a.gpu}: no pending job fits the free GPU memory ({cx.gpu_free_mib(a.gpu)} MiB), waiting", flush=True)
            time.sleep(300)
            continue
        if job is None:
            print(f"[{now()}] lane gpu{a.gpu}: queue empty, exiting", flush=True)
            return
        print(f"[{now()}] lane gpu{a.gpu}: start {job}", flush=True)
        t0 = time.time()
        rc = run_job(job, a.gpu)
        finish(job, rc, t0)
        print(f"[{now()}] lane gpu{a.gpu}: {job} exit {rc} ({(time.time() - t0) / 3600:.2f} h)", flush=True)


if __name__ == "__main__":
    main()
