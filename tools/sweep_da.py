#!/usr/bin/env python3
"""
Run igm-smb-inference's Optuna DA sweep (optuna_sweep_da_nsga.py, unmodified)
from the ContinuIX folder: igm_run executes with cwd = ContinuIX and the
sweep database/trials land in --sweep-dir instead of the igm-smb-inference repo.

    python tools/sweep_da.py --sweep-dir results/EXP01_G03_vel/sweep \
        --experiment params_cx_EXP01_G03_vel_da --n-trials 30 --gpus 0,1
All other arguments are passed through to optuna_sweep_da_nsga.py.
"""
import argparse
import importlib.util
import os
import sys

# Detached runs inherit the login session's DISPLAY (ssh X11 forward). matplotlib's backend probe then
# blocks forever in XOpenDisplay when that forward stalls or the session closes -> hung trials. Never use X.
os.environ["MPLBACKEND"] = "Agg"
os.environ.pop("DISPLAY", None)
CX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_SCRIPT = "/home/klleshi/Documents/igm-smb-inference/optuna_sweep_da_nsga.py"

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--sweep-dir", required=True)
known, rest = ap.parse_known_args()
sweep_dir = os.path.abspath(known.sweep_dir)

spec = importlib.util.spec_from_file_location("optuna_sweep_da_nsga", SWEEP_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.PROJECT_DIR = CX_DIR          # cwd of every igm_run trial


def _paths(experiment):
    return {"sweep_dir": sweep_dir,
            "db_path": os.path.join(sweep_dir, "optuna_study.db"),
            "plots_dir": os.path.join(sweep_dir, "plots"),
            "study_name": f"da_tpe_{experiment}"}


mod._experiment_paths = _paths
sys.argv = [SWEEP_SCRIPT] + rest
mod.main()
