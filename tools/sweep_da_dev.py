#!/usr/bin/env python3
"""
sweep_da.py for the dev stack (tools/devstack.py): igm-smb-inference's Optuna DA sweep
(optuna_sweep_da_nsga.py, unmodified) driving igm_run on IGM dev + the igm-smb-inference-user module.

The sweep script hard-codes the paper stack's override names. Rather than copying it, its `subprocess` is
replaced by a shim that, per trial, (1) translates the overrides to the dev config
    processes.iceflow.physics.init_slidingco=X  -> ...physics.sliding.slidingco=X  (+ tau_ref = X*u_ref**(1/3))
    processes.iceflow.physics.init_arrhenius=A  -> ...physics.viscosity.arrhenius=A
    processes.data_assimilation.*               -> assimilations.data_assimilation.*
(2) runs igm_run in devrun/ with igm_dev first on PYTHONPATH. Search space, sampler, objective (final DA
velsurf cost), summary and plots are the paper's.

    python tools/sweep_da_dev.py --sweep-dir results/EXP01_S02_cnn/sweep \
        --experiment params_cx_EXP01_S02_cnn_da --n-trials 30 --gpus 0 --workers-per-gpu 2
"""
import argparse
import importlib.util
import os
import subprocess as _sp
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import devstack  # noqa: E402

CX_DIR = devstack.CX_DIR
SWEEP_SCRIPT = "/home/klleshi/Documents/igm-smb-inference/optuna_sweep_da_nsga.py"
IGM_RUN = os.path.join(os.path.dirname(sys.executable), "igm_run")


def translate(ov):
    """Paper-stack Hydra override -> list of dev-stack overrides."""
    if ov.startswith("processes.iceflow.physics.init_slidingco="):
        v = float(ov.split("=", 1)[1])
        return [f"processes.iceflow.physics.sliding.slidingco={v}",
                f"processes.iceflow.physics.sliding.tau_ref={v * 100.0 ** (1.0 / 3.0)}"]
    if ov.startswith("processes.iceflow.physics.init_arrhenius="):
        return ["processes.iceflow.physics.viscosity.arrhenius=" + ov.split("=", 1)[1]]
    if ov.startswith("processes.data_assimilation."):
        return ["assimilations.data_assimilation." + ov[len("processes.data_assimilation."):]]
    return [ov]


class _DevSubprocess:
    """Stands in for the sweep script's `subprocess` module: same API, dev-stack igm_run."""

    def __getattr__(self, name):
        return getattr(_sp, name)

    @staticmethod
    def run(cmd, **kw):
        gpu = (kw.get("env") or {}).get("CUDA_VISIBLE_DEVICES", "0")
        cmd = [IGM_RUN] + [o for ov in cmd[1:] for o in translate(ov)]
        kw["cwd"] = devstack.DEVRUN
        kw["env"] = devstack.env(gpu)
        return _sp.run(cmd, **kw)


ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--sweep-dir", required=True)
known, rest = ap.parse_known_args()
sweep_dir = os.path.abspath(known.sweep_dir)

devstack.ensure_devrun()
spec = importlib.util.spec_from_file_location("optuna_sweep_da_nsga", SWEEP_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.subprocess = _DevSubprocess()
mod.PROJECT_DIR = devstack.DEVRUN


def _paths(experiment):
    return {"sweep_dir": sweep_dir,
            "db_path": os.path.join(sweep_dir, "optuna_study.db"),
            "plots_dir": os.path.join(sweep_dir, "plots"),
            "study_name": f"da_tpe_{experiment}"}


mod._experiment_paths = _paths
sys.argv = [SWEEP_SCRIPT] + rest
mod.main()
