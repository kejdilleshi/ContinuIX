# ContinuIX — GROUP_UNIL

Submission of the transient, differentiable SMB inversion (IGM, `igm-smb-inference`) to the ContinuIX
intercomparison: SMB and flux divergence from dh/dt, surface velocity and (optionally) GPR thickness, for
EXP01–20 on the mandatory glaciers (G01–G06, S01, S02).

**Results and method: [report/report_final.pdf](report/report_final.pdf)** (LaTeX source: [report/report_final.tex](report/report_final.tex)).

## Layout

| path | content |
|---|---|
| `tools/` | pipeline scripts (`run_continuix.py`, `batch.py`, `continuix_to_igm.py`, `igm_to_continuix.py`, …) and `queue.txt` (job list) |
| `igm_dev/` | IGM `dev` export with the bundled CNN emulator (one patch: `state.slidingco`) |
| `smbrun/`, `devrun/` | run configs for the SMB phase / dev stack |
| `report/` | report sources, figures, tables |

Not in git (too large): `input_data/`, `data/`, `results/`, `archive/`, `logs/`, `GROUP_UNIL/` (the submission files).
The `user` symlink points to `../igm-smb-inference/user`.

## Run

```bash
export PATH=/home/klleshi/miniconda3/envs/igm-pretrain/bin:$PATH

# one job, all stages (resumable)
python tools/run_continuix.py --exp EXP01 --gid G03 --variant cnn --gpus 0 --resume

# background lanes, one per GPU; progress and retry
setsid nohup python tools/batch.py --gpu 0 > logs/lane0.log 2>&1 < /dev/null &
python tools/batch.py --status
python tools/batch.py --retry-failed
```

Per-job logs: `logs/<EXP>_<GID>_<variant>.log`; runs: `results/<EXP>_<GID>_<variant>/`.

## Rebuild the report

```bash
python report/make_assets_final.py && tectonic report/report_final.tex
```
