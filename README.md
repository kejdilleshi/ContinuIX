# ContinuIX — GROUP_UNIL plan

Plan for submitting the transient, differentiable SMB inversion (Lleshi et al., IGM) to the
ContinuIX intercomparison, a review of what has been built so far, and how to get the best
results out of the method. Status as of 2026-10-12 (23:20): Phase 0 and gates G1-G3 done; EXP01 running (§4, §6a).

**2026-10-13 11:15 — restart.** All experiments are rerun from scratch with the current method, variant `cnn`
(§3a): the CNN emulator bundled with IGM instead of the Aletsch SIADecompNet, no provided THK/BED field anywhere,
GPR profiles in the DA where available. Synthetic glaciers first (`tools/queue.txt`, 97 jobs). Everything produced
before (results, logs, configs, the EXP01 method01 files) is in `archive/siadecompnet_2026-10-13/`; §3–§6b below
describe that earlier procedure and its evidence.

## 1. Goal and scope

ContinuIX (IACS working group) compares methods that infer SMB from mass conservation
(dh/dt, velocity, thickness). Participants receive the same anonymised inputs; the organisers
score the returned **SMB** and **FDIV** (m i.e. yr⁻¹, on the input grid) with bias, R² and RMSE
against hidden stake SMB, stake flux divergence and the glacier-wide geodetic balance, and they
assess the **uncertainties** (`UNCT_SMB`, `UNCT_FDIV`).

What we submit: one NetCDF per experiment × glacier × method (`EXP##_G##_method##.nc`), a
README per method and a computation log, under `GROUP_UNIL/`.

Binding decisions:

- Deadline extension granted.
- Everything (data, configs, runs, outputs) lives in this folder; the method code is reused
  read-only from `../igm-smb-inference` through the `user` symlink.
- Perturbation experiments run the **full procedure** (Optuna sweep, DA L-curve, τref family)
  per experiment and glacier, with no reuse of parameters — **mandatory glaciers only**.
- Compute: node28 only (2× RTX 4090); runs must survive closing the session.
- Deliverable per glacier: the **family of best fits (paper Fig. 7)**, not a single profile.

## 2. Data

| ID | likely glacier | native dx | window | DEM epoch | model dx | ice cells | jobs |
|---|---|---|---|---|---|---|---|
| G01 | Hofsjökull ice cap (11 basins) | 25 m | 2013–2023 | 2018 | 100 m | 83 261 | 19 |
| G02 | Zongo | 25 m | 2006–2013 | 2006–2013 (2009.5 assumed) | 50 m | 780 | 6 |
| G03 | Argentière | 20 m | 2012–2021 | 2017 | 100 m | 1 229 | 6 |
| G04 | Sarytor | 2 m | 2021–2025 | 2021–2025 (2023 assumed) | 50 m | 887 | 6 |
| G05 | Great Aletsch | 10 m | 2017.67–2023.64 | 2017.67 | 100 m | 7 904 | 19 |
| G06 | Gepatschferner | 10 m | 2006–2017 | 2017 (THK/BED 2006) | 100 m | 1 672 | 6 |
| S01 | Icepack half-pipe, steady state | 1 m | none (10 yr assumed) | – | 50 m | 822 | 14 |
| S02 | Elmer/Ice "idealised Aletsch", 10-yr perturbation | 100 m | none (10 yr assumed) | – | 100 m | 8 174 | 14 |

Identities are inferred from CRS and epochs (the exercise is anonymised). "jobs" = entries in
`tools/queue.txt` (90 in total): EXP01 for all 8; EXP03–15 for the mandatory G01, G05, S01, S02;
EXP16–20 for the 6 real glaciers; EXP02 (raw, 7 glaciers) was added later (§6b), 97 jobs in total.

## 3a. Method as applied since 2026-10-13 (variant `cnn`, method01)

Same five stages as §3, on the "dev stack" (`tools/devstack.py`): IGM `dev` at 0d9df551 (`igm_dev/`, exported, one
patch: `state.slidingco` for the pinnbp emulators) + the `igm-smb-inference-user` module (clamp fix applied) + IGM's
bundled `pinnbp_10_4_cnn_16_32_2_1_a` (Nz = 10, Lagrange/central). Every `igm_run` executes in `devrun/`.

- **convert** — `continuix_to_igm.py`, then `tools/gpr_input.py` → `data/<EXP>_<GID>_da.nc`: DA first guess =
  perfect-plasticity thickness from the surface slope (no provided THK/BED). **GPR**: the EXP02 raw thickness where it
  is profile data (≤ 25 % of the glacier: G02, G03, G05, S02 — not G01/G04/G06, gridded products, nor S01), valued
  with the experiment's own THK at those pixels moved to t_DA (EXP03–07 perturbations propagate; none for EXP17–20,
  model THK). Split by profile segment (8-connected), seed 0: train/val/test 60/20/20 (≥ 10 segments) or 70/30/0.
- **sweep** — as before (τref, A; objective = final DA velocity misfit); with GPR the DA cost also fits the train profiles.
- **lcurve** — without GPR: the reg.thk L-curve. With GPR: **gprtune**, a DA grid reg.thk {100, 300, 1000} ×
  thkobs_std {10, 20, 40, 80} m scored on the **validation** profiles; anti-overfitting pick = runs within 1.5× the
  best velocity misfit, validation RMSE within 5 % of the best, then the largest thkobs_std and reg.thk.
  Test profiles are only reported (`gpr_eval.py family`, `results/<job>/gpr_scores.json`).
- **family / export** — as before; files go to `GROUP_UNIL/<EXP>/` as method01.

## 3. Method as applied (until 2026-10-13, archived)

Per experiment × glacier (`tools/run_continuix.py`, stages in order, all resumable):

1. **convert** — ContinuIX NetCDF → IGM input. Block-average to 100 m (50 m below 5 km², the
   emulator's training range). From the single DEM and DHDT: start surface
   `DEM + DHDT (t1 − t_DEM)`, target `DEM + DHDT (t2 − t_DEM)`, DA surface at
   `t_DA = clip(t_velocity, t1, t2)` (the paper's "epoch closest to the velocities").
   Provided THK, moved to `t_DA` with DHDT, is the DA initial guess.
2. **sweep** — Optuna TPE, 30 trials over τref ∈ [0.05, 2] MPa and A ∈ [30, 120], velocity-only
   DA (σ_u = 2.1 m yr⁻¹), objective = surface-velocity misfit.
3. **lcurve** — DA reg.thk ∈ {50, 100, 300, 600, 1000, 2000} at the best (τref, A); corner picked
   automatically.
4. **family** — combined DA + SMB inversion for up to 5 τref values with sweep misfit ≤ 1.5× the
   best (thinned ×1.25 in log space), at the best A and the L-curve reg.thk. SMB(z) on 200 m
   nodes, curvature regularisation 0.05, L-BFGS, CFL 0.2.
5. **export** — admissible members (final DA misfit ≤ 1.5× the family best) → native grid.

**Deliverable = the Fig. 7 family.** Each file carries the admissible-family mean as `SMB`/`FDIV`
and the family spread in `UNCT_*`; every member's profile, τref, A, velocity misfit, ELA and
admissible flag are stored too, and a Fig. 7-style plot (members, admissible solid, rejected
dashed, best bold, ELA band) is produced per glacier/experiment (to be added, §6 step 0.3).

Output fields: `SMB` = profile at the mid-window surface; `FDIV` = SMB − modelled dH/dt (the
window-averaged transient flux divergence, paper Fig. 10b); `DHDT` = SMB − FDIV (modelled,
reported as modified); `THK` (DA, at t_DA); `DENSITY` = 917 kg m⁻³.

## 4. Status — what is done

- `tools/`: `continuix_to_igm.py`, `assemble_glob.py` (EXP16–20), `run_continuix.py`,
  `sweep_da.py`, `igm_to_continuix.py`, `batch.py` (background lanes), `diag_run.py`,
  `queue.txt` (97 jobs). All inputs convert, incl. EXP02 (assembled from the raw files, §6b).
- G03 smoke test (paper parameters τref = 0.199 / 0.127, A = 93.07, reg.thk = 100, no sweep):
  DA mean thickness 89 m (paper 90 m); first export written
  (`results/EXP01_G03_vel/export_smoke/`) and checked against the input grid/CRS.
- Phase 0 done (converter gap fills, exporter family + uncertainty + Fig. 7 plot, dev results moved to
  `results/_dev/`, method README). Gates G1-G3 done (§6a). `--resume`, `--only`, `--results-root`,
  `--family-parallel`, and `MPLBACKEND=Agg`/no `DISPLAY` added to the tools.
- **EXP01 done 2026-10-13 06:47** (8 jobs, 16 lane-hours). **Full queue released 2026-10-13 08:48** on both lanes (no `--only`),
  on your instruction to run all mandatory experiments (EXP01–20 on the mandatory glaciers of the organisers' table).
  GPU1 is shared with other users: `batch.py` waits for / sizes jobs to the free GPU memory (§9).

## 5. Review — issues found

Evidence from G03 runs (`tools/diag_run.py`) unless stated. Stakes = the paper's GLACIOCLIM
averages, used for checking only.

**In the method code / paper data**

| # | sev. | issue | evidence | status |
|---|---|---|---|---|
| M1 | critical | `pretrained_cnn_step.py` clamped H ≥ 0 *before* adding SMB (paper Eq. 6 and IGM `thk.py:45` clamp after). With the −10 m yr⁻¹ off-mask SMB (IGM convention) off-ice cells went negative every step and leaked mass. | 2 946 cells with H < 0; −0.47 m yr⁻¹ lost off-mask per glacier area; glacier-wide SMB −0.745 vs geodetic −0.857 m i.e. | **Fixed** (uncommitted). After: 0 cells, +0.02 m yr⁻¹ off-mask, glacier-wide SMB −0.874. All paper runs predate the fix. |
| M2 | high | The assimilated state exports ice through the outline; in the forward run that ice meets the −10 m yr⁻¹ off-mask SMB and is lost. The model over-thins the upper basins and FDIV is biased. | DA mean divflux on mask +0.31 m yr⁻¹ (edge cells 32 m thick); modelled vs observed dh/dt at 3000–3200 m −1.76 vs −0.26; mean FDIV +0.38; stake bias +0.49 m w.e. yr⁻¹ | Tested in G3: no variant fixes it without a cost (§6a); method01 stays the paper's DA |
| M3 | high | Stake `Alt` is 50–67 m below the DEMs (paper's and ContinuIX's), likely ellipsoidal vs orthometric heights. Evaluating the profile at stake `Alt` is optimistic. | τref 0.199: RMSE 0.41 at `Alt` vs 0.68 m w.e. yr⁻¹ sampling the field at stake XY | Open — check paper Table 2 / Fig. 6a |
| M4 | low | Paper text gives A ∈ [30, 200]; the sweep script uses [30, 120] (deliberate, per its comment). | `optuna_sweep_da_nsga.py` | Align the paper text |

**In the ContinuIX tooling**

| # | sev. | issue | evidence | status |
|---|---|---|---|---|
| T1 | high | THK perturbations (EXP03–07) were ignored: thickness came from DEM − BED but those experiments perturb THK only. | EXP06/07 differ from EXP01 in THK only (BED identical) | **Fixed**: thickness = THK moved to t_DA (G05 init 146 → 190 / 102 m for +/−30 %) |
| T2 | high | Compute: G01, G05, S02 were estimated to dominate (weeks of GPU time). | G03: ~5 min per DA, ~21 s per SMB evaluation (87 evaluations ≈ 30 min). G01 grid is 29× larger with ~2.3× more time steps. | Re-measured in G1: runs are CPU-bound, G05/G01 are cheap (§6a, Compute); concurrency levers in `batch.py` |
| T3 | medium | Zero-filled DHDT inside outlines counted as real zeros; THK = 0 inside outlines would become thickness observations in a thickness-aware variant. | DHDT = 0 on 5.2 % (G02), 3.3 % (G03) of glacier pixels; THK = 0 on 10.7 % (G02) | **Fixed** (0.2) |
| T4 | medium | Export keeps only the family mean and spread; `UNCT_*` = family spread only (NaN for a single member), likely too small. | `igm_to_continuix.py` | **Fixed** (0.3): all members stored, UNCT = family ⊕ band representation ⊕ σ_DHDT |
| T5 | medium | Smoke/test runs sit in `results/EXP01_G03_vel/family` and would be picked up by the export. | export globs `family/slid_*` | **Fixed** (0.4) |
| T6 | low | `GROUP_UNIL/README_GROUP_UNIL_method01.txt` says the thickness guess is DEM − BED. | now THK | **Fixed** (0.5) |
| T7 | info | Initial SMB guess −10 instead of −5 changes nothing. | misfit 14.69 vs 14.68 | Closed |

Checked and fine: velocity orientation after the y-flip (83–100 % of cells flow downslope;
G06 71 %); output grid, CRS and no-data convention identical to the input; GLOB assembly
introduces no new gaps.

## 6. Plan

**Phase 0 — fixes before launch** (code only, ~½ day)

- 0.1 Clamp order (M1) — done.
- 0.2 Converter: on-ice DHDT exactly 0 → fill with the median of its 50 m elevation band;
  on-ice THK exactly 0 → missing for thickness observations (T3).
- 0.3 Export (T4): store all members (profile, τref, A, velocity misfit, ELA, admissible flag);
  write the Fig. 7 plot; `UNCT_SMB` = √(family spread² + representation error² + σ_DHDT²), where
  the representation error is the RMS of the end-of-window geometry residual (H_sim − H_obs)/Δt
  within the pixel's elevation band (what a single SMB(z) cannot explain) and σ_DHDT is the
  provided DHDT uncertainty. Same construction for `UNCT_FDIV`.
- 0.4 Move `results/EXP01_G03_vel` to `results/_dev/` (T5).
- 0.5 Update the method README (T6).

**Phase 1 — gates** (GPU, ~1 day; decisions come from these, not from stakes)

- **G1 timing.** One DA + one SMB inversion each for G05 and G01 (G01 at 100 m and 200 m).
  Decide the G01 resolution and the run settings below, then fix the schedule.
- **G2 pipeline truth test.** S01 is in steady state and its file attributes give the SMB law
  (ELA 2350 m, 0.01 yr⁻¹, capped at 5 m yr⁻¹). The full procedure must recover it; if not, fix
  before anything else. Check only — not used to tune.
- **G3 DA variant (M2).** On G03 with the paper parameters, compare
  (a) velocity-only (done), (b) + IGM's `divfluxfcz` cost with `divflux.force_zero_sum: true`,
  (c) thickness-aware DA (provided THK as observation, σ = 20 m).
  Select on stake-free criteria: mean FDIV ≈ 0, glacier-wide SMB vs geodetic, geometry and
  velocity misfit. G03 stakes (sampled at XY) are reported, not used to choose.
  Outcome: method01 = the paper method (a); method02 = the best of (b)/(c) if it clearly
  improves mass closure.

**Phase 2 — EXP01** (8 glaciers, method01 [+ method02])

- Run through `tools/batch.py`; inspect each Fig. 7 family, mass closure (`diag_run.py`) and
  the exported file. Freeze the settings — the same for every experiment afterwards.

**Phase 3 — EXP03–15 and EXP16–20** (82 jobs)

- Same lanes and queue order (EXP order, small glaciers first within each EXP).
- Monitor with `python tools/batch.py --status`; failed jobs are retried with `--retry-failed`.

**Phase 4 — package** — log from `results/batch_state.json` (hours per job, GPU), method
README(s), submission checklist, file validation (grid, names, units, NaN outside ice).

**Compute** (measured in G1/G3, node28; details in §6a)

The runs are CPU/latency-bound: GPU utilisation 0-5 %, ~1 core per TF process, 12 cores. Cost is almost
independent of grid size, so G05/S02/G01 are not the monsters the first estimate assumed. Several processes
share one GPU until the cores (~10 processes) or the 24 GB run out; `batch.py` sets the concurrency per job
(small grids: 3 Optuna workers, 3 L-curve jobs, 5 family members; 8k cells: 3/3/2; G01: 2/2/1).

| glacier | per job, 2 lanes running (estimate) | jobs |
|---|---|---|
| G02, G03, G04, G06, S01 | ~1.5-2 h | 38 |
| G05, S02 | ~2-3 h | 33 |
| G01 at 100 m | ~3-4 h | 19 |

Whole queue (90 jobs) ≈ 170-250 lane-hours → **~4-5 days on two lanes**; EXP01 ≈ 8-10 h. The first jobs will replace
these numbers (`results/batch_state.json`, `python tools/batch.py --status`).
G01 stays at 100 m (200 m is only ~1.3x cheaper and fits worse). Further levers if needed: stop L-BFGS on
relative tolerance (already on), Optuna trials 30 → 20, optional method02 only on EXP01.

## 6a. Gate results (2026-10-12)

**G1 timing** (solo, GPU idle apart from the test; `tools/gate_timing.py results/_dev/G1/*.log`)

| run | DA 1000 it | SMB / evaluation | total DA + SMB | closure (FDIV mean, SMB − dh/dt obs) |
|---|---|---|---|---|
| G03 (4.4k cells) | 2.8 min | 21 s | ~35 min (94 evals) | +0.41, +0.04 |
| G05 (22.9k cells) | 2.8 min | 21 s | 20 min (46 evals) | +0.006, +0.001 |
| G01 at 100 m (129k cells) | 4.8 min | 26 s | 16 min (22 evals) | −0.14, −0.09 |
| G01 at 200 m (34k cells) | 3.2 min | 15 s | 13 min (35 evals) | −0.09, −0.07 |

G01@100 needs ~17 GB for the SMB reverse pass (one per GPU), G05 ~9 GB, small grids 2-3 GB. G01@100 vs 200:
geometry misfit 16.6 vs 19.4 m, same closure → 100 m.

**G2 truth test, S01** (`tools/check_s01.py`, `results/_dev/G2/`; law ELA 2350 m, 0.01 yr⁻¹, cap 5 m yr⁻¹):
- Mechanics fine: pixel SMB bias +0.03, R² 0.84, glacier-wide SMB −0.111 vs law −0.138, ELA of the family
  mean ≈ 2335 m, 73 % of pixels inside ±UNCT_SMB (nominal 68 %), no grid/unit problem found.
- But the velocity-selected family is biased by thickness–sliding equifinality: the low-τref members are 22-35 m too
  thin (DA mean 18-31 m vs the provided 53 m) and miss the law at both ends (profile RMSE 1.5-2.0 m i.e. yr⁻¹; ELA
  2181-2399 m); the member whose thickness matches THK (τref 0.226, bias +0.1 m) is the best (RMSE 0.76) yet rejected
  by the velocity criterion (misfit 39 vs 22). Sweep A = 30.25 sits at the lower bound.
- Reading: no pipeline fault; the paper method with velocity-only selection recovers the law only loosely on a steep
  synthetic glacier. The provided THK would fix it → supports method02 (below).

**G3 DA variants on G03** (paper parameters, same input; `results/_dev/G3/diag_table.txt`, stake-free)

| | a vel-only | b divfluxfcz + zero-sum | b2 (b, σ = 5) | c thickness-aware |
|---|---|---|---|---|
| DA velocity misfit | 1.27 | **8.09** | 1.73 | 1.61 |
| SMB-inversion misfit | 14.44 | 12.29 | 14.24 | 15.80 |
| geometry RMS residual (m) | 18.9 | 16.3 | 18.7 | 20.6 |
| mean FDIV on mask | +0.41 | **+0.005** | +0.38 | +0.58 |
| glacier-wide SMB − dh/dt obs | +0.04 | −0.14 | +0.11 | +0.15 |
| DA mean thickness (provided 142 m) | 89 | 82 | 87 | 103 |

Only (b) closes mass, at 6× the velocity misfit (outside the 1.5× admissibility) and a worse glacier-wide balance;
(b2) and (c) do not improve closure. Decision by the rule in §6: **method01 = (a), the paper method; no method02 from
(b)/(c)**. Stakes at XY (check only, m w.e. yr⁻¹, RMSE / bias): a 0.70/+0.53, b 0.62/+0.26, b2 0.66/+0.45, c 0.72/+0.56.
Caveat for (c): τref, A were the paper's, not re-tuned with the thickness cost.

**Proposed method02 (not started, your call):** thickness-aware selection — the Optuna sweep and family admissibility
use velocity + thickness misfit against the provided THK. The S01 evidence above says it removes the equifinality there;
for G03 it is untested.

## 6b. EXP02 (raw data)

The organisers' table lists EXP02 with 7 mandatory glaciers (G01–G06, S02); the text calls it optional. Run as mandatory.
Inputs: `tools/raw_to_continuix.py` assembles each real glacier's raw GeoTIFFs, GPR points and outline shapefiles
into `input_data/EXP02/EXP02_<GID>.nc` on the **EXP01 grid** (the protocol: results on "the same grid as the input
data"; the raw rasters come on 2–200 m grids). S02's raw file is already a NetCDF on the EXP01 grid (linked). From
there the jobs are the same as EXP01 (same procedure, same settings; method01).

What is different from EXP01 (checked with `tools/cmp_raw_vs_ref.py`; on ice the raw fields reproduce EXP01 to within
< 2 m / 0.03 m yr⁻¹ wherever both exist, EXP01 being the same data with gaps filled):
- gaps stay NaN: velocities G04 34 %, G06 68 %, S02 33 % of the glacier cells (EXP01: 98–100 %); DHDT/DEM gaps are small;
- thickness is GPR only: points averaged per cell for G02, G03, G05 (7 %, 2 %, 1.5 % of the cells), rasters for G01, G04,
  G06; elsewhere the DA initial guess falls back to DEM − BED (BED rasters are complete). Method01 uses thickness only
  as the initial guess, so this changes little; S02 has neither BED nor THK except 62 cells → plastic-rheology
  thickness H = 100 kPa / (0.8 ρ g sin α) from the smoothed surface slope as the initial guess (`continuix_to_igm.plastic_thickness`);
- several epochs: DEMs shifted to their mean epoch with the observed DHDT and averaged (G01 ×3, G02, G04 ×5, G05);
  several DHDT windows combined (G01), a provided full-window DHDT used as is (G04 2021–2025); velocity periods averaged;
- on-ice DEM gaps filled by inverse-distance interpolation (≤ 500 m; G06: 164 cells); G01's DEM is ellipsoidal (ICEGEOID
  note in EXP01) and sits 66.9 m above BED/THK → shifted by the DEM − BED − THK median (agrees with EXP01 to 1.8 m rms);
- ICEMASK rasterised from the outlines (last outline year): identical cell counts to EXP01 (G05 +0.5 %).
All of this is in `README_GROUP_UNIL_method01.txt`. Queued: `EXP02 G03` first (pipeline smoke test), the others last.

## 7. Making the best of the model (ranked)

1. Mass-conserving forward model (M1) — done.
2. Mass-consistent DA state (M2) — gate G3; largest expected gain for FDIV and stake bias.
3. Fig. 7 family as the product, ensemble mean as the estimate, honest uncertainty (step 0.3).
   The paper's Fig. 11 shows that the single velocity-selected member can be the worst one,
   so the mean of the admissible family is the safer estimate.
4. Thickness-aware DA as method02 (gate G3): ContinuIX provides GPR-based thickness, which
   removes the thickness–sliding equifinality the velocity-only method has to live with.
5. Optional, data-only extensions (after Phase 3, if time allows):
   per-basin profiles for the ice cap G01 (BASINMASK; Hofsjökull has a strong N–S
   precipitation gradient); a linear horizontal SMB term for S02 (east–west gradient);
   EXP02 is now in the queue (§6b). A raw-data variant using two real DEMs (the method's natural input) would need
   bespoke preprocessing per glacier, GPR points, gaps).

Not allowed: tuning anything on stakes or on information outside the provided data.

## 8. Submission checklist

- `GROUP_UNIL/EXP##/EXP##_<GID>_method##.nc` for every queued job (S01/S02 keep their IDs).
- Variables: `SMB`, `FDIV`, `DHDT`, `THK`, `DENSITY`, `UNCT_SMB`, `UNCT_FDIV`, `UNCT_THK`,
  family variables; same x/y/CRS as the input; NaN outside the glacier.
- `README_GROUP_UNIL_method01.txt` (+ `method02` if submitted).
- `log_GROUP_UNIL.txt`: hours per job and GPU (RTX 4090), from `results/batch_state.json`.
- `SUBMISSION_CHECKLIST.txt` filled.

## 9. How to run

```bash
cd ~/Documents/ContinuIX
export PATH=/home/klleshi/miniconda3/envs/igm-pretrain/bin:$PATH

# one job, all stages (resumable)
python tools/run_continuix.py --exp EXP01 --gid G03 --variant cnn --gpus 0 --resume

# background lanes (survive closing the session), one per GPU
setsid nohup python tools/batch.py --gpu 0 [--only EXP01] > logs/lane0.log 2>&1 < /dev/null &
setsid nohup python tools/batch.py --gpu 1 [--only EXP01] > logs/lane1.log 2>&1 < /dev/null &
python tools/batch.py --status          # progress (--only EXP01 restricts the lane to EXP01)
python tools/batch.py --retry-failed    # re-queue failed jobs

# diagnostics of a run (mass closure, misfit; --stakes only for checking)
python tools/diag_run.py results/EXP01_G03_vel/family/slid_0.1991 --meta data/EXP01_G03.json

# GLOB inputs (EXP16-20) are assembled once with
python tools/assemble_glob.py EXP16 EXP17 EXP18 EXP19 EXP20
```

Per-job logs: `logs/<EXP>_<GID>_<variant>.log`; runs: `results/<EXP>_<GID>_<variant>/`.

The cluster is shared (another user's job took 14.5 GB of GPU1 on 2026-10-13). `batch.py` therefore (i) picks only a job
that fits the free GPU memory, waiting otherwise, (ii) cuts the Optuna workers / L-curve jobs to what fits, and
(iii) `run_continuix.py` sizes each wave of family members from the free memory. Restarting a lane is safe: a job left
`running` by a dead lane is resumed (stop the lane `batch.py` first, then its job processes; never edit a `.sh` that
is running).

EXP02 inputs: `python tools/raw_to_continuix.py G01 G02 G03 G04 G05 G06 S02` (done), check with `tools/cmp_raw_vs_ref.py`.

## 10. Open questions

For the organisers:
- Stake elevations: which vertical datum? (Our G03 check found a ~55 m offset to the DEM.)
- G02: mean DHDT is +1.55 m yr⁻¹ over 2006–2013 — is the sign/units correct?
- Synthetic cases: inversion window and DEM epoch of S01/S02?
- DEM epochs given as ranges (G02 "2006-2013", G04 "2021-2025"): which epoch?
- Is EXP02 mandatory (table: 7 glaciers; text: optional)? Is there a WP3 priority list? (run as mandatory)
- EXP02 output grid: the EXP01 grid was chosen (raw rasters have 2–200 m grids); is that acceptable?
- File names for synthetic glaciers: `S01` or `G##`?

For you:
- Method02 (thickness-aware sweep/selection) — build and test on S01/G03, then submit? (§6a)
- Release the rest of `queue.txt` after the EXP01 check: `batch.py --gpu N` without `--only` (lanes started with
  `--only EXP01` exit when EXP01 is done).
- σ_DHDT is provided only for S01/S02 (field) and G03 (±0.07 m/yr); I read G02 "±2 m" and G04 "≤ 0.5 m" as
  elevation-change errors divided by the window; G01/G05/G06 "unknown" → term omitted (see `sigma_DHDT_source`).
- Commit the clamp fix, and re-run the paper's results with it?
- Method02 after gate G3 — submit it for all experiments or EXP01 only?
- G01 resolution and the speed levers after gate G1.
- The optional extensions: yes/no.
