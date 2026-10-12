# LegacyPM runbook

How to run everything, reconstructed 2026-10-12. If something here turns out to be wrong, fix it here.

## TL;DR

```bash
# 1. make a catalogue for whole RA slices (one Slurm job; survives logouts; no FireWorks needed)
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
mkdir -p /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 060 061 062 063 --dry-run   # count
sbatch run_bricks.sbatch 060 061 062 063                    # new bricks from Dustin
sbatch run_bricks.sbatch 060 061 062 063 --from-archive     # rerun the 548 archived bricks
squeue --me                                                  # monitor
# 2. copy finished output to CFS (pscratch is purged!)
# 3. analyse in NERSC Jupyter with the desi-pm kernel (run/*.ipynb)
```

## 1. Environments

| What | Where | Used for |
|---|---|---|
| Shifter image `docker:nadafalou/legacypm:5` (May 2025) | Perlmutter (`shifterimg images \| grep legacypm`) | Running the pipeline (`py_files/*.py`). Provides legacypipe, astrometry.net, fitsio, astropy, fireworks. |
| conda env `desi-pm` (`~/anaconda3/envs/desi-pm`) | login node / Jupyter kernel "desi-pm" | Notebooks in `run/`, analysis, quick scripts. Has legacypipe on `PYTHONPATH` via `~/.bashrc`. |
| conda env `fireworks` | login node | `lpad`, `qlaunch` (FireWorks route only). |
| `pmwork` (shell function in `~/.bashrc`) | login node | Attaches to / creates a tmux session `legacypm` in `fireworks/` with the `fireworks` env and Mongo credentials (`~/.legacypm_env.sh`) loaded. |

**The image only provides the software.** The pipeline code always comes from this repo clone
(`/pscratch/sd/n/nelfalou/LegacyPM/py_files/`): both the Slurm driver and the FireWorks workflow call the scripts by
absolute path. (The Dockerfile also `git clone`s the repo, but that copy is not used.) So you do **not** need to rebuild
the image after changing code -- only if you need a new package.

Note: there is a second, older clone at `~/LegacyPM` (home). The 2025 FireWorks runs were launched from there. Use the
pscratch clone; consider deleting the home one to avoid confusion.

## 2. Data

- **Input** (Dustin's forced photometry, DR10): `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/<RRR>/`
  `forced-<brick>.fits` + `tractor-forced-<brick>.fits`. RA 000-057 from an older run on CFS:
  `/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick/` (use `--input-root`).
  As of 2026-10-12, 060-063 are complete (4,037 bricks on pscratch + 548 in the archive = all 4,585 DR10 bricks).
- **Archive** (Sept 2025 copy of the Feb-Mar 2025 run): `/pscratch/sd/n/nelfalou/archive/legacypm-catalogue/06[0-3]/`
  (548 bricks: corrected `forced-*` + `tractor-pm-*`). Their raw inputs are gone, hence `--from-archive`.
- **Output** (default): `/pscratch/sd/n/nelfalou/legacypm-catalogue/<RRR>/` -- `forced-<brick>.fits` (raw + `dcr_`, `rm_`,
  `lm_` columns), `tractor-pm-<brick>.fits` (the catalogue), `logs/<brick>.log`, `failed.txt`.
- **pscratch is purged** (files untouched for ~8 weeks). Everything from 2025 except what you copied was lost. When a run
  finishes, copy it to CFS, e.g.
  `rsync -a /pscratch/sd/n/nelfalou/legacypm-catalogue/060/ /global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue/060/`
- Calibration files read by the code: `survey-ccds-decam-dr10-bprp-kd.fits` (DCR; Dustin's CFS), `ringmaps.fits`
  (CFS `ls-motions/rm-corrected-forced-motions/`), `py_files/lateral_model_poly3.json`.

## 3. The pipeline (per brick)

Each step reads the previous step's `forced-<brick>.fits` and writes it again with extra columns.

| Step | Command | Adds |
|---|---|---|
| 1 DCR | `python py_files/brick_DCR.py forced-<b>.fits <out_dir> <input_dir>` | `dcr_full_fit_*` |
| (1') archive prep | `python py_files/prep_archive_rerun.py forced-<b>.fits <out_dir> <archive_dir>` | copies archived DCR output, drops old `rm_`/`lm_`/PM columns |
| 2 ring maps | `python py_files/brick_ringmaps.py forced-<b>.fits <out_dir> <out_dir>` | `rm_full_fit_*` |
| 3 lateral colour | `python py_files/brick_lateralmaps.py forced-<b>.fits <out_dir> <out_dir> <tractor_dir>` | `lm_full_fit_*` |
| 4 PM catalogue | `python py_files/create_brick_catalogue.py forced-<b>.fits <out_dir> <out_dir> <tractor_dir>` | `tractor-pm-<b>.fits` |

Output columns added to the tractor catalogue: `pmra/pmdec_no_plx` (4-parameter fit), `pmra/pmdec_plx` and `parallax_new`
(5-parameter fit), `new_ra/new_dec_{no_plx,plx}` (position at `ref_mjd`), their `_ivar`s, `pm_flag`, `ref_mjd`. All are
absolute, also for Gaia stars (Gaia motion added back since 2026-10-12); the original `pmra`, `pmdec`, `parallax` columns
are Gaia's.

Directories need a trailing `/`. `tractor_dir` is the input dir for new bricks and `out_dir` for archive reruns.
One brick takes ~20-30 s on one core for an archive rerun and ~1-2 min for a new brick (DCR loads a large CCD table).

**Debug a single brick** on a login node:
```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 064 --out-root $SCRATCH/test --workers 1
```
(or run the four commands above by hand inside `shifter --image=docker:nadafalou/legacypm:5 bash`).

## 4. Running many bricks -- Slurm driver (recommended)

`slurm/run_bricks.py` finds the bricks in the given slices, skips any with a finished `tractor-pm-*`, and runs the four
steps per brick in a pool of worker processes. `slurm/run_bricks.sbatch` runs it in Shifter on one CPU node.

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
mkdir -p /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs        # once
sbatch run_bricks.sbatch 060 061 062 063                                # extra args are passed to run_bricks.py
squeue --me
tail -f /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs/legacypm-<jobid>.out
cat /pscratch/sd/n/nelfalou/legacypm-catalogue/060/failed.txt           # after it ends
less /pscratch/sd/n/nelfalou/legacypm-catalogue/060/logs/<brick>.log
```

- Logging out does not affect a submitted Slurm job, so no tmux needed.
- If it hits the time limit or some bricks fail: fix the cause and resubmit the same command; finished bricks are skipped.
  A failed brick may leave a partial `forced-<brick>.fits`; that's fine, it is rewritten from step 1.
- Sizing: ~4,000 new bricks x ~1.5 min / 64 workers is about 1.5 h. Walltime is 4 h. If memory runs out, lower `--workers`
  in the sbatch file. For a quick test use `--qos=debug --time=00:30:00` on the sbatch command line.
- Other options: `--input-root` (e.g. Dustin's CFS run for RA < 058), `--out-root`, `--dry-run`.

**Why this instead of FireWorks?** The four steps are just four Python commands per brick, and the driver handles
"skip what's done / log what failed". It needs no database, and it submits **one** job instead of four jobs per brick
(~18,000 jobs for 060-063). The part of FireWorks that needed tmux (`qlaunch ... --nlaunches infinite`) runs on a
login node; a plain `sbatch` job doesn't.

## 5. Running many bricks -- FireWorks (the 2025 route; currently blocked)

**Blocked (2026-10-12): MongoDB authentication fails** with the credentials in `~/.legacypm_env.sh` /
`fireworks/my_launchpad.yaml` (host `mongodb05.nersc.gov`, db `legacypm`). The account or password must have changed or
expired; ask NERSC (help.nersc.gov ticket) to reset it, then update both files.

Once it works:
```bash
pmwork                                        # tmux session in fireworks/, env + creds loaded
python legacypm_workflow.py 060 061 --dry-run
python legacypm_workflow.py 060 061           # adds one workflow per slice (4 fireworks per brick)
python legacypm_workflow.py 060 --from-archive
qlaunch -r rapidfire -m 50 --nlaunches infinite   # submits one Slurm job per firework; keep running in tmux
lpad get_wflows -d less                      # progress
lpad get_fws -s FIZZLED -d ids               # failures
lpad rerun_fws -s FIZZLED                    # retry after fixing
```
- `my_qadapter.yaml`: one 1-core job per firework, 45 min, shifter image. Fixed 2026-10-12: `rocket_launch` used
  `rlaunch singlerun`, which doesn't exist (needs `singleshot`, required by `qlaunch -r`) and didn't say where the
  launchpad file is (`-l .../my_launchpad.yaml`). **Untested since the fix** because of the Mongo problem.
- `qlaunch` creates `launcher_*` dirs in `fireworks/` (61k of them from 2025 are still there; safe to delete).
- `lpad reset` wipes the whole database -- don't.

## 6. Analysis and figures

- NERSC Jupyter (jupyter.nersc.gov), kernel **desi-pm**. Notebooks in `run/` import the code with
  `sys.path.append(...); from py_files import ...`, so open them from `run/`.
- Loading hundreds of bricks: use a Jupyter server on a compute node (or `salloc -q interactive -C cpu -t 2:00:00`),
  not a login node.
- Paper figures only from `run/` notebooks, saved to `writeup/plots/` (see CLAUDE.md). Main one: `err-plots.ipynb`.
- Notebooks still point at old dirs (`legacypm-catalogue/190-192`, etc.); repoint to the new output dirs.

## 7. Calibration models

| Model | How to rebuild | Status |
|---|---|---|
| DCR amplitudes `K_b` (`DCR_AMPLITUDES` in `brick_DCR.py`) | `run/dcr-plots.ipynb` | K_i = K_z = 1: negligible by design (Nada, 2026-10-12) |
| Ring maps `ringmaps.fits` | `get_gaia_data` + `make_ringmaps` in `brick_ringmaps.py` (input star files purged) | Deferred; see PLAN.md |
| Lateral colour `lateral_model_poly3.json` | `run/lateral-model.ipynb`, cell marked [APPROVED] | Current (2026-10-10) |

After changing a model, check it with the end-to-end test: `run/e2e/` (`run_e2e.py`, then `eval_e2e.py`, `faint.py`;
outputs in `$LEGACYPM_E2E_DIR`, default `/pscratch/sd/n/nelfalou/e2e-test`). It reruns 8 held-out archive bricks and
compares old and new per-epoch scatter and Gaia-star PMs. The 2026-10-10 results are in PLAN.md.

## 8. Paper

```bash
module load texlive/2024
cp -r writeup $SCRATCH/wbuild && cd $SCRATCH/wbuild
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
`writeup/` was copied from Overleaf on 2026-10-11; keep the two in sync.
