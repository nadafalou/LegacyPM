# LegacyPM runbook

How to run everything, reconstructed 2026-10-12. Current goal: RA 000-063 processed with the current code. If something here turns out to be wrong, fix it here.

## TL;DR -- goal: RA 000-063 processed and ready for analysis

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
mkdir -p /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs                      # once
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 000-063 --dry-run   # what's done / to do, per slice
sbatch --qos=debug --time=00:30:00 run_bricks.sbatch 058                            # small test job (done 2026-10-12)
sbatch run_bricks.sbatch 000-015      # then 4 jobs, one per block of 16 slices (never overlap two running jobs)
sbatch run_bricks.sbatch 016-031
sbatch run_bricks.sbatch 032-047
sbatch run_bricks.sbatch 048-063
squeue --me                           # monitor; logging out is fine
# when Dustin delivers more bricks (missing_bricks.txt): resubmit the same ranges -- only new bricks run
# then: combine per slice for the notebooks (section 6), copy to CFS (section 4), run the run/ notebooks
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

- **Inputs** -- `run_bricks.py` takes each brick from the first of these that has it (`forced-<brick>.fits` +
  `tractor-forced-<brick>.fits`, or the archive equivalents):
  1. `scratch`: Dustin's current DR10 run, `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/<RRR>/`
     (RA 060-063 complete, plus reruns of a few failed bricks elsewhere).
  2. `cfs`: Dustin's Dec 2024 - Jan 2025 DR10 run, `/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick/`
     (RA 000-057).
  3. `archive`: our Feb-Mar 2025 output (Sept 2025 copy), `/pscratch/sd/n/nelfalou/archive/legacypm-catalogue/06[0-3]/`
     (548 bricks; their raw inputs are gone, so RM/LM/PM are rerun from the archived DCR columns).
- **Coverage of RA 000-063** (2026-10-12): 84,358 DR10-south bricks; inputs exist for **81,491**. Missing: RA 058-059
  (only 12 of 2,322 bricks exist) and 557 scattered bricks in 000-057 (196 of them in 057). These are listed in
  `missing_bricks.txt` -- ask Dustin. Everything else (000-057 minus gaps, 060-063) can run now.
- **Output** (default since 2026-10-23, run v2): `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/<RRR>/` (v1, the
  Oct 2026 run with the old corrections, stays in `/pscratch/sd/n/nelfalou/legacypm-catalogue/` until v2 is checked; the
  analysis code's `pm_analysis.COMBINED_DIR` still points at v1's `combined/` until v2 is combined) -- `forced-<brick>.fits` (raw + `dcr_`, `rm_`,
  `lm_` columns), `tractor-pm-<brick>.fits` (the catalogue), `logs/<brick>.log`, `failed.txt`.
- **pscratch is purged** (files untouched for ~8 weeks). Everything from 2025 except what you copied was lost. When a run
  finishes, copy the catalogues to CFS (section 4).
- Calibration files read by the code: `survey-ccds-decam-dr10-bprp-kd.fits` (DCR; Dustin's CFS), `ringmaps-dr10.fits`
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
Run standalone, `brick_DCR.py` spends ~45 s and 14 GB just loading the CCD table at import. `run_bricks.py` avoids that by
loading it once and sharing it with its workers. A typical full brick (~4,000 point sources, 250-380k measurements) then
takes ~35-45 s on one core: DCR 18-26 s, ring maps 5-7 s, lateral 5-7 s, PM fit ~6 s. Sparse bricks take a few seconds.

**Debug a single brick** on a login node:
```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 058 --out-root $SCRATCH/test --workers 2
```
(or run the four commands above by hand inside `shifter --image=docker:nadafalou/legacypm:5 bash`).

## 4. Running many bricks -- Slurm driver (recommended)

`slurm/run_bricks.py` finds the bricks in the given slices (`060 061`, or ranges like `000-063`), picks each brick's input
source (section 2), skips bricks with a finished `tractor-pm-*`, and runs the four steps per brick in a pool of worker
processes. `slurm/run_bricks.sbatch` runs it in Shifter on one CPU node (128 workers = one per core, 6 h).

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
sbatch run_bricks.sbatch 000-015                                        # extra args go to run_bricks.py
squeue --me
tail -f /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs/legacypm-<jobid>.out
seff <jobid>                                                            # after it ends: time + memory used
cat /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/0*/failed.txt
less /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/<RRR>/logs/<brick>.log
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 000-063 --dry-run   # "N done, 0 to run" = finished
```

- **Plan for 000-063:** 4 jobs of 16 slices. Measured (2026-10-12): 4.3-5.2 h per job for 15-23k bricks with 128
  workers, i.e. ~100 s per brick per worker on a full node (vs ~40 s for a brick alone: shared I/O and memory bandwidth).
  Keep the 6 h walltime; blocks of more than ~25k bricks need splitting. Test job 59704930 (debug, 058): peak 20 GB.
- Don't run two jobs on the same slice at the same time (both would process the same bricks).
- Logging out does not affect a submitted Slurm job, so no tmux needed.
- If a job hits the time limit or bricks fail: fix the cause and resubmit the same command; finished bricks are skipped.
  A failed brick may leave a partial `forced-<brick>.fits`; that's fine, it is rewritten from step 1.
- Memory: the shared CCD table is ~14 GB plus ~1-2 GB per busy worker; if a job runs out of memory, add `--workers 64`.
- Options: `--sources scratch,cfs,archive` (choose/reorder sources), `--input-root DIR` (+ `--from-archive`) to use one
  directory only (testing), `--out-root`, `--dry-run`.
- **Output size:** ~60 MB per brick (mostly the corrected `forced-*` file; `tractor-pm-*` ~16 MB) -> ~5 TB for 000-063 on
  pscratch (quota 20 TB). Copy at least the catalogues to CFS when a block finishes:
  `rsync -a --include='*/' --include='tractor-pm-*' --exclude='*' /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/ /global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue-v2/`
  (~1.3 TB). The `forced-*` files (~3.5 TB) are needed for per-epoch error analysis; CFS `cosmo` is 88% full, so ask
  Dustin before copying those too. Until then they live on pscratch and are purged if untouched for ~8 weeks.

**Why this instead of FireWorks?** The four steps are just four Python commands per brick, and the driver handles
"skip what's done / log what failed". It needs no database, and it submits **one** job instead of four jobs per brick
(~330,000 jobs for 000-063). The part of FireWorks that needed tmux (`qlaunch ... --nlaunches infinite`) runs on a
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
- `legacypm_workflow.py` takes one input root per call (`--input-root`, `--from-archive`) and does not pick sources
  automatically like `run_bricks.py`; it also runs each step as a separate process (45 s / 14 GB DCR import per brick).
- `qlaunch` creates `launcher_*` dirs in `fireworks/` (61k of them from 2025 are still there; safe to delete).
- `lpad reset` wipes the whole database -- don't.

## 6. Analysis and figures

**Step 1 -- combine the per-brick output** (the notebooks never read per-brick files any more):
```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
sbatch -A desi -q regular -C cpu -N 1 -t 04:00:00 --image=docker:nadafalou/legacypm:5 \
  -o /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs/combine-%j.out \
  --wrap "cd $PWD && shifter python combine_catalogue.py 000-063 --workers 128"
```
(or `~/anaconda3/envs/desi-pm/bin/python combine_catalogue.py 058` for one slice on a login node). It writes, per slice,
into `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/combined/`: `pm-<deg>.fits` (PSF objects with PMs, plus `n_epochs` and
`baseline_days` from valid epochs), `gaia-objects-<deg>.fits`, `gaia-epochs-<deg>.fits` (per-epoch rows of Gaia stars,
raw + corrected positions) and `bricks-<deg>.fits` (one row per brick). Rerun it whenever more bricks finish; it
overwrites. Only finished bricks are included. Copy `combined/` to CFS with the catalogues.

**Step 2 -- notebooks.** NERSC Jupyter (jupyter.nersc.gov), kernel **desi-pm**, opened from `run/`. Loading helpers in
`py_files/pm_analysis.py`: `load_combined(kind, degrees)` and `load_gaia_epochs(degrees)`.
- `run/err-plots.ipynb`: set `EPOCH_DEGREES` (per-epoch figures; one or a few slices) and `PM_DEGREES` (catalogue
  figures; default `000-063`) in cell 1. Saves F1, F5-F9, F11 into `writeup/plots/`.
- `run/dcr-plots.ipynb`: `EPOCH_DEGREES` in cell 1; saves F2. Builds the 14 GB CCD lookup, so use a compute node.
- Cells changed for this are marked `# [EDITED 2026-10-12]`.
- Use a Jupyter server on a compute node (or `salloc -q interactive -C cpu -t 2:00:00`) for anything beyond a slice.
- Paper figures only from `run/` notebooks, saved to `writeup/plots/` (see CLAUDE.md).

## 7. Calibration models

| Model | How to rebuild | Status |
|---|---|---|
| DCR amplitudes `K_b` (`DCR_AMPLITUDES` in `brick_DCR.py`) | `run/dcr-plots.ipynb` | K_i = K_z = 1: negligible by design (Nada, 2026-10-12) |
| Ring maps `ringmaps-dr10.fits` | `py_files/ringmap_build.py sample validate build` on a compute node (~35 min; reads `combined/gaia-epochs-*`), then `run/ringmap-build.ipynb`; copy the chosen `ringmaps-dr10-b16.fits` to CFS | Rebuilt 2026-10-23 (B1) |
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
