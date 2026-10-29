# LegacyPM runbook

How to run everything. Rewritten 2026-10-29. If something here turns out to be wrong, fix it here.

**State:** RA 000-063 is processed with the current code (run "v2": 81,491 bricks, 266M point sources with PMs),
combined, and used by the paper. Inputs for more sky do not exist yet (section 3); everything below works for new sky
once Dustin's forced photometry covers it.

## 1. Quick reference

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
IMG="shifter --image=docker:nadafalou/legacypm:5"

# what exists / what is left, per slice (no job needed)
$IMG python run_bricks.py 064-069 --dry-run
# process slices: one node per block of ~16 slices / <= 25k bricks; resubmit the same command to continue
sbatch run_bricks.sbatch 064-079
# combine per slice for analysis
sbatch -A desi -q regular -C cpu -N 1 -t 04:00:00 --image=docker:nadafalou/legacypm:5 \
  -o /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/combine-%j.out \
  --wrap "cd $PWD && shifter python combine_catalogue.py 000-079 --workers 64"
# after a new error model or zero-point table: errors + zero points + PM fit only
sbatch redo_errors.sbatch 000-079
# figure notebooks on a CPU node
sbatch run_notebooks.sbatch err-plots dcr-plots pm-error-model
# monitor
squeue --me; tail -f /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/<name>-<jobid>.out
```

New sky needs the two-pass procedure of section 5, because the exposure zero points are measured from processed bricks.

## 2. Software

| What | Where | Used for |
|---|---|---|
| Shifter image `docker:nadafalou/legacypm:5` | Perlmutter | The pipeline (`run_bricks.py`, `combine_catalogue.py`). Provides legacypipe, astrometry.net, fitsio, astropy. |
| conda env `desi-pm` (`~/anaconda3/envs/desi-pm`) | login node, batch jobs, Jupyter kernel "desi-pm" | Notebooks in `run/`, calibration builds, analysis. |

The image only provides software; **the code always runs from this clone** (`/pscratch/sd/n/nelfalou/LegacyPM/`), so
code changes need no image rebuild. (An older clone in `~/LegacyPM` is not used.) Logging out does not affect submitted
Slurm jobs.

## 3. Data

**Inputs** (Dustin's forced photometry): per brick `forced-<brick>.fits` + `tractor-forced-<brick>.fits`.
`run_bricks.py` takes each brick from the first source that has it:

| Source | Path | Covers (2026-10-29) | Dec-ivar bug |
|---|---|---|---|
| `scratch` | `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/<RRR>/` | 020-023, 051-069 (064-069: 156 bricks) | fixed (header `DDIVFIX = T`) |
| `cfs` | `/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick/<RRR>/` | 000-057 | present (corrected by us) |
| `archive` | `/pscratch/sd/n/nelfalou/archive/legacypm-catalogue/06[0-3]/` | 548 bricks of 060-063 | present (corrected by us) |

Nothing exists beyond 069: **more sky needs Dustin to run the forced photometry first.** Within 000-063, 2,867 of 84,358
DR10-south bricks have no input (`missing_bricks.txt`; mostly 058-059).

**Output** (`run_bricks.py --out-root`, default `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/`):

| Path | What |
|---|---|
| `<RRR>/forced-<brick>.fits` | per-epoch rows: raw + `dcr_`, `rm_`, `lm_`, `zp_` positions; `full_fit_*_ivar_total`, `full_fit_radec_cov_total` |
| `<RRR>/tractor-pm-<brick>.fits` | **the catalogue**: Tractor columns + PMs, parallax, flags; headers record the models used |
| `<RRR>/tractor-forced-<brick>.fits` | archive bricks only (rebuilt input) |
| `<RRR>/logs/<brick>.log`, `failed.txt`, `failed_redo_errors.txt` | per-brick logs, failures of the last run |
| `combined/` | per-slice analysis files (section 7) |
| `slurm-logs/` | job logs |

Size: ~130 GB per slice (mostly `forced-*`), ~170 GB `combined/` for 64 slices. pscratch quota is 20 TB and **pscratch
is purged** (files untouched ~8 weeks): copy finished output to CFS (section 7).

**Calibration files** read by the pipeline:

| File | Used by | Built by (section 9) |
|---|---|---|
| `survey-ccds-decam-dr10-bprp-kd.fits` (Dustin's CFS) | DCR reference colours | Dustin |
| `py_files/dcr_model_twoline.json` | DCR | `run/dcr-polyfit.ipynb` |
| `py_files/colour_fallback.json` | g-i from g-z | `run/colour-fallback.ipynb` |
| CFS `ls-motions/rm-corrected-forced-motions/ringmaps-dr10.fits` | CCD maps | `py_files/ringmap_build.py` + `run/ringmap-build.ipynb` |
| `py_files/lateral_model_poly3.json` | lateral colour | `run/lateral-model-dr10.ipynb` |
| CFS `ls-motions/zeropoints/zeropoints-dr10-v2.fits` | exposure zero points | `py_files/zeropoints.py build` |
| `py_files/error_model.json` | per-epoch errors | `run/error-model-fit.ipynb` |

## 4. The pipeline (per brick)

`run_bricks.py` runs, per brick, in a pool of workers sharing the big tables:

| Step | Code | Adds to `forced-<brick>.fits` |
|---|---|---|
| (0) archive bricks only: rebuild inputs | `prep_archive_rerun.py` | drops all derived columns |
| 1 DCR | `brick_DCR.py` | `dcr_full_fit_*`; header `INPDATE` (input file date), `DDIVFIX` copied from the input |
| 2 CCD maps | `brick_ringmaps.py` | `rm_full_fit_*` |
| 3 lateral colour, errors, zero points | `brick_lateralmaps.py` (+ `error_model.py`, `zeropoints.py`) | `lm_full_fit_*`; `full_fit_{dra,ddec}_ivar_total`, `full_fit_radec_cov_total`; `zp_full_fit_*` |
| 4 PM fit | `create_brick_catalogue.py` | writes `tractor-pm-<brick>.fits` |

The PM fit uses `zp_full_fit_*` (else `lm_`), measured epochs only, the full 2x2 error of every epoch, and flags rather
than cuts (`pm_quality`). Catalogue headers record what produced it: `ERRMODEL` (error-model ID) and `SIGSYS_*`,
`SIGSYK_*`, `ALONGA_*`, `ALONGK_*`; `ZPTABLE` (zero-point table ID), `ZPFRAC` (fraction of measured epochs that had a
zero point), `ZPCORR`; `DECFIX` / `DECBASIS` / `DECRATIO` (Dec-ivar bug correction); `BADNSIG`, `BADFRAC`; `OBJKEY`.
See CLAUDE.md for what each step does.

Timing: one node, 128 workers: ~15-23k bricks in 3.5-5 h (full pipeline); 64 workers: ~12-17k bricks in 1-1.8 h
(`--redo-errors`). Peak memory: the 14 GB CCD table plus ~1-2 GB per busy worker.

**One brick by hand** (debugging), on a login node:
```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 064 --out-root $SCRATCH/test --workers 2
```

## 5. Processing new sky (two passes)

The zero-point table is measured from the Gaia stars of processed bricks, so a new region gets its zero points only
after it has been processed once. In the first pass, epochs on exposure-CCDs not yet in the table get no zero point;
the catalogue headers say how many did (`ZPFRAC`, ~0 for new sky).

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
IMG="shifter --image=docker:nadafalou/legacypm:5"
# 1. first pass (full pipeline); blocks of <= ~25k bricks per job; resubmit the same command if a job stops
$IMG python run_bricks.py 064-079 --dry-run
sbatch run_bricks.sbatch 064-079
cat /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/06[4-9]/failed.txt      # fix causes, resubmit
# 2. combine the new slices (wrap command of section 1 with 064-079)
# 3. rebuild the zero-point table from ALL combined slices (old + new), to a new file first
sbatch -A desi -q regular -C cpu -N 1 -t 01:00:00 -o /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/zp-%j.out \
  --wrap "cd /pscratch/sd/n/nelfalou/LegacyPM/py_files && ~/anaconda3/envs/desi-pm/bin/python zeropoints.py build --workers 32 \
          --out /global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/zeropoints/zeropoints-dr10-new.fits"
#    then make it the table the pipeline uses: move it over zeropoints-dr10-v2.fits (keep the old one under another name)
# 4. second pass: errors + zero points + PM fit with the new table, on EVERY slice (the table ID changed, so all bricks
#    are redone; the old area changes little: its exposure-CCDs gain a few more stars)
sbatch redo_errors.sbatch 000-079
# 5. combine again (all slices), check (section 6), copy to CFS (section 7)
```

Building the table reads every combined `gaia-epochs-*` file (~10 min for 70 slices; memory grows with the number of
slices: the whole DECam sky needs a large-memory node or splitting by exposure number). `LEGACYPM_ZP_TABLE=<path>`
makes the pipeline use another table (tests).

Health check 2026-10-29 (`/pscratch/sd/n/nelfalou/healthcheck/`, test output; safe to delete): the full pipeline ran on
slices 064-069 (156 bricks, outside the calibrated area) without failures. Pass 1: 12% of epochs had a zero point (0% in
065-069). Pass 2 (table rebuilt from 70 slices into a test file, `LEGACYPM_ZP_TABLE`, then the redo): median 99.96%,
minimum 97.6% per brick. The production table was not changed.

**New sky checklist**
- Calibrations transfer: DCR, CCD maps, lateral colour and colour fallback are per CCD / per band / global. The error
  model was fitted on slices 000 and 040: check the pulls on new sky (`run/error-model-fit.ipynb`, `run/pull-vs-mag.ipynb`
  with the new slices; `py_files/pm_pulls.py <slice>` first) and the Gaia comparison before trusting the errors.
- Disk: ~130 GB per slice on pscratch (20 TB quota). For large areas run in blocks and copy each block to CFS; the
  `forced-*` files are needed again by pass 2.
- Dense fields (Magellanic Clouds, low Galactic latitude): bigger bricks, more memory per worker. If a job runs out of
  memory, use fewer workers (`--workers 64`).

## 6. Checking a run

```bash
$IMG python run_bricks.py 000-079 --dry-run                     # "N done, 0 to run" for every slice = finished
$IMG python run_bricks.py 000-079 --redo-errors --dry-run       # "0 to redo" = every catalogue on the current models
cat /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/0*/failed*.txt
seff <jobid>                                                    # time and memory of a finished job
```
- `combined/bricks-<deg>.fits` has one row per brick, including `zp_frac` (fraction of epochs with a zero point):
  bricks with low `zp_frac` after pass 2 lack Gaia stars on their exposures.
- Gaia comparison of the whole catalogue: `legacypm-catalogue-v2/diag/gaia_check.py` (debug-queue job; prints mean
  offsets, scatter and pull widths by magnitude and declination). Reference values (000-063): mean -0.008 / +0.003
  mas/yr, scatter 1.68 / 1.79 mas/yr, pulls 1.20 / 1.28.

## 7. Combine, and copy to CFS

`combine_catalogue.py <slices>` writes per slice into `<out-root>/combined/`: `pm-<deg>.fits` (PSF objects with PMs),
`gaia-objects-<deg>.fits`, `gaia-epochs-<deg>.fits` (per-epoch rows of Gaia stars, all position stages and total
errors), `bricks-<deg>.fits`. It overwrites; rerun after any rerun or redo. ~2 h for 64 slices with 64 workers.

Copy to CFS when a run is final:
```bash
rsync -a --include='*/' --include='tractor-pm-*' --exclude='*' /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/ \
  /global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue-v2/
rsync -a /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/combined/ \
  /global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue-v2/combined/
```
The `forced-*` files (~120 GB per slice) are needed for per-epoch analysis and for `--redo-errors`; ask Dustin about CFS
space before copying them.

## 8. Reruns after a model change

| Changed | What to rerun |
|---|---|
| `error_model.json` or the zero-point table | `redo_errors.sbatch <slices>` (errors + zero points + PM fit), then combine |
| DCR model, colour fallback, CCD maps, lateral model | the full pipeline: use a **new** `--out-root` (e.g. `legacypm-catalogue-v3`; `run_bricks.py` skips bricks that already have a catalogue), then rebuild the zero-point table from the new combined files and redo errors (pass 2 of section 5); point `pm_analysis.COMBINED_DIR` and the defaults in `run_bricks.py` / `combine_catalogue.py` at the new root |
| PM fit / flags only (`create_brick_catalogue.py`) | the redo reruns the PM step, but skips bricks already on the current error model and table: use a new `--out-root` (full pipeline), or move the affected `tractor-pm-*` aside first |

The calibration builds read the v1 combined files they were fitted on (`pm_analysis.V1_COMBINED_DIR`), so the adopted
models stay reproducible; refitting them on v2 or more sky means pointing them at `COMBINED_DIR`.

## 9. Calibration models

| Model | File | Rebuild with | Data | Last built |
|---|---|---|---|---|
| DCR (two lines in colour x tan z) | `dcr_model_twoline.json` | `run/dcr-polyfit.ipynb` (last cell) | v1 combined, 000-063 | 2026-10-23 |
| g-i from g-z | `colour_fallback.json` | `run/colour-fallback.ipynb` | v1 combined, 4 slices | 2026-10-23 |
| CCD maps (16-px bins) | `ringmaps-dr10.fits` (CFS) | `ringmap_build.py sample validate build` (compute node, ~35 min) + `run/ringmap-build.ipynb`; copy `combined/ringmaps-dr10-b16.fits` to CFS | v1 combined, 000-063 | 2026-10-23 |
| Lateral colour | `lateral_model_poly3.json` | `run/lateral-model-dr10.ipynb` (compute node; writes a candidate file, adopt by moving it) | ring-map sample | 2026-10-23 |
| Exposure zero points | `zeropoints-dr10-v2.fits` (CFS) | `zeropoints.py build` (section 5) | v2 combined, 000-063 | 2026-10-26 |
| Error model | `error_model.json` | `pm_pulls.py <slice>` (compute node), then `run/error-model-fit.ipynb` (writes a candidate) | v2, slices 000 + 040, after zero points | 2026-10-26 |

Fitting order matters: DCR -> CCD maps -> lateral colour -> zero points -> error model; each is fitted on the output of
the previous ones. A change early in the chain means refitting everything after it.

## 10. Notebooks and figures

```bash
cd /pscratch/sd/n/nelfalou/LegacyPM/slurm
sbatch run_notebooks.sbatch err-plots dcr-plots pm-error-model h3-error-budget
tail -f /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/slurm-logs/notebooks-<jobid>.out   # '### <name> ok' / FAILED
```
Notebooks run in place (outputs saved); paper figures go to `writeup/plots/`, diagnostics to `run/plots/`. Paper figures
come only from `run/` notebooks (CLAUDE.md). Loading helpers: `py_files/pm_analysis.py` (`load_combined(kind, slices)`,
default all combined slices; `clean_mask`). Slice choices are at the top of each notebook.

| Notebook | Makes |
|---|---|
| `err-plots` | scatter vs magnitude, lateral colour, positional errors, footprint, PM vs Gaia, pulls, parallaxes |
| `dcr-plots` | DCR model per band, DCR before/after |
| `ringmap-plots` | CCD maps (one CCD, whole focal plane) |
| `pm-error-model` | PM uncertainty vs magnitude |
| `h3-error-budget` | error-budget table (`writeup/tables/error_budget.tex`), error floor vs magnitude |
| `dcr-polyfit`, `colour-fallback`, `ringmap-build`, `lateral-model-dr10`, `error-model-fit`, `pull-vs-mag` | calibration fits and their checks |
| `h1`, `h4`, `h6`, `lateral-model` (superseded) | investigations, kept for the record |

## 11. Troubleshooting

- **Job hit the time limit, or bricks failed:** fix the cause, resubmit the same command; finished bricks are skipped.
  Files are written to `.tmp` and renamed, so an interrupted brick never leaves a truncated output.
- **Out of memory:** fewer workers (`--workers 64` or 48). A memory leak in `--redo-errors` (astropy table objects) was
  fixed on 2026-10-27; if memory grows with every brick again, profile one worker on ~10 bricks.
- **Dec errors look wrong for some inputs:** check the catalogue header `DECFIX` / `DECBASIS` (section 3, Dec-ivar bug).
- **Zero points missing:** `ZPFRAC` in the catalogue headers / `zp_frac` in `combined/bricks-*.fits`; new sky needs
  pass 2 (section 5).
- **Two jobs on the same slices:** don't; both would process the same bricks.

## 12. Paper

```bash
module load texlive/2024
cp -r /pscratch/sd/n/nelfalou/LegacyPM/writeup $SCRATCH/wbuild && cd $SCRATCH/wbuild
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Build in a copy to keep `writeup/` free of aux files.

## Appendix: FireWorks (the 2025 route; not used)

`fireworks/` holds the older MongoDB-based workflow (`legacypm_workflow.py`, one Slurm job per step per brick). Its Mongo
login failed in Oct 2026 and it predates every correction added since; use `run_bricks.py`.
