# LegacyPM: proper motions from the DESI Legacy Imaging Surveys

Proper motions (and parallaxes) for point sources in the DESI Legacy Imaging Surveys DR10 (DECam), down to
r ≈ 25 mag, about four magnitudes fainter than Gaia. Each star's position is measured in every individual exposure
by forced photometry. The positions are corrected for systematic errors and calibrated against Gaia, and a straight
line (or line plus parallax) is fitted through them.

Author: Nada El-Falou

**For the science** (method, corrections, validation against Gaia, error budget), read the paper draft in
[`writeup/main.tex`](writeup/main.tex), with its figures in [`writeup/plots/`](writeup/plots/). Earlier slides are in
`writeup/Proper Motions from the DESI Legacy Imaging Surveys.pdf`.

## How it works

The sky is processed in DR10 "bricks" (0.25° × 0.25°), grouped into 1° RA "slices" (`000`–`359`). For each brick:

| Step | Code | What it does |
|---|---|---|
| input | Dustin Lang's forced photometry | `forced-<brick>.fits`: one row per object per exposure, with its measured offset (`full_fit_dra/ddec`); `tractor-forced-<brick>.fits`: the Tractor catalogue |
| 1 | `py_files/brick_DCR.py` | differential chromatic refraction: shift along the parallactic direction = tan z × two straight lines in colour, per band (`dcr_model_twoline.json`) |
| 2 | `py_files/brick_ringmaps.py` | CCD distortions (silicon "tree rings", edge and tape effects): per-CCD, per-band maps |
| 3 | `py_files/brick_lateralmaps.py` | lateral colour (chromatic aberration): smooth model in focal-plane position × colour; also writes the total per-epoch errors |
| 3b | `py_files/zeropoints.py` | exposure zero points: subtracts, per CCD and exposure, the median offset of its Gaia stars (ties every exposure to Gaia) |
| 4 | `py_files/create_brick_catalogue.py` | weighted least-squares PM (and PM + parallax) fit per point source → `tractor-pm-<brick>.fits` |

Each correction is built on the previous step's output. Each epoch's error is a 2×2 covariance: the forced-photometry
error plus a systematic term that grows with airmass and an extra term along the refraction direction
(`py_files/error_model.py` + `error_model.json`, fitted in `run/error-model-fit.ipynb`); the PM fit uses the full
covariance. The colour corrections use g−i, or g−i predicted from g−z (0.025 mag accurate) where
i is missing (`py_files/colours.py`, fitted in `run/colour-fallback.ipynb`). For Gaia stars, the Gaia motion is added back, so
every PM in the catalogue is absolute.

**Quality is flagged, not cut.** Every point source with at least 3 measured epochs gets a PM. The `pm_quality`
bitmask records why one may be unreliable:

| Bit | Flag | Meaning |
|---|---|---|
| 1 | FEW_EPOCHS | fewer than 10 measured epochs |
| 2 | SHORT_BASELINE | epochs span less than 2 years |
| 4 | NO_COLOUR | no g−i colour, measured or from g−z, so no DCR or lateral-colour correction |
| 8 | BRIGHT | near or is a bright star (Tractor mask) |
| 16 | SATURATED | saturated in g, r, i or z |
| 32 | FIT_FAILED | too few epochs, or a singular fit |
| 64 | BAD_EPOCHS | at least 10% of the epochs more than 5σ from the fit (kept in the fit; count in `pm_n_bad_epochs`) |

`pm_quality == 0` selects "clean" objects, which is what the figures use.

## Repository layout

| Directory | Contents |
|---|---|
| `py_files/` | The pipeline steps above, plus shared analysis code: `pm_analysis.py` (loading combined catalogues, quality and validation masks), `plot_style.py` (figure style), `error_model.py`, `dcr_fit.py` / `dcr_geometry.py` (DCR model fitting), the DCR coefficients `dcr_model_twoline.json`, `colours.py` (g−i with the g−z fallback), `zeropoints.py` (exposure zero points), `error_model.json` (per-epoch error model), `pm_outliers.py` (outlier diagnostics), `prep_archive_rerun.py`, and the lateral-colour coefficients `lateral_model_poly3.json`. `proper_motion.py` is an early prototype, no longer used. |
| `slurm/` | Batch running: `run_bricks.py` / `run_bricks.sbatch` run all steps for whole RA slices on one node; `redo_errors.sbatch` reruns errors, zero points and the PM fit only; `run_notebooks.sbatch` executes figure notebooks; `combine_catalogue.py` combines per-brick output into a few files per slice for analysis. |
| `run/` | Jupyter notebooks. **All paper figures come from here** (main ones: `err-plots`, `dcr-plots`, `ringmap-plots`, `pm-error-model`). Model fits: `dcr-polyfit`, `colour-fallback`, `ringmap-build`, `lateral-model-dr10`, `error-model-fit` (checks: `pull-vs-mag`, `h3-error-budget`). Investigations: `h1-outliers`, `h2-correction-summary`. `run/plots/` holds exploratory (non-paper) figures; `run/e2e/` an end-to-end test of the corrections on held-out bricks. Older notebooks (`create-maps`, `map-plots`, `compare-PM`, `pm-plots`, `tests`) point at data that no longer exists. |
| `writeup/` | The paper draft (MNRAS LaTeX, `main.tex`, `mybib.bib`), its figures (`plots/`), slides and figure notes. |
| `docker/` | Dockerfile for the software image (legacypipe + FireWorks). The image only provides software: the code always runs from this repository's `py_files/`. |
| `fireworks/` | The older FireWorks/MongoDB workflow for submitting bricks; superseded by `slurm/`. |

Project files: `RUNBOOK.md` (how to run everything, in detail), `PLAN.md` (progress tracker and findings),
`CLAUDE.md` (project context for the AI assistant).

## Quick start (NERSC Perlmutter)

The pipeline runs in the Shifter image `docker:nadafalou/legacypm:5`; the notebooks use a conda environment with
legacypipe, astrometry.net, fitsio, astropy and matplotlib (Jupyter kernel `desi-pm`). Paths below are the
current defaults; `RUNBOOK.md` has the details.

**One brick, interactively** (to test or debug; directories need a trailing `/`):
```bash
IN=/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/060/   # forced-<brick>.fits + tractor-forced-<brick>.fits
OUT=$SCRATCH/test/ ; mkdir -p $OUT ; F=forced-0600m045.fits
shifter --image=docker:nadafalou/legacypm:5 bash -c "cd py_files &&
  python brick_DCR.py $F $OUT $IN && python brick_ringmaps.py $F $OUT $OUT &&
  python brick_lateralmaps.py $F $OUT $OUT $IN && python create_brick_catalogue.py $F $OUT $OUT $IN"
```

**Whole RA slices, as batch jobs** (restartable; finished bricks are skipped). Output goes to
`/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/<slice>/`, logs to its `slurm-logs/`:
```bash
cd slurm
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 064-079 --dry-run   # what exists / what's left
sbatch run_bricks.sbatch 064-079            # full pipeline, inputs found automatically
sbatch redo_errors.sbatch 000-079           # after a new error model or zero-point table: errors + zero points + PM only
```
New sky needs two passes (the zero points are measured from processed bricks): see `RUNBOOK.md`, section 5.

**Combine for analysis** (per slice: point sources with PMs, Gaia stars and their epochs, per-brick summaries):
```bash
cd slurm
shifter --image=docker:nadafalou/legacypm:5 python combine_catalogue.py 000-063 --workers 64   # in a batch job (RUNBOOK section 1)
```

**Figures:** `cd slurm && sbatch run_notebooks.sbatch err-plots dcr-plots pm-error-model` runs notebooks on a CPU node; or open the `run/` notebooks in NERSC Jupyter (kernel `desi-pm`, ideally on a compute node), set the slices
in the first cell, and run. Paper figures are saved to `writeup/plots/`.

**Paper:** `module load texlive/2024`, then in a copy of `writeup/`:
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Output

Per brick: `tractor-pm-<brick>.fits`, the Tractor catalogue with these columns added:
- PMs from the 4-parameter fit (`pmra_no_plx`, `pmdec_no_plx`);
- PMs and parallax from the 5-parameter fit (`pmra_plx`, `pmdec_plx`, `parallax_new`);
- positions at the reference epoch (`new_ra_*`, `new_dec_*`) and `ref_mjd`;
- inverse variances for all of the above;
- `pm_flag`, `pm_quality`, `pm_n_epochs`, `pm_baseline_days`, `pm_n_bad_epochs`, `pm_colour_source` (g−i used: 1 measured, 2 from g−z, 0 none).

Header keywords record what produced each catalogue: `ERRMODEL` and the error-model parameters (`SIGSYS_*`, `SIGSYK_*`,
`ALONGA_*`, `ALONGK_*`); `ZPTABLE`, `ZPFRAC` (fraction of epochs with an exposure zero point); `BADNSIG`, `BADFRAC`
(bad-epoch rule); `DECFIX` (whether the input's Dec errors needed the cos²(Dec) correction, inputs made before
16 Sep 2026). The corrected per-epoch positions are in `forced-<brick>.fits` (`dcr_`, `rm_`, `lm_`, `zp_` columns,
`full_fit_*_ivar_total`, `full_fit_radec_cov_total`). The paper (`writeup/main.tex`, Tables of flags and columns)
describes every column and flag.
