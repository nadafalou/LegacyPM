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
| 4 | `py_files/create_brick_catalogue.py` | weighted least-squares PM (and PM + parallax) fit per point source → `tractor-pm-<brick>.fits` |

Each correction is built on the previous step's output. Per-epoch errors are the forced-photometry error plus a
systematic error per band in quadrature (g 16, r 12, i 12, z 10 mas; `py_files/error_model.py`, fitted in
`run/pull-vs-mag.ipynb`). The colour corrections use g−i, or g−i predicted from g−z (0.025 mag accurate) where
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
| `py_files/` | The pipeline steps above, plus shared analysis code: `pm_analysis.py` (loading combined catalogues, quality and validation masks), `plot_style.py` (figure style), `error_model.py`, `dcr_fit.py` / `dcr_geometry.py` (DCR model fitting), the DCR coefficients `dcr_model_twoline.json`, `pm_outliers.py` (outlier diagnostics), `prep_archive_rerun.py`, and the lateral-colour coefficients `lateral_model_poly3.json`. `proper_motion.py` is an early prototype, no longer used. |
| `slurm/` | Batch running: `run_bricks.py` / `run_bricks.sbatch` run the four steps for whole RA slices on one node; `combine_catalogue.py` combines per-brick output into a few files per slice for analysis. |
| `run/` | Jupyter notebooks. **All paper figures come from here** (main ones: `err-plots`, `dcr-plots`, `ringmap-plots`, `pm-error-model`). Model fits: `lateral-model`, `dcr-polyfit`. Investigations: `h1-outliers`, `h2-correction-summary`. `run/plots/` holds exploratory (non-paper) figures; `run/e2e/` an end-to-end test of the corrections on held-out bricks. Older notebooks (`create-maps`, `map-plots`, `compare-PM`, `pm-plots`, `tests`) point at data that no longer exists. |
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

**Whole RA slices, as a batch job** (restartable; finished bricks are skipped):
```bash
cd slurm
mkdir -p /pscratch/sd/n/nelfalou/legacypm-catalogue/slurm-logs
sbatch run_bricks.sbatch 000-015            # inputs found automatically; output in legacypm-catalogue/<slice>/
shifter --image=docker:nadafalou/legacypm:5 python run_bricks.py 000-063 --dry-run   # what's done / left
```

**Combine for analysis** (per slice: point sources with PMs, Gaia stars and their epochs, per-brick summaries):
```bash
cd slurm
shifter --image=docker:nadafalou/legacypm:5 python combine_catalogue.py 000-063 --workers 128   # in a batch job
```

**Figures:** open the `run/` notebooks in NERSC Jupyter (kernel `desi-pm`, ideally on a compute node), set the slices
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

The header keywords `SIGSYS_G/R/I/Z`, `BADNSIG` and `BADFRAC` record the systematic errors and the bad-epoch rule
used; `DECFIX` records whether the input's Dec errors needed the cos²(Dec) correction (inputs made before 16 Sep 2026). The corrected per-epoch positions are in
`forced-<brick>.fits` (`dcr_`, `rm_`, `lm_` columns and `full_fit_*_ivar_total`).

A full description of the catalogue's columns and flags is planned for the paper's data section.
