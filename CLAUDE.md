# LegacyPM

Proper motions (PMs) and parallaxes for stars in the DESI Legacy Imaging Surveys DR10 (DECam),
calibrated against Gaia. Author: Nada El-Falou; advisors Dustin Lang (PI, "dstn") and James Taylor.
Paper draft: `writeup/main.tex` (MNRAS class). Progress tracker: `PLAN.md` -- read it first and keep it updated.

## Pipeline (per 0.25x0.25 deg "brick")
Input from Dustin's forced-photometry runs: `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/<RRR>/`
(`forced-<brick>.fits` = per-epoch "full fit" positions `full_fit_dra/ddec`; `tractor-forced-<brick>.fits` = Tractor catalogue).
Corrections are applied **in sequence**; each map is built on the previous stage's output:

1. `py_files/brick_DCR.py` -> `dcr_full_fit_*` columns. Δ = K_b · c_rel · sqrt(X²−1) along parallactic direction.
   K_b in mas/mag (`DCR_AMPLITUDES`: g 29.82, r 7.91, i 1, z 1). c_rel = (g−i) − ref colour of the exposure/CCD,
   ref colour = −0.56 + 1.34·(Gaia BP−RP) from `survey-ccds-decam-dr10-bprp-kd.fits`.
2. `py_files/brick_ringmaps.py` -> `rm_full_fit_*`. Per CCD × filter maps of median (Gaia − measured) offset,
   16-px bins, bilinear spline. Maps: `ringmaps.fits`.
3. `py_files/brick_lateralmaps.py` -> `lm_full_fit_*`. Smooth lateral-colour model (since 2026-10-10): shift =
   (g−i − 1.642) × 2-D order-3 polynomial in focal-plane position, per filter. Coefficients:
   `py_files/lateral_model_poly3.json`, fit by `run/lateral-model.ipynb`. (The old binned `lateralmaps.fits` was
   noise-dominated and is no longer used.)
   Steps 2-3 build maps from σ < 10 mas measurements but apply them to every measurement with a valid fit.
4. `py_files/create_brick_catalogue.py` -> `tractor-pm-<brick>.fits`. Weighted LSQ on `lm_full_fit_*`, PSF sources only,
   4-param (PM) and 5-param (PM+parallax) fits; 14.8 mas floor added in quadrature to per-epoch errors;
   cuts MIN_EPOCHS=5, MIN_BASELINE_DAYS=180, cond<1e8.

Orchestration: FireWorks (`fireworks/`, launchpad on MongoDB at NERSC) + Docker image (`docker/`). How to run is
**not yet documented** -- see PLAN.md "Runbook" section.

## Data locations (Perlmutter)
- Current analysis sample: `/pscratch/sd/n/nelfalou/archive/legacypm-catalogue/060-063/` -- 548 valid
  `tractor-pm-*` bricks + corrected `forced-*` files. Other degree dirs are empty/0-byte.
- Gaia calibration stars: the files used to build `ringmaps.fits` (Apr 2024; `gaia-*.fits` from the old
  `/pscratch/sd/d/dstn/forced-motions-3/` run) and `/pscratch/sd/n/nelfalou/{dcr,rm,lm}-gaia-stars/` are gone (purged/empty).
  Maps survive at `/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/{rm,lm}-corrected-forced-motions/`.
  Any rebuild must pull Gaia stars (`ref_cat == 'GE'`) from the per-brick `forced-*` files, as `run/lateral-model.ipynb` does.
- `/pscratch/sd/n/nelfalou/legacypm-catalogue/<deg>/` dirs are now empty (notebooks still point there -- repoint to archive).

## Constraints
- New catalogue files can't be generated right now: waiting on Dustin's forced-photometry regeneration.
  Work with the 548 archived bricks. The archived bricks predate the epoch/baseline guard -> apply that cut at
  analysis time (recompute N_epochs/baseline from the `forced-*` files).
- **All paper figures must be produced by notebooks in `run/`** (`err-plots.ipynb` is the main one), saved into
  `writeup/plots/`. Don't make figures with ad hoc scripts.
- Heavy analysis (loading hundreds of bricks) should run on a compute node / NERSC Jupyter, not a login node.
- Recent slides with the latest plots: `writeup/Proper Motions from the DESI Legacy Imaging Surveys.pdf` (July 2025).

## Writing conventions (main.tex)
- Build: `module load texlive/2024`, then `pdflatex main && bibtex main && pdflatex main && pdflatex main` in
  `writeup/` (mnras.cls ships with texlive). Build in a scratch copy to keep `writeup/` free of aux files.
- `\dstn{...}` = Dustin's comments (teal). `\todo{...}` = open items (red). `\figtodo{...}` = figure not yet made.
- British spelling ("colour", "catalogue") is used in prose.
