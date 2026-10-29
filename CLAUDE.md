# LegacyPM

Proper motions (PMs) and parallaxes for stars in the DESI Legacy Imaging Surveys DR10 (DECam),
calibrated against Gaia. Author: Nada El-Falou; advisors Dustin Lang (PI, "dstn") and James Taylor.
Paper draft: `writeup/main.tex` (MNRAS class). Progress tracker: `PLAN.md` -- read it first and keep it updated.

## Pipeline (per 0.25x0.25 deg "brick")
Input from Dustin's forced-photometry runs: `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/<RRR>/`
(`forced-<brick>.fits` = per-epoch "full fit" positions `full_fit_dra/ddec`; `tractor-forced-<brick>.fits` = Tractor catalogue).
Corrections are applied **in sequence**; each map is built on the previous stage's output (redo errors + zero points +
PM only on finished bricks: `slurm/run_bricks.py --redo-errors`, restartable via ERRMODEL/ZPTABLE headers):

1. `py_files/brick_DCR.py` -> `dcr_full_fit_*` columns. Since 2026-10-23: Δ (mas, along the parallactic direction) =
   tan z · (a + b_blue·min(c−c_b,0) + b_red·max(c−c_b,0)), c = c_rel clipped to [−1.5, 2.5], per band (James's "two
   lines"; coefficients `py_files/dcr_model_twoline.json`, fit by `run/dcr-polyfit.ipynb`; the old linear
   K_b = g 29.82, r 7.91, i 1, z 1 is gone). Also applied to bricks without i band (skipped before 2026-10-23).
   g−i from `py_files/colours.py`: measured, else predicted from g−z (cubic, `colour_fallback.json`, fit by
   `run/colour-fallback.ipynb`; 0.025 mag; only for 0 < g−i < 3.2). c_rel = (g−i) − ref colour of the exposure/CCD,
   ref colour = −0.56 + 1.34·(Gaia BP−RP) from `survey-ccds-decam-dr10-bprp-kd.fits`.
2. `py_files/brick_ringmaps.py` -> `rm_full_fit_*`. Per CCD × filter maps of median (measured − Gaia) offset,
   16-px bins, bilinear spline. Maps since 2026-10-23: `ringmaps-dr10.fits` (CFS ls-motions/rm-corrected-forced-motions/),
   rebuilt from all 000-063 Gaia stars after the two-line DCR by `py_files/ringmap_build.py`, validated in
   `run/ringmap-build.ipynb` (removes 6-8 mas per axis in every band on held-out bricks; the Apr 2024
   `ringmaps.fits` added 9 mas in g).
3. `py_files/brick_lateralmaps.py` -> `lm_full_fit_*`. Smooth lateral-colour model (since 2026-10-10): shift =
   (g−i − c0) × 2-D order-3 polynomial in focal-plane position, per filter. Coefficients:
   `py_files/lateral_model_poly3.json`, refit 2026-10-23 (C1) by `run/lateral-model-dr10.ipynb` on all 000-063 Gaia stars
   after the two-line DCR and `ringmaps-dr10.fits`; c0 = 1.469 = median g−i of the map stars (was 1.642). (The old binned `lateralmaps.fits` was
   noise-dominated and is no longer used.)
   Steps 2-3 build maps from σ < 10 mas measurements but apply them to every measurement with a valid fit.
3b. Exposure-CCD zero points (since 2026-10-26; `py_files/zeropoints.py`) -> `zp_full_fit_dra/ddec` = lm offsets minus
   the median offset of the CCD's bright Gaia stars in that exposure (2.2 / 4.2 mas rms in RA / Dec; turbulence + per-CCD
   astrometric-solution errors; also removes a frame drift of ~0.2 mas/yr). Table (3.15M CCD-exposures) on CFS
   `ls-motions/zeropoints/zeropoints-dr10-v2.fits`, built from the v2 combined Gaia epochs. Written by the lateral step
   (or `run_bricks.py --redo-errors`); the PM fit uses zp_ columns when present. Paper: sec:zeropoints.
4. `py_files/create_brick_catalogue.py` -> `tractor-pm-<brick>.fits`. Weighted LSQ on `zp_full_fit_*` (else `lm_`), PSF sources only,
   measured epochs only (ivar > 0 and full_fit_dra != 0); 4-param (PM) and 5-param (PM+parallax) fits. Per-epoch
   error = 2x2 covariance per epoch (since 2026-10-26, `py_files/error_model.py` + `error_model.json`, fit by
   `run/error-model-fit.ipynb`): statistical + isotropic (s_b + m_b tan z)^2 + extra (a_b + k_b tan z)^2 along the
   refraction direction (RA/Dec correlated); the PM fit whitens each epoch's RA/Dec pair. The lateral step / redo writes
   `full_fit_{dra,ddec}_ivar_total`, `full_fit_radec_cov_total` + headers ERRMODEL, SIGSYS_*, SIGSYK_*, ALONGA_*, ALONGK_*.
   PM pulls vs Gaia are 1.20 / 1.28 (RA / Dec; correlated errors) -- reported in the paper, NOT rescaled (Nada). INPUT BUG: forced photometry before legacypipe 7aece157 (2026-09-16) has `full_fit_ddec_ivar`
   too large by 1/cos^2(Dec) (all CFS inputs, archive bricks; scratch 000-063 inputs are fixed). The lateral step
   multiplies it by cos^2(Dec) when the input is affected, decided by Dustin's header card DDIVFIX = T if present (input PRIMARY header; all scratch inputs have it since 2026-10-23), else by
   the input file date (INPDATE, stamped by brick_DCR), else by the data (`error_model.corrected_ddec_ivar`); headers
   record DECFIX / DECBASIS / DECRATIO. Quality is FLAGGED, not cut: PMs for every PSF object with >= 3 measured epochs;
   `pm_quality` bitmask (FEW_EPOCHS < 10, SHORT_BASELINE < 730 d, NO_COLOUR, BRIGHT, SATURATED, FIT_FAILED,
   BAD_EPOCHS 64 = >= 10% of epochs > 5σ from the fit; bad epochs are kept in the fit, counted in `pm_n_bad_epochs`).
   Figures/analysis use clean objects only (`pm_analysis.clean_mask`, pm_quality == 0); catalogues keep everything.
   Gaia stars (ref_epoch > 0) are measured relative to their Gaia motion by the forced photometry; the Gaia PM/parallax
   is added back, so all output PMs, parallaxes and positions (float64) are absolute. Catalogues made before
   2026-10-12 (incl. the archive) have relative Gaia PMs, unweighted offset terms and wrong positions -> rerun.

How to run everything: **RUNBOOK.md**. Batch runs: `slurm/run_bricks.sbatch` (one node, Shifter image
`docker:nadafalou/legacypm:5`, no database). FireWorks (`fireworks/`, MongoDB launchpad) is the older route; its Mongo
login currently fails. The image provides software only; the code always runs from this clone's `py_files/`.

## Data locations (Perlmutter)
- Current analysis sample: `/pscratch/sd/n/nelfalou/archive/legacypm-catalogue/060-063/` -- 548 valid
  `tractor-pm-*` bricks + corrected `forced-*` files. Other degree dirs are empty/0-byte.
- Gaia calibration stars: the files used to build `ringmaps.fits` (Apr 2024; `gaia-*.fits` from the old
  `/pscratch/sd/d/dstn/forced-motions-3/` run) and `/pscratch/sd/n/nelfalou/{dcr,rm,lm}-gaia-stars/` are gone (purged/empty).
  Maps survive at `/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/{rm,lm}-corrected-forced-motions/`.
  Any rebuild must pull Gaia stars (`ref_cat == 'GE'`) from the per-brick `forced-*` files, as `run/lateral-model.ipynb` does.
- `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/<deg>/` (+ `combined/`, `slurm-logs/`) is the default output root of
  `slurm/run_bricks.py` and what `pm_analysis.COMBINED_DIR` points at. The v1 root `legacypm-catalogue/` is superseded.
- New sky (beyond 000-063) needs TWO passes because the zero points are measured from processed bricks: run -> combine
  -> rebuild the zero-point table from all combined slices -> `--redo-errors` everywhere (RUNBOOK section 5; tested on
  064-069, 2026-10-29). Header ZPFRAC = fraction of epochs that got a zero point. Forced-photometry inputs exist only
  up to slice 069.

## Constraints
- Goal: RA 000-063 processed with the current code (`slurm/run_bricks.py 000-063`, see RUNBOOK.md) for the analysis
  and plots. Inputs exist for 81,491 of 84,358 bricks; the rest (058-059 and 557 gaps) are in missing_bricks.txt.
  The 548 archived bricks predate the current code; run_bricks.py rebuilds their inputs and reruns every step
  (DCR included) automatically. Run v2 (all current corrections) writes to `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/`.
- pscratch is purged: copy finished output to CFS.
- **All paper figures must be produced by notebooks in `run/`** (`err-plots.ipynb` is the main one), saved into
  `writeup/plots/`. Don't make figures with ad hoc scripts.
- **Figure style:** every paper figure calls `ps.apply()` from `py_files/plot_style.py` and is drawn at printed size
  (`ps.COLUMN` / `ps.PAGE` wide). Colours: `ps.DIVERGING` for signed quantities (blue negative, red positive,
  symmetric limits), `ps.DENSITY` (greys) for point density under overlays, `ps.SEQUENTIAL` for magnitudes,
  `ps.SERIES` + `ps.MARKERS` in fixed order for lines (always with a legend). All colour-blind safe (Nada's preference).
  Use raw strings for LaTeX labels (`r'$G_{\rm BP}$'`).
- Heavy analysis (loading hundreds of bricks) should run on a compute node / NERSC Jupyter, not a login node.
- Recent slides with the latest plots: `writeup/Proper Motions from the DESI Legacy Imaging Surveys.pdf` (July 2025).

## Writing conventions (main.tex)
- Build: `module load texlive/2024`, then `pdflatex main && bibtex main && pdflatex main && pdflatex main` in
  `writeup/` (mnras.cls ships with texlive). Build in a scratch copy to keep `writeup/` free of aux files.
- `\dstn{...}` = Dustin's comments (teal). `\todo{...}` = open items (red). `\figtodo{...}` = figure not yet made.
- British spelling ("colour", "catalogue") is used in prose.
