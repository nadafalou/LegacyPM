# LegacyPM progress tracker

Last updated: 2026-11-01. Status keys: [x] done · [~] in progress · [ ] todo · [!] blocked.
Finished work is summarised under "Done"; step-by-step history is in git (`git show f2b9252:PLAN.md` for the
pre-2026-10-29 tracker, `git log -p PLAN.md` after).

## Where things stand
RA 000-063 (the paper's analysis sample) is processed with the final pipeline (v2 run + error model v3 + exposure zero
points): 81,491 bricks, 0 failed, every catalogue on ERRMODEL 5121325b / ZPTABLE 93045419720. Output:
`/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/` (per-slice bricks, `combined/`, `slurm-logs/`); tractor-pm files and
`combined/` also on CFS `/global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue-v2/`; `forced-*` only on pscratch.

Paper (`writeup/main.tex`, 22 pp, builds cleanly) is complete apart from the abstract and the data release. It is
written as the catalogue of the whole DECam DR10 footprint, with RA 000-063 as the "analysis sample" (sec:res_catalogue).
Nada has reviewed up to Section 6.

Headline numbers (analysis sample; `legacypm-catalogue-v2/diag/results_numbers.py`, `run/sample-stats.ipynb`):
- 266.0M PSF objects with PMs, 202.2M clean (76%), 12.2M clean Gaia stars; median 32 epochs over 4.8 yr; images
  2013-09 to 2021-08; 8.1 billion measurements in the fits. r > 21 with sigma_PM < 2 / 5 / 10 mas/yr: 2.0M / 8.5M / 17.0M.
- Clean Gaia stars (10.7M), ours - Gaia: mean -0.008 / +0.003 mas/yr; robust scatter 1.68 / 1.79; PM pulls 1.20 / 1.28
  (reported, not rescaled); parallax offset -0.05 mas, pull 1.17; |dPM| > 30: 0.10%.
- Per-coordinate per-epoch scatter after all corrections: g 9.9, r 9.2, i 8.4, z 8.2 mas (from 21.2 / 14.0 / 12.7 /
  11.2); systematic left 9.0 / 8.6 / 7.9 / 7.7 (tab:error_budget).
- Quasars (DESI DR1, 126k): pulls 1.07-1.17 to r = 23.5; bias < 0.2 mas/yr at Dec < 0.

## Remaining work

### Paper
- [~] Colour-conversion check (`run/colour-conversion-check.ipynb`, job queued 2026-11-01): own fit of g-i vs BP-RP on 8
      slices, residuals vs BP-RP / G / E(B-V), and per CCD-exposure median g-i vs the reference colour. Then update the
      DCR-section sentence ("-0.57 + 1.36, within 0.08 mag for BP-RP < 2.5" is from the old slice-000 check, H5).
- [~] Lateral-colour figure rerun with the measured-epoch fix (job queued): check the spots are gone, then delete the
      caption sentence about the "dotted pattern on the CCD grid".
- [ ] pos_err_vs_mag: the Legacy-Surveys-only cloud stops flat at ~200 mas for r > 24 -- data, not axis. Explain in the
      caption or find the cause (Tractor cap? PM-sample selection?). Waiting for Nada.
- [ ] I3 (James): one sentence in sec:res_gaia that the net pmra > 0 / pmdec < 0 of Gaia stars in the sample is Galactic
      structure + solar reflex, not a systematic.
- [ ] Data release: where and in what form (two \todo's: sec:columns, Conclusions). Needs Dustin.
- [ ] Abstract (last, per Dustin).
- [ ] Nada: finish reading from Section 6 on.
- [ ] Commit outstanding work (err-plots dark-spot + C2 cells, regenerated figures, paper edits, this file).

### Questions for Dustin
- [ ] Data release (where, what form).
- [ ] CFS space for the `forced-*` files (~120 GB per slice, ~8 TB for the sample); needed for per-epoch work and
      `--redo-errors`.
- [ ] Missing bricks (`missing_bricks.txt`: all of 058-059 + 557 gaps in 000-057); forced photometry for 064-359.

### Housekeeping
- [ ] Delete test output: `/pscratch/sd/n/nelfalou/healthcheck/`, scratch texbuild dirs (Claude's rm is blocked).
- [ ] Low priority: MongoDB login for FireWorks (superseded by slurm/).

### After this paper
- [ ] Full sky (RA 064-359): needs Dustin's forced photometry (inputs stop at 069). Two-pass zero points (RUNBOOK
      section 5, tested on 064-069). Recheck pulls (error model fitted on slices 000/040) at new Dec / airmass.
- [ ] Then update the paper: full-catalogue totals in Section 2 (footprint 366,912 bricks / 22,900 deg^2 already there),
      survey-table N, Conclusions. Sample numbers in sec:res_catalogue stay unless the analysis is redone on more sky;
      calibration sample sizes (80M DCR, 294M CCD maps, 3.15M zero points) only if the models are refit. Marked by
      '% NOTE' comments in main.tex.
- [ ] Candidate fix for the full-sky run: DCR for blue objects -- per-band clip c_min g -1.90, r -2.40, i -1.85, z -1.50,
      applied to all colours (drop brick_DCR's -2 < c < 3 cut); residual at c < -1: g 13.1 -> 2.0, r 3.5 -> 1.0 mas/tan z
      (`run/dcr-blue-extension.ipynb`). Would fix hot white dwarfs (11% of their epochs get no DCR correction now).
- [ ] Optional: per-epoch confirmation of quasar DCR (QSO offsets along the refraction direction vs tan z by redshift).
- [ ] High-PM / non-Gaia candidates; science extensions (`EXTENSIONS_BRIEF.md`, `application/`).

## Decisions (Nada unless noted)
- 2026-10-12: Gaia-star PMs made absolute (Gaia motion added back); quality flagged, not cut (pm_quality bitmask).
- 2026-10-23: two-line DCR model (James); flag bad epochs, keep them in the fit; BAD_EPOCHS when >= 10% of epochs are
  > 5 sigma; g-i fallback from g-z only; Dec-ivar bug handled by DDIVFIX > INPDATE > data.
- 2026-10-26: error model v3 (isotropic + along-refraction terms); exposure zero points adopted; no PM-level floor --
  report the pull widths instead.
- 2026-10-29: g-band DCR metallicity dependence documented as a limitation, not corrected.
- 2026-10-30: paper presents the whole DECam catalogue; 2026-11-01: RA 000-063 named the "analysis sample".
- 2026-10-31: DCR for blue objects (white dwarfs) not fixed for this paper -- not worth a rerun; caution in the Discussion.
- 2026-10-31: validation details moved to appendices A-D; equations centred (fleqn removed, 2026-11-01).

## Done (summary; details in git history)
- Pipeline: two-line DCR (`dcr_model_twoline.json`); CCD maps rebuilt from 294M Gaia epochs (`ringmaps-dr10.fits`,
  16-px bins); smooth lateral-colour model (`lateral_model_poly3.json`, c0 = 1.469); exposure-CCD zero points (3.15M
  CCD-exposures, drift -0.20 / +0.12 mas/yr removed); error model v3 (`error_model.json`); measured epochs only;
  objid clashes keyed on (brickid, objid); fitsio --redo-errors; new-sky two-pass zero points (ZPFRAC,
  LEGACYPM_ZP_TABLE, redo_errors.sbatch). Input Dec-ivar bug corrected (DECFIX headers).
- Validation and investigations (notebooks in run/):
  * H1 outliers -> pm_quality flags; H2 per-stage plots; H3 error budget (now one row per band with what remains);
    H4 refraction-direction excess -> v3 along term; H5 reference colour = median BP-RP of CP calibrators;
    H6 colour modes = two populations.
  * H7 DCR vs DESI DR1 spectra (`dcr-spectra`): g-band metallicity dependence for blue stars (~20% of the DCR) -> App. B.
  * H8 CCD-map per-bin check (`ringmap-residuals`): after = noise (rms within 1 mas, robust width within 0.05 mas).
    2024 maps: right in the core in r/i/z, wildly wrong in a few bins, worse in g.
  * Q1 quasar test (`qso-test`): pulls 1.07-1.17; northern bias is quasar DCR (sign flips at z > 3), not the frame.
  * F12 completeness (`completeness`): our clean sigma<10 PMs exceed Gaia's from r = 20.9; +1.44M first PMs for Gaia
    2-parameter sources; within Gaia's range we add little (+0.7% at r < 20, +6.7% at r < 21). gaia-objects has 5.8%
    brick-overlap duplicates (dedup by ref_id). 4.2% of Gaia point sources unfitted (1-2 epochs / one night / missing).
  * Pulls figure: core Gaussian (robust width), excess in tails; small-PM stars same; sigma_mu >= 2.5: 1.06 / 1.16.
  * C2: no focal-plane dipole (<= 0.4 mas); g edge ring +2.3 mas after lateral, +0.3 after zero points.
  * White dwarfs (`wd-dcr`, 18,626): DCR residuals 2-6 mas/tan z in g; 38% of WD epochs clipped, 11% uncorrected.
  * Lateral-colour figure dark spots: plotting bug -- unmeasured epochs (13.5% in g) have junk fitted pixel positions
    that collapse to ~one focal-plane point per CCD. Pipeline unaffected; fixed in err-plots cell 15.
  * Sample statistics (`sample-stats`); new-sky health check on 064-069.
- Paper: all methods, results, application (reduced proper motion diagram), Discussion, Conclusions, science case,
  roadmap paragraph, appendices A-D; figures reviewed and restyled (pulls with log row, pos_err to r = 26, legends);
  literature todos resolved (SDSS, NIR depths, DR10 depths, Rubin DP2, NSC); Dustin's answers in (colour conversion =
  empirical fit; CP astrometry on Gaia DR2, legacypipe refits on EDR3); all bib entries verified by Nada.
- Docs: RUNBOOK.md, README.md, CLAUDE.md current.

## Data notes
- Inputs for 000-063: 81,491 of 84,358 DR10 bricks (Dustin pscratch `forced-motions-dr10/forced-brick/` for 020-023,
  051-069; Dustin CFS for 000-057; our corrected archive for 548 bricks of 060-063). Missing: 058-059 and 557 gaps
  (`missing_bricks.txt`). Nothing exists beyond slice 069.
- DR10 DECam footprint: 366,912 bricks, 22,900 deg^2, Dec -90 to +36 (`survey-bricks-dr10-south.fits.gz`); median 5-sigma
  PSF depths g 24.9, r 24.4, i 24.0, z 23.4.
- Calibration-model fits (ringmap_build, dcr-polyfit, colour-fallback, lateral-model-dr10) read the v1 combined files
  (`pm_analysis.V1_COMBINED_DIR`) so the adopted models stay reproducible; everything else reads v2.
- Old maps kept for reference on CFS `ls-motions/{rm,lm}-corrected-forced-motions/` (ringmaps.fits Apr 2024,
  lateralmaps.fits); not used.

## Meetings
- James: Tuesdays 1:00 pm (reminder Tuesday morning). Dustin: Thursdays; Slack updates in between.
