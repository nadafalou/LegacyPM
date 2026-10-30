# LegacyPM progress tracker

Last updated: 2026-10-29. Status keys: [x] done · [~] in progress · [ ] todo · [!] blocked.
Cleaned up 2026-10-29: finished work is summarised under "Done"; the full step-by-step history (old item numbers
A1-A10, B, C, D, E, F, H, I) is in git (`git show f2b9252:PLAN.md`).

## Where things stand
RA 000-063 is processed with the final pipeline (v2 run + error model v3 + exposure zero points): 81,491 bricks,
0 failed, every catalogue on ERRMODEL 5121325b / ZPTABLE 93045419720. Output: `/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/`
(per-slice bricks, `combined/`, `slurm-logs/`). tractor-pm files and `combined/` copied to CFS
`/global/cfs/cdirs/cosmo/work/users/nelfalou/legacypm-catalogue-v2/` (2026-10-29/30); `forced-*` files only on pscratch.
Paper (`writeup/main.tex`) has methods, results and an application section; it builds cleanly (18 pp).

Headline numbers (full sample; `legacypm-catalogue-v2/diag/results_numbers.py`, `paper_numbers.py`):
- 266.0M PSF objects with PMs, 202.2M clean (pm_quality == 0, 76%), 12.2M clean Gaia stars; median 32 epochs over 4.8 yr;
  r > 21 with sigma_PM < 2 / 5 / 10 mas/yr: 2.0M / 8.5M / 17.0M.
- Clean Gaia stars (10.7M), ours - Gaia: mean -0.008 / +0.003 mas/yr; robust scatter 1.68 / 1.79 (G 17-19 1.42 / 1.53);
  PM pulls 1.20 / 1.28 (reported, not rescaled -- Nada); parallax offset -0.05 mas, parallax pull 1.17; |dPM| > 30: 0.10%.
- Per-axis per-epoch scatter after all corrections (bright clean Gaia stars): g 9.9, r 9.2, i 8.4, z 8.2 mas
  (from 21.2 / 14.0 / 12.7 / 11.2); `writeup/tables/error_budget.tex`.

## Remaining work (priority order)

### 1. Data safety
- [x] tractor-pm + `combined/` on CFS (2026-10-30; 262 combined files, matches pscratch).
- [ ] Ask Dustin about CFS space for the `forced-*` files (~120 GB per slice, ~8 TB); needed for per-epoch work and
      `--redo-errors`.

### 2. Validation still missing
- [x] Q1 (2026-10-30). Quasar test (`run/qso-test.ipynb` -> writeup/plots/qso_pm.pdf; paper sec:res_qso, fig:qso).
      DESI DR1 QSOs matched at 0.5": 126,448 clean non-Gaia, r < 23.5. Pulls 1.07-1.17 (PM), 1.06-1.11 (plx), flat to
      r 23.5. Bias Dec < 0: |median| < 0.2 mas/yr; Dec > 10: -0.5..-4 (fainter = larger); plx -0.6..-1.6 everywhere.
      Sign flips at z > 3 (pmdec +2.5, plx +1.5) -> quasar DCR (emission lines / Lya forest), not the frame: Gaia stars
      at Dec > 10, G 19-21 agree with Gaia to 0.1 mas/yr. Optional: per-epoch confirmation (QSO offsets along the
      refraction direction vs tan z by redshift; needs forced files, ~1 h compute).
- [x] F12 (James; 2026-10-30) `run/completeness.ipynb` -> writeup/plots/completeness_vs_gaia.pdf (not yet in the paper).
      22.8M unique Gaia sources (gaia-objects has 5.8% brick-overlap duplicates, brick_primary = False, never fitted;
      de-duplicated by ref_id). Our clean sigma<10 counts exceed Gaia-with-PM counts from r = 20.9; r > 21: 17.0M (ours,
      sigma<10) vs 0.53M (Gaia with PM). 1.44M Gaia sources without a Gaia PM get a clean PM from us (median sigma 1.8
      mas/yr, 91% < 5; median G 21.0). We fit 94.6% of Gaia sources with a Gaia PM; clean 65-68% at G 17-20.5 (flagged
      30.6%: SHORT_BASELINE 17.3, FEW_EPOCHS 14.0, NO_COLOUR 7.1, BRIGHT 5.9, BAD_EPOCHS 4.7, SAT 1.0); none clean at
      G < 16 (Tractor MEDIUM mask marks every G < 16 Gaia star -> BRIGHT). In the paper: sec:res_completeness,
      fig:completeness. Open \todo: why 4.0% of Gaia point sources have no fit (overlap duplicates of missing bricks?).
- [ ] C2. Left/right dipole in the lateral-colour figure (colour-independent, so not lateral colour): check whether it
      survives the zero points; if so, CCD maps / WCS.
- [ ] Optional: zero points vs time and across the focal plane (paper \todo); paper version of the DESI metallicity plot.

### 3. Paper
- [x] 2026-10-30 Reviewed all paper figures: legends fixed (scatter_vs_mag, footprint, error_floor), labels
      'statistical only' / 'full (error model)' in pm_errors; text fixed: pos_err numbers, DCR sample size, parallax
      numbers (scatter 3.7-4.1 mas at G 16-19, 5.8 at 20.7; median error 3.6), pm_vs_gaia streak sentence, lateral
      caption (colour-independent residual). Noted, not changed: g-band DCR model ~10 mas/tan z below the data at
      c_rel < -1.2 (clip at -1.5; data flatten earlier).
- [ ] Write: Discussion (only a commented TODO now; notes at the end of main.tex: empirical maps vs Bernstein's
      parametric model, limitations incl. the g-band metallicity DCR term), Conclusions, abstract (last, per Dustin).
- [ ] Sample description: imaging date range, N_epoch / baseline distributions, numbers of bricks and measurements
      (three \todo's).
- [ ] Data release: where and in what form (Dustin) -> sec:columns \todo.
- [ ] I2 (James). Science case for PMs at 22-25 mag (with F12). I3: say that the net pmra > 0 / pmdec < 0 of Gaia stars
      here is Galactic structure + solar reflex, not a systematic.
- [ ] Verify against ADS every bib entry typed from memory: Hog2000, Monet2003, Bramich2008, Munn2004, Tian2017;
      Bergeron1995, Bedard2020, GentileFusillo2021, Harris2006, DESI2016, Luyten1922, SalimGould2002, LepineShara2005,
      Oppenheimer2001, Kilic2006, Munn2017, Smith2009; DESI_DR1, Koposov2024, Kaczmarczik2009.
- [ ] Remaining \todo's: SDSS depth, J depths, DR10 point-source depth; \dstn: LSST DM pipeline paper on PMs.
- [ ] Consistency pass over the older sections (Introduction, Existing Data, Data Catalogs).
- [ ] Nada: read the whole draft.

### 4. Questions for Dustin
- [ ] Source of (g-i) = -0.56 + 1.34 (BP-RP) (paper \todo).
- [ ] Which Gaia release / reference epoch the CP astrometry used per survey (could explain the -0.20 / +0.12 mas/yr
      zero-point drift; paper \todo).
- [ ] Missing bricks (`missing_bricks.txt`: all of 058-059 + 557 gaps in 000-057); CFS space (above); data release.

### 5. Housekeeping
- [ ] Commit outstanding work (James's checks, paper, this file).
- [ ] Delete test output: `/pscratch/sd/n/nelfalou/healthcheck/`, scratch texbuild dirs (Claude's rm is blocked).
- [ ] Low priority: MongoDB login for FireWorks (superseded by slurm/).

### 6. After this paper
- [ ] RA 064-359: needs Dustin's forced photometry (inputs stop at 069). Two-pass procedure in RUNBOOK section 5
      (tested on 064-069). Recheck pulls (error model fitted on slices 000/040) and the maps on new Dec / airmass.
- [ ] High-PM / non-Gaia candidates; science extensions (`EXTENSIONS_BRIEF.md`, `application/`).

## Decisions (Nada unless noted)
- 2026-10-12: Gaia-star PMs made absolute (Gaia motion added back); quality flagged, not cut (pm_quality bitmask).
- 2026-10-23: two-line DCR model (James); flag bad epochs, keep them in the fit; BAD_EPOCHS when >= 10% of epochs are
  > 5 sigma; g-i fallback from g-z only; Dec-ivar bug handled by DDIVFIX > INPDATE > data.
- 2026-10-26: error model v3 (isotropic + along-refraction terms); exposure zero points adopted; no PM-level floor --
  report the pull widths instead.
- 2026-10-29: g-band DCR metallicity dependence documented as a limitation, not corrected (no metallicity for most stars).

## Done (summary; details in git history)
- Pipeline: DCR two-line model (`dcr_model_twoline.json`, held-out cell wRMS g 1.25, r 0.55, i 0.42, z 0.43 mas); CCD
  maps rebuilt from 294M Gaia epochs (`ringmaps-dr10.fits`, 16-px bins, remove 6-8 mas per axis; 2024 maps had added
  9 mas in g); smooth lateral-colour model (`lateral_model_poly3.json`, c0 = 1.469); exposure-CCD zero points (3.15M
  CCD-exposures, jitter 2.2 / 4.2 mas RA / Dec, frame drift -0.20 / +0.12 mas/yr removed); error model v3
  (`error_model.json`); measured epochs only; objid clashes keyed on (brickid, objid); fitsio rewrite of --redo-errors
  (memory leak fixed); new-sky two-pass zero points (ZPFRAC card, LEGACYPM_ZP_TABLE, redo_errors.sbatch).
- Input bug: forced-photometry Dec ivars too large by 1/cos^2(Dec) before legacypipe 7aece157; corrected in the
  lateral step (DECFIX / DECBASIS / DECRATIO headers). Faint-star pulls 0.90-1.02 in both axes afterwards.
- Investigations for James / Dustin:
  * H1 outliers in ours vs Gaia: explained by identifiable conditions (short baseline, saturation, bright mask, few
    epochs, no colour, unmeasured epochs) -> pm_quality flags; clean |dPM| > 30 now 0.10%.
  * H2 per-stage plots measure different things (not a bug). H3 error budget (tab:error_budget, fig:error_floor).
  * H4 Dec > RA errors: extra scatter along the refraction direction, mostly N-S from CTIO -> the along term of v3.
  * H5 reference colour definition checked (median BP-RP of CP calibrators per CCD-exposure).
  * H6 colour modes are two field populations; kinematically hot and cold stars share the DCR curve.
  * H7 (2026-10-29) DCR vs DESI DR1 spectra (`run/dcr-spectra.ipynb`; 127,773 stars, 3.75M epochs): blue mode = turnoff
    dwarfs incl. a halo component ([Fe/H] ~ -1.5), red mode = K/M dwarfs ([Fe/H] ~ -0.3). r/i/z: no metallicity or
    log g dependence at fixed colour. g, blue stars: after DCR, [Fe/H] > -0.5 left with +6..+8 mas/tan z, < -1.2 with
    -1..-2 (~20% of the DCR there). In the paper (sec:dcr).
  * H8 (2026-10-29) CCD-map medians (`run/ringmap-residuals.ipynb`): per-bin medians of held-out bricks 7-12 mas rms
    before, 3.2-5.7 after = noise within 1 mas, median 0.00 +- 0.04; 2024 maps leave 20-25 mas. In the paper (sec:ccd).
- Paper: methods rewritten for the current pipeline (DCR, CCD maps, lateral colour, zero points, uncertainties, flags,
  columns); results (corrections, catalogue, Gaia comparison, uncertainties, parallaxes); application section
  (reduced proper motion diagram beyond Gaia, white-dwarf candidates; `run/rpm-diagram.ipynb`, `application/`);
  all \figtodo's resolved; Dustin's comments resolved except the abstract and the LSST DM paper.
- Docs: RUNBOOK.md (rewritten 2026-10-29), README.md, CLAUDE.md current; new-sky health check on 064-069 (no failures
  once the two-pass zero points are used: ZPFRAC median 0.9996).

## Data notes
- Inputs for 000-063: 81,491 of 84,358 DR10 bricks (Dustin pscratch `forced-motions-dr10/forced-brick/` for 020-023,
  051-069; Dustin CFS for 000-057; our corrected archive for 548 bricks of 060-063). Missing: 058-059 and 557 gaps
  (`missing_bricks.txt`). Nothing exists beyond slice 069.
- Imaging date range of the archive sample: 2013-11-09 to 2021-03-26 (re-measure for the full sample for the paper).
- Calibration-model fits (ringmap_build, dcr-polyfit, colour-fallback, lateral-model-dr10) read the v1 combined files
  (`pm_analysis.V1_COMBINED_DIR`) so the adopted models stay reproducible; everything else reads v2.
- Old maps kept for reference on CFS `ls-motions/{rm,lm}-corrected-forced-motions/` (ringmaps.fits Apr 2024,
  lateralmaps.fits); not used.

## Meetings
- James: Tuesdays 1:00 pm (reminder Tuesday morning). Dustin: Thursdays; Slack updates in between.
