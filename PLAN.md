# LegacyPM progress tracker

Last updated: 2026-10-29. Status keys: [x] done · [~] in progress · [ ] todo · [!] blocked

## CURRENT PLAN -- from the meetings with Dustin (2026-10-22) and James (2026-10-2x), in priority order
Goal: submit the paper in ~2 weeks (Nada). James: two issues MUST be understood before submission (H1, H2).
Ordering principle for the pipeline: every correction is built on the previous stage's output (DCR -> CCD maps ->
lateral -> PM), so settle DCR first, rebuild the maps on top of it, then do ONE full rerun of 000-063 with all fixes.
Steps A-C and H run on the combined files (no pipeline rerun needed).

### H. Must-fix before submission (James) -- can start now, on the current 000-063 output
- [~] H1. The vertical feature in F7 (our PM large where Gaia's is ~0). Started 2026-10-23: py_files/pm_outliers.py;
      slice 000, 281k Gaia stars in the F7 selection, outlier = |ours - Gaia| > 30 mas/yr (4.2%).
      TopCat file: combined/h1_gaia_stars_000.fits (all 281k stars, 74 columns: catalogue properties, per-star epoch
      summaries -- chi_max, n_chi_gt5, resid_rms_mas, frac_edge, frac_dqmask, airmass, n_bands, n_years, max shift of
      each correction -- refits, pulls, flags is_outlier / is_outlier_after_fix / is_pull_outlier / flag_*).
      FINDINGS:
      * Outliers are explained by identifiable conditions. Clean stars (G > 16, no bright/saturation maskbits, g-i
        colour available, >= 10 epochs, baseline >= 2 yr; 55% of stars): outlier rate 0.13%. Rates after the fixes
        below: baseline < 2 yr 25.7%, saturated (maskbits 2-4) 12.0%, BRIGHT/MEDIUM mask 10.2%, < 10 epochs 9.7%,
        no g-i colour 7.7% (1.4% even when otherwise clean -- these stars get NO DCR or lateral correction).
      * 28.6% of Gaia stars have no g-i colour (no i-band flux) -> no DCR/lateral correction at all. Needs a fallback
        colour (g-r, or BP-RP for Gaia stars) in the DCR model (A2).
      * BUG: 4.9% of epochs have full_fit_dra == 0 with ivar > 0 (no offset measured); the PM fit includes them (the
        correction steps already treat them as invalid), and after the corrections 82% carry a fake non-zero offset.
        For Gaia stars they pull the fit TOWARDS Gaia (dropping them raises outliers 4.1 -> 4.9%: the Gaia validation
        is optimistic); for non-Gaia stars they pull PMs towards 0. Ask Dustin what they are; exclude them (D6).
      * Fix test (refit all stars): drop zero-offset epochs + iterative 5-sigma epoch rejection: outliers 4.22 ->
        3.33% (G 16-19: 3.07 -> 1.96%; G 19-21: 3.33 -> 2.42%); robust sigma of (ours - Gaia) 3.10/3.78 -> 2.81/3.29
        mas/yr (RA/Dec). Bright stars (G < 16) stay ~12%.
      * Worst bricks: a patch at Dec ~ +26 deg with 1-yr baseline and ~9 epochs (40-60% "outliers"). With so little
        time coverage, a fixed 30 mas/yr threshold flags imprecise stars, not wrong ones -> also use pulls:
        |pull| > 5: 16.9k stars, |dpm| > 30: 11.9k, both: 6.2k.
      Internal plots: run/h1-outliers.ipynb -> run/plots/h1_*.png/pdf (TopCat: later, not now).
      NEXT:
      catalogue quality flags (saturation, no colour/no DCR, short time coverage) and a "clean" validation sample.
- [x] H2 (2026-10-23): NOT a bug -- the plots measure different things. run/h2-correction-summary.ipynb ->
      run/plots/h2_components_by_stage.pdf (py_files/dcr_fit.load_stage_sample; slice 000, 4.9M Gaia-star epochs with
      sigma < 10 mas). g band, raw -> DCR: systematic offset along the parallactic direction (RMS of colour x airmass
      cell medians = what F2/F3 show) 17.1 -> 3.7 mas (-78%); scatter along 24.7 -> 15.9 (-36%); scatter across
      13.0 -> 13.0; total 2-D median (= F6) 22.7 -> 17.1 (-24%). DCR removes its own signal; the rest of each epoch's
      error adds in quadrature, so the total drops ~25%. r: 6.1 -> 2.0 systematic, total 16.0 -> 15.1.
      Also: g-band CCD maps worsen BOTH directions (along 15.9 -> 18.6, across 13.0 -> 16.1): adds ~10 mas in
      quadrature -> B1. And after DCR the along-parallactic scatter still exceeds the across scatter (g 15.9 vs 13.0,
      r 13.6 vs 11.9, i 13.6 vs 12.3): an extra ~6-9 mas along the parallactic direction not removed by a colour x
      airmass model (turbulence/refraction?) -- likely the Dec pull excess (H4).
- [x] H3 (2026-10-23): run/h3-error-budget.ipynb -> writeup/tables/error_budget.tex (\input-able, label tab:error_budget)
      and writeup/plots/error_floor_vs_mag.pdf. 25M epochs from a random 5% of bricks in all 64 slices (cache
      combined/stage_sample_000-063_0.05.npz), clean Gaia stars, sigma < 10 mas. Per-axis scatter (mas), none / DCR /
      CCD maps / lateral: g 21.4 / 15.1 / 17.8 / 17.1; r 14.1 / 13.2 / 12.5 / 12.4; i 12.8 / 12.7 / 11.6 / 11.6;
      z 11.3 / 11.3 / 10.7 / 10.7. Removed (quadrature): DCR g 15.2, r 5.0, i 1.7; CCD maps r 4.4, i 5.1, z 3.5 but
      g -9.3 (adds; B1); lateral g 4.7, r 1.2. DCR signal g 18.7 -> 4.2 mas.
      Floor vs G (clean): per-axis robust scatter ~11.5 mas at G 16-18 rising to ~28 at G 21; implied systematic
      (scatter^2 - stat^2) 15 mas at G 16-18, 17-22 at G 19-20, 25-35 at G > 20.5 -> the statistical errors of faint
      stars are also underestimated. Flagged stars: up to 33 mas at G 14 (saturation).
      NOTE: per-epoch ROBUST scatter (~12 mas) is below the adopted 20 mas, yet sigma_sys = 20 matches the PM-level
      scatter (F13) -> epoch errors have heavy tails (a least-squares fit feels the RMS, not the MAD). Consider a
      magnitude-dependent floor and/or epoch rejection (D4) when finalising the error model.
- [x] H4 (2026-10-23): run/h4-ra-dec.ipynb -> run/plots/h4_geometry.pdf, h4_along_across_vs_airmass.pdf,
      h4_ra_dec_vs_dec.pdf (slice 000, clean Gaia stars, sigma < 10 mas, fully corrected). EXPLAINED: per-epoch scatter
      is larger ALONG the parallactic direction than across it in every band (excess ~6-10 mas in g, ~5-6 in r/i, 2-5 in
      z, rising to ~10 at tan z > 1), and the parallactic direction is mostly north-south (median |cos PA| 0.79), so the
      excess lands in Dec: Dec/RA per-epoch scatter ratio 1.20 = Dec/RA PM pull ratio 1.30/1.08 = 1.20.
      The excess is not pure DCR (it is present near the zenith in g/r and doesn't scale as c_rel tan z); candidates:
      image elongation from refraction across the bandpass (centroid noise along the elongation), residual DCR from
      colour/SED errors. Options: (a) re-measure after the new DCR model (A2) and rebuilt g maps (B1); (b) an
      anisotropic error model: extra variance along the parallactic direction per band, a_b + k_b tan z (needs the
      parallactic angle per epoch in the PM step -- available from ra, dec, mjd, exptime as in brick_DCR).
- [x] H5 (2026-10-23): reference colour checked (slice 000).
      * Definition (legacypipe legacyzpts/legacy_zeropoints.py l.1500-1506, 1632): ccdbprp = MEDIAN Gaia BP-RP of the
        astrometric calibration stars on that CCD in that exposure; brick_DCR converts it with -0.56 + 1.34 (BP-RP).
      * Conversion verified: our 118k Gaia stars (G 16-20, unsaturated) give g-i = -0.57 + 1.36 (BP-RP); within
        +-0.08 mag per bin up to BP-RP 2.5; too red by 0.15-0.27 mag for BP-RP > 2.5 (M dwarfs).
      * ccdbprp matches the median BP-RP of our Gaia stars per CCD-exposure (58,518 of them): +0.01, scatter 0.22.
      * CORRECTION: there is no -0.8 mag offset. Median c_rel (sigma < 10 mas epochs) is -0.27; the -0.8 was the blue
        PEAK of the bimodal colour distribution, misread as an offset.
      * Hypothesis for the DCR constant terms in r/i (~-2 mas / tan z): the astrometric solution zeroes the calibrators'
        MEAN offset (~mean colour), but the reference is their MEDIAN colour; with a red tail the mean is redder
        (r zero point ~0.27 mag redder than the median). Not seen in g. Absorbed by a constant term in the new model.
      * Still open: source/citation of -0.56 + 1.34 (not in legacypipe) -> Dustin; the paper can also cite our fit.
- [x] H6 (2026-10-23): run/h6-populations.ipynb -> run/plots/h6_cmd_colour_modes.pdf, h6_reduced_pm.pdf,
      h6_dcr_by_population.pdf (189k clean Gaia stars, slices 000/020/040/060, 30% of bricks).
      * The two colour modes are the classic two field populations at high latitude: a blue peak at BP-RP ~ 0.8-1.0
        (main-sequence turnoff stars, thick disk + halo) and a red peak at BP-RP ~ 2-2.5 (K/M dwarfs, thin disk).
        Bright stars (G < 17.5) are mostly turnoff stars; at G 19-21 the M dwarfs dominate and the blue peak moves
        bluer (more halo). At higher |b| the blue peak is bluer too. (Our sample is almost all |b| > 30.)
      * Clean calibration stars stop at G ~ 16 (brighter ones are flagged BRIGHT/SATURATED).
      * DCR at fixed colour is the same for kinematically hot (halo-like, high reduced PM) and cold stars in r, i, z;
        in g the hot half is lower by ~1-2.5 mas per unit tan z at -0.7 < c_rel < 0.5 (small: a few % of the DCR
        signal; consistent with metal-poor/alpha-enhanced SEDs). Both groups follow the same kinked curve -> the
        two-line DCR shape is a colour effect, not a mix of populations. A colour-only DCR model is adequate.
      * Small artefact: a column at BP-RP = 0 (missing Gaia BP/RP stored as 0) -- exclude in any colour-based use.

### A. DCR model (blocks B-E; decision with Dustin)
- [x] A1 (2026-10-22): run/dcr-polyfit.ipynb (py_files/dcr_fit.py; geometry in py_files/dcr_geometry.py so analysis
      needn't load brick_DCR's 14 GB table). 80.7M Gaia-star epochs (sigma < 10 mas) from a random 25% of bricks in
      all 64 slices (cached: combined/dcr_sample_000-063_0.25.npz); fit on half the bricks, scored on the other half.
      Figures for Dustin in run/plots/: dcr_polyfit.pdf, dcr_curvature_g_r.pdf, dcr_tanz_polyfit.pdf.
      * Polynomials in x = c_rel tan z: held-out RMS falls with order in g (2.5 -> 0.36 mas binned in x), but don't help
        r/i at all once tested in (c_rel, tan z) cells, and high orders blow up in sparse corners.
      * The curvature is in COLOUR, not x: offset / tan z vs c_rel is the same curve at all airmasses
        (dcr_curvature_g_r.pdf). Model DCR = tan z x P(c_rel) beats polynomials in x in every band. Held-out wRMS (mas),
        linear-x (current) / x order 5 / tan z x order 2 / tan z x order 4:
        g 2.57 / 1.24 / 1.53 / 1.15;  r 1.18 / 1.15 / 0.63 / 0.60;  i 0.78 / 0.76 / 0.46 / 0.41;  z 0.51 / 0.49 / 0.43 / 0.42.
      * Data flatten at the colour extremes (c_rel < ~-1.5, > ~2.2), which no polynomial follows.
      * PROPOSAL for Dustin: DCR = tan z x quadratic(c_rel), c_rel clamped to ~[-1.5, 2.2]; z can stay linear.
        Quadratic coefficients (mas per unit tan z, c^2, c, const; 5% sample): g -4.16 32.39 0.33; r 1.08 8.73 -2.33;
        i 0.61 4.39 -1.92; z (linear) 1.38 -0.45. The constant terms in r/i (~-2 mas/tan z) mean stars at the
        reference colour still shift with airmass: equivalent to the (g-i)_ref transformation being ~0.3-0.4 mag off.
- [x] A1b (2026-10-22, James's suggestion: two linear fits / "two modes"): run/dcr-polyfit.ipynb section "Two linear
      fits"; figures run/plots/dcr_twolines.pdf, dcr_two_modes.pdf. The two modes are COLOUR populations (blue turnoff vs
      red K/M dwarfs; c_rel is bimodal): g slope 40.6 for c_rel < 0.1 vs 26.1 above; splitting by airmass gives no
      difference (30.9 vs 32.5). Held-out wRMS in (c_rel, tan z) cells, tan z x broken line in colour: g 1.25, r 0.55,
      i 0.42, z 0.43 mas -- best or joint-best in every band (beats tan z x quadratic in r/i; ties two-lines-in-x and
      x order 5 in g). Breaks / slopes (blue, red; mas per mag per tan z): g 0.10 / 39.5, 24.6; r 0.75 / 8.0, 13.2;
      i 0.50 / 3.5, 6.3; z 0.40 / 1.2, 1.6. NEW PROPOSAL: DCR = tan z x broken line in c_rel, c_rel clamped ~-1.5.
- [x] A2 (2026-10-23, Nada: James's two lines). Implemented: run/dcr-polyfit.ipynb last cell fits all 000-063 cells
      -> py_files/dcr_model_twoline.json; brick_DCR.dcr_shift_mas applies it (c_rel clipped to [-1.5, 2.5]). Final
      breaks / a / slopes (blue, red): g 0.05 / 3.5 / 39.7, 25.1; r 0.70 / 3.3 / 7.9, 13.1; i 0.50 / -0.1 / 3.4, 6.3;
      z -0.10 / -0.8 / 0.9, 1.5 mas per tan z. i and z now get a real correction (old K = 1).
      Was: Decide the model with Dustin (send run/plots/dcr_tanz_polyfit.pdf, dcr_twolines.pdf, dcr_two_modes.pdf):
      current proposal tan z x broken line in c_rel (James's two modes), clamped at c_rel ~ -1.5. Then update brick_DCR
      (correction = tan z x f_b(c_rel) instead of K_b c_rel tan z) and dcr_fit for the coefficients.
- [x] A3 (code 2026-10-23; rerun dcr-plots after the rerun). Remake F3 for the paper with the chosen model (data + fit only; done for the linear fit).
- [x] A4 (2026-10-23) Pulls vs magnitude (run/pull-vs-mag.ipynb, py_files/pm_pulls.py; slice 000, 25% of bricks, 23M
      epochs of clean objects, residuals from each star's straight-line fit). **BUG FOUND in the inputs:
      full_fit_ddec_ivar is too large by 1/cos^2(Dec)** (sigma_Dec/sigma_RA = cos Dec exactly: 0.71 at -45, 0.43 at -65,
      0.13 at -82; faint-star Dec pulls 1.36 / 2.4 / ~7 there, RA fine). With sigma_Dec divided by cos Dec, faint pulls
      are 0.90-1.03 in both axes at every Dec, so the statistical errors are right (k = 1; Nada's constant-sigma_sys
      argument holds). Core best-fit sigma_sys: g 16, r 12, i 12, z 10 mas (robust; 20 also covers the tails).
      Left: Dec pulls ~10-20% above RA for bright stars (the H4 parallactic excess), and |pull| > 5 rising to ~5% of
      epochs at mag 25-26. Consequences: pmdec / parallax ivars too large south of ~-30 (likely much of the F8 pmdec pull
      1.40 vs pmra 1.10 and of H4's PM pull ratio); BAD_EPOCHS over-fires in Dec there. TODO: confirm with Dustin, then
      fix (ddec ivar x cos^2 Dec) in error_model / brick_lateralmaps before the rerun.
- [x] A5 (2026-10-23, Nada: adopt per-band sigma_sys) error_model.SIGMA_SYS_MAS = g 16, r 12, i 12, z 10 (A4 fit).
      Headers SIGSYS_G..Z. pm-error-model / h3 / pull-vs-mag notebooks updated.
- [x] A6 (2026-10-23, Nada: fallback colours if accurate) colours.py + run/colour-fallback.ipynb: g-i from g-z,
      0.025 mag scatter, |bias| < 0.025 for 0 < g-i < 3.2 (g-r, r-z, BP-RP rejected: 0.12-0.17 mag, biases to 0.27).
      14% of PSF objects lack g-i (80% of them: no i coverage); g-z covers 81% of those. New column pm_colour_source.
      Also found: brick_DCR skipped DCR entirely for bricks without i band -- removed (no-i test brick 0010m295: DCR
      now on 86% of epochs, NO_COLOUR 100% -> 14%).
- [x] A7 (2026-10-23, Nada: adopt >= 10%) BAD_EPOCHS rule: 1-7% of all epochs are > 5 sigma, spread evenly over exposures (no bad exposures), so "any
      bad epoch" mostly counts epochs (40% flagged at 23 epochs, 80% at 55). Slice 000 clean Gaia stars: any >= 1 flags
      37% and catches 95% of |ours - Gaia| > 30 mas/yr; fraction of bad epochs >= 10% flags 4.7% and catches 89%.
      Proposed: flag when >= 10% of epochs are bad. Nada to decide.
- [x] A8 (2026-10-23) Dec-ivar bug handling. Dustin: fixed in legacypipe 7aece157 (2026-09-16; only ivar, positions
      fine); no marker in old files; he is adding header card DDIVFIX = T and patching his pscratch files. Survey of 1627
      input bricks of 000-063: CFS files (2024-12 .. 2025-01) all affected, scratch files (>= 2026-09-17) all fixed.
      A per-brick data test can't be the rule (real PSF elongation along Dec raises sigma_Dec/sigma_RA by 5-8% in the
      north, comparable to 1 - cos Dec at |Dec| < 35). Rule: DDIVFIX if present, else input date (INPDATE header,
      stamped by brick_DCR / prep_archive_rerun), else data. Tested on 0600m045 (scratch, not corrected) and 0010m295
      (CFS, corrected). 2026-10-23: Dustin patched all his pscratch inputs with DDIVFIX = T (boolean, in the PRIMARY
      header); brick_DCR copies it to the table header; verified on 0600m045 (DECBASIS = DDIVFIX). Pulls on slice 000 with it: faint widths 0.90-1.02 in RA and Dec at all Dec.
- [x] A9 (2026-10-26) Error model v3 (Nada: refit sigma_sys + term along refraction): run/error-model-fit.ipynb, per epoch
      C = C_stat + (s_b + m_b tan z)^2 I + (a_b + k_b tan z)^2 p p^T; PM fit whitens each epoch's RA/Dec pair
      (create_brick_catalogue._wls); run_bricks.py --redo-errors (errors + zero points + PM only, restartable via
      ERRMODEL/ZPTABLE headers). INCIDENT: first redo test wrote headers listing a column the data lacked for 1455
      slice-000 forced files; repaired (legacypm-catalogue-v2/repair/repair_header.py, positions verified bit-for-bit);
      fixed (tables rebuilt via from_columns, rewrite read back before replace).
      Slice-000 test showed PM pulls 1.09/1.28 with v3: per-epoch errors right, but a FIELD-COHERENT PM error remained
      (per-brick 0.47/0.59 mas/yr beyond noise; mean offset -0.13/+0.31; extra x baseline ~ const).
- [x] A10 (2026-10-26, Nada: go ahead) Exposure-CCD zero points (py_files/zeropoints.py; table on CFS
      ls-motions/zeropoints/zeropoints-dr10-v2.fits: 3,153,093 exposure-CCDs, 58,445 exposures, 290.5M Gaia epochs,
      median 74 stars; own ZP 98.9%, exposure fallback 0.9%, none 0.2%). Jitter beyond noise RA 2.2 / Dec 4.2 mas
      (g 3.7/5.2 .. z 1.5/3.5), mostly CCD-to-CCD; time drift -0.20 / +0.12 mas/yr. Cross-validated (other half's ZPs),
      slices 000+040, G 16-19.5: mean offset -0.23/+0.26 -> +0.03/+0.01; scatter 1.56/1.81 -> 1.52/1.64; brick-coherent
      0.47/0.55 -> 0.36/0.32 mas/yr. Pipeline: zp_full_fit_* columns (final update step), PM fit uses them. Paper:
      sec:zeropoints added. Error model refit on ZP-corrected residuals and adopted (ERRMODEL 5121325b; iso g 6.5+1.5tz,
      r 5.5+2.0tz, i 5.5+1.5tz, z 5.5+2.0tz; along g 1.0+6.0tz, r 6.5tz, i 1.0+3.5tz, z 5.0tz mas). Slice-000 test, clean
      Gaia stars: mean offset +0.06/+0.01, scatter 1.57/1.73 mas/yr (v2 1.72/2.01), PM pulls 1.13/1.24 (bright 1.26/1.36),
      parallax pull 1.11. A PM-level floor (s/T) would fix the pulls; Nada 2026-10-26: not worth it -> report the
      measured pull widths in the paper instead. Full redo of 000-063: first attempts ran OUT OF MEMORY (1 job then 4
      jobs, ~4000 bricks each; 17,139 + slice 000 done, nothing corrupted). Cause: update_total_errors built astropy HDU
      objects, which leaked ~ the forced-file size per brick (+1.1 GB over 12 bricks). 2026-10-27: rewritten with fitsio +
      numpy (total_error_values / zeropoints.zp_values shared with the astropy path); verified identical columns, units
      and headers (primary DDIVFIX kept), +43 MB over 12 bricks. Resubmitted as 4 jobs (64 workers): all 81,491
      catalogues on ERRMODEL 5121325b + ZPTABLE 93045419720 (header-checked). Combined 2026-10-27: 266,019,817 PSF with PMs.
      FULL-SAMPLE Gaia check (10.7M clean Gaia stars): mean -0.008/+0.003 mas/yr; scatter 1.68/1.79 (G 17-19 1.42/1.53);
      PM pulls 1.20/1.28 (G 14-17 1.30/1.42 .. G 20-21 1.13/1.18), ~flat in Dec; parallax offset -0.05 mas (was -0.3),
      parallax pull 1.17; |dPM| > 30: 0.10%. Final flags (Gaia stars, |dPM|>30): clean 0.10%, FEW_EPOCHS 41%, SHORT 58%,
      NO_COLOUR 57%, BRIGHT 20%, SAT 18%, BAD_EPOCHS 29%; BAD_EPOCHS flags 2.1% of otherwise-clean, catches 59%.
      Unmeasured epochs 7.3%. Figures regenerated (err-plots, pm-error-model); h3 rerun pending (zp-stage pixel sign bug
      in dcr_fit fixed 2026-10-28). Paper: uncertainties rewritten (eq:errors, tab:errmodel, PM pulls reported, no
      rescaling), flags/ZP numbers filled. 2026-10-28: h3 rerun (zp stage removes 4.0-6.1 mas; final per-axis g 9.9,
      r 9.2, i 8.4, z 8.2; floor ~10 mas at G 16-18.5); err-plots F6 shows the zp stage; paper sec:corr_analysis rewritten
      around tab:error_budget (now \input, table*), fig:scatter_vs_mag included. 2026-10-29: last \figtodo's resolved
      (fig:lateral = lateral_colour_bins.png, fig:dcr = dcr_before_after.png; captions corrected); stale Results paragraph
      removed. NEW Results subsections: corrections, catalogue (fig:footprint), comparison with Gaia (fig:pm_vs_gaia),
      uncertainties (fig:pulls, fig:pm_errors), parallaxes (fig:parallax); fig:error_floor in the uncertainties section;
      fig:pos_err in the Gaia subsection (resolves Dustin's value-added comment). pm_precision_vs_mag.pdf not used
      (redundant with pm_errors). Headline numbers (diag/results_numbers.py): 266.0M PSF with PMs, 202.2M clean (76%),
      12.2M clean Gaia; median 32 epochs / 4.8 yr; r > 21 with sigma < 2/5/10 mas/yr: 2.0M/8.5M/17.0M. Paper 17 pp, clean
      build. Remaining: copy v2 to CFS; commit. Then:
      combine, re-check PM pulls, figures. Ask Dustin: Gaia release / epoch used by CP astrometry per survey (drift).
### B. Rebuild the CCD (tree-ring) maps from the full sample (needs A2)
- [x] B1 (2026-10-23) py_files/ringmap_build.py on a compute node (35 min): 294M Gaia-star epochs of 000-063
      (sigma < 10 mas, with colour, not bright/saturated), new DCR removed. Held-out bricks (20%), per-axis scatter g/r/i/z:
      none 14.60/13.11/12.37/11.33; 2024 maps 17.14/12.31/11.27/10.82; new 16 px 12.41/10.82/9.87/9.56 (best; 28-45
      epochs per bin); 32 px 12.67/11.17/10.20/9.73; 64 px 13.22/11.77/10.78/10.06. Adopted 16 px ->
      CFS ls-motions/rm-corrected-forced-motions/ringmaps-dr10.fits (2024 file kept); brick_ringmaps.RINGMAPS_PATH points
      at it and infers the bin size from the map shape. run/ringmap-build.ipynb; F4a/F4b regenerated.
      Was: Recompute DCR on the combined Gaia-star epochs with the new model; build ringmaps from ALL 000-063 Gaia
      stars (sigma < 10 mas), replacing the Apr 2024 maps (~60 stars/bin then; far more now). Expected to fix the
      g-band problem (CCD maps currently make g worse: 15.6 -> 18.3 mas). (James: the ring-map section/figures are fine.)
- [x] B2 (2026-10-23, with B1). Check per band that the new maps improve scatter; regenerate F4a (one CCD) and F4b (whole focal plane)
      from the new maps (run/ringmap-plots.ipynb: point RINGMAPS_PATH at the new file).
### C. Lateral colour (needs B1) -- James: the correction looks right
- [x] C1 (2026-10-23) run/lateral-model-dr10.ipynb (compute node): 43.8M epochs from 12,021 bricks (15% of the B1
      sample) after the new DCR + ringmaps-dr10. Held-out per-axis scatter g/r/i/z: none 12.47/10.93/9.73/9.27, old
      coefficients 11.97/10.91/9.88/9.27 (made i worse), refit with c0 = 1.469 (median g-i of the map stars)
      11.84/10.88/9.71/9.26 -- adopted (py_files/lateral_model_poly3.json). Per-band c0 (g 0.92 .. z 1.69) tested on a
      4% subset: no gain (g 11.89 vs 11.87). Lateral colour matters mainly in g (rms applied shift 6.3 mas; r 1.6, i 0.9,
      z 0.4). Was: Refit the smooth lateral model on the new CCD-map output (also re-check the pivot c0: data median 1.77).
- [ ] C2. Investigate the colour-independent left/right dipole left in F5 (not lateral colour; CCD maps / WCS?).
### D. Pipeline fixes to include in the rerun
- [x] D1 (2026-10-23) done: brick_DCR colour lookup and create_brick_catalogue grouping + row indexing on
      (brickid, objid); unmatched objects get no colour (was g-i = 0); tractor-pm header OBJKEY, and combine sets
      objid_clash only for older outputs. Test 0201m135 (69 clashing PSF objects): epoch counts = own epochs for every
      PSF row; all 2067 measured epochs of clashing objects get their own colour; 0600m045 (no clashes) identical.
      Was: objid clashes: key brick_DCR's colour lookup and create_brick_catalogue's row grouping on (brickid, objid).
- [x] D2 (2026-10-23). prep_archive_rerun drops dcr_*/dp1/colour_airmass too; run_bricks redoes DCR on archive
      bricks. Was: Archive bricks (548): raw full_fit_* columns survive, so prep_archive_rerun must also drop dcr_* columns and
      the run must redo DCR (currently it starts from the archived DCR columns = old amplitudes).
- [x] D3. Done with D6.
- [x] D4 (2026-10-23, Nada: flag, don't reject). Epochs > 5 sigma from the fit (5-param; 4-param if it failed) are
      kept; pm_n_bad_epochs counts them and pm_quality bit 64 BAD_EPOCHS is set, so clean_mask drops them from plots.
      (H1: rejecting them instead would take outliers 4.2 -> 3.3% on slice 000.)
- [x] D6 (2026-10-23, Nada: don't use what the inputs don't trust): create_brick_catalogue now fits only measured
      epochs (ivar > 0 and full_fit_dra != 0); also covers D3 (epoch/baseline cuts count measured epochs). Tested on
      0001m002: only stars with unmeasured epochs change (median 0.03-0.05 sigma); Gaia RA scatter 1.81 -> 2.12
      mas/yr there (the fake epochs had pulled Gaia stars towards Gaia). Takes effect in the rerun.
- [ ] D7. From H1: catalogue quality flags (saturated / bright mask, no g-i colour = no DCR, short time coverage).
- [ ] D5. Magnitude/band-dependent error floor (pull width > 2 at G < 15.5), from H3.
- [x] D8 (2026-10-23, Nada: flag, don't cut): create_brick_catalogue computes a PM for every PSF object with >= 3
      measured epochs and a non-zero baseline; quality goes in the new bitmask pm_quality (1 FEW_EPOCHS < 10,
      2 SHORT_BASELINE < 730 d, 4 NO_COLOUR (no DCR/lateral correction), 8 BRIGHT (maskbits BRIGHT|MEDIUM),
      16 SATURATED (SATUR_G/R/Z/I), 32 FIT_FAILED), plus pm_n_epochs and pm_baseline_days. pm_flag unchanged
      (1 both fits OK, 0 no fit, -1 not PSF). Analysis/figures use clean = pm_quality == 0
      (pm_analysis.clean_mask / gaia_star_mask(clean=True) -- rebuilt from columns for pre-rerun output; 53% of Gaia
      stars in slice 000). combine_catalogue carries the new columns. Tested on 0001m002 (83% of PSF objects clean).
- [x] D9 (2026-10-23, Nada): realistic per-epoch error sigma_total = sqrt(sigma_stat^2 + sigma_sys^2), sigma_sys = 20 mas,
      in py_files/error_model.py (replaces the 14.8 mas floor). The lateral step writes full_fit_{dra,ddec}_ivar_total
      (+ header SIGSYS) to the corrected forced files; the PM fit reads those columns (falls back to computing them for
      older files) and writes SIGSYS into each catalogue header. Check (run/pm-error-model.ipynb): realistic PM errors match
      the scatter of ours - Gaia for bright clean Gaia stars (~2.2 mas/yr), slightly conservative in RA.
### E. Full rerun of 000-063 (needs A2, B1, C1, D1-D2 (+D4/D5); ~20-25 node-hours, 4 jobs) -> combine -> rsync to CFS
- [x] E (2026-10-25) v2 run: 4 jobs (3.5-5 h each), 81,491/81,491 bricks ok, 0 failed (v1: 195 failed) ->
      /pscratch/sd/n/nelfalou/legacypm-catalogue-v2/. Combined (v2/combined/, 169 GB): 266,021,111 PSF objects with PMs,
      24,184,320 Gaia stars. pm_analysis.COMBINED_DIR -> v2 (V1_COMBINED_DIR kept for the calibration fits: ringmap_build,
      dcr-polyfit, colour-fallback, lateral-model-dr10 stay on v1 so the adopted models are reproducible).
      Spot check (170 bricks): all columns/cards present; DECFIX by source correct (CFS/archive INPDATE True, scratch
      DDIVFIX False); clean 74%; flags FEW 11.1, SHORT 9.6, BAD_EPOCHS 9.5, BRIGHT 5.0, NO_COLOUR 4.4, FIT_FAILED 4.4,
      SAT 1.0%; colour g-i 87.6 / g-z 7.9 / none 4.4%.
      First look, slice 000 clean Gaia stars, ours - Gaia (4-param), v1 -> v2: robust scatter RA 2.40 -> 1.72, Dec 2.87
      -> 2.01 mas/yr; |dPM| > 30: 1.00% -> 0.09% (0.18% ignoring BAD_EPOCHS); |pull| > 5: 3.67 -> 0.66%; pull width
      RA 1.05 -> 0.91, Dec 1.31 -> 1.08. Dec/RA pulls by Dec: ~1.0-1.1/0.9 near Dec -45..+15, rising to 1.22 at -80 and
      1.40 at +15..+30 (far from CTIO zenith, high airmass) -> the parallactic-direction excess (decision B) is real and
      airmass-driven; RA pulls < 1 suggest sigma_sys slightly high for RA. TODO: copy v2 to CFS; rerun figure notebooks.
### F. Figures
- [x] F-v2 (2026-10-25) Regenerated on v2 (compute node): err-plots, dcr-plots (F3 now the two-line model), pm-error-model,
      h3-error-budget (table + floor), pull-vs-mag; ringmap figures unchanged (same maps).
      BUG FIXED: err-plots loaded Gaia stars with pma._COLUMNS, which lacks pm_quality/maskbits/fluxes, so clean_mask
      silently fell back to epochs + baseline + pm_flag only (no BAD_EPOCHS/BRIGHT/SATURATED/NO_COLOUR) -- true for ALL
      earlier versions of F7/F8/F9/parallax figures. load_combined now always adds QUALITY_INPUT_COLUMNS. With it
      (slices 000-001): 351,676 clean Gaia stars (was 446,987), PM pull widths RA 0.92 / Dec 1.09 (4-param), |pull| > 5
      0.2 / 0.5% (was 3.1 / 3.4%); parallax pull 0.88, ours - Gaia parallax median -0.3 mas at all G (zero-point offset;
      mention in paper), scatter ~5 mas. F7 stripe gone.
      H3 v2 per-axis scatter none -> all corrections: g 21.45 -> 11.76, r 14.13 -> 10.61, i 12.82 -> 9.71, z 11.27 -> 9.19
      mas (g CCD maps now remove 8.4 mas). Implied systematic ~11.5-12 mas at G 15.5-18.5.
      Pull refit on v2 (slice 000): k = 1 all bands; core sigma_sys g 10, r 8, i 8, z 8 mas (adopted 16/12/12/10 were fit
      on v1) -> RA PM pulls 0.92 consistent with sigma_sys too large; Dec 1.09 = high-airmass parallactic excess
      (decision B). Both need the LM + PM steps rerun (ivar_total is written by the lateral step).
      F11 footprint: whole 000-063 (81,491 bricks), binned median objects/brick, FEW_EPOCHS / SHORT_BASELINE lines.
- [x] Quick plot changes (2026-10-22, tested on 000-001; F2/F3 on 058): F1 median lines (outlined) + 16-84% band,
      faint points; F2 >= 100 epochs per bin, coarser bins; F3 data + fit only; F5 three rows (data before / smooth
      model / data after), >= 50 epochs per bin; F6 one panel per band, valid fits only (full_fit_dra != 0), bins
      >= 500 epochs; F7 dashed 1:1; F8 legend gives |pull| > 5 fraction. Diagnostics in run/plots/:
      pm_vs_gaia_by_mag.png, pull_width_vs_dec_G.pdf.
- FINDINGS from those (2026-10-22):
  * F5: the smooth model reproduces the colour-dependent edge pattern and the residual row removes it. A left/right
    dipole remains in all three colour bins: colour-independent, so not lateral colour (CCD maps / WCS?).
  * F7 stripe: NOT just faint stars (that earlier explanation was wrong). |ours - Gaia| > 30 mas/yr for 2.4-3.2% of
    Gaia stars at G 16-21, 5% at G 15-16, 8.7% at G 14-15 (slices 000-001). Same population as the F8 tails
    (|pull| > 5: 3.3% RA, 4.5% Dec, 3.6% parallax; 0.00006% for a Gaussian). -> NEW TASK: investigate outliers
    (epochs, baseline, bad epochs/blends/saturation; robust (clipped) PM fit?).
  * Pull width vs Dec: Dec pulls grow far from Dec ~ -20..+10 (2.5 at -85, 1.9 at +35); RA much flatter. Consistent
    with an airmass / parallactic-direction effect (DCR or refraction residual).
  * Pull width vs G: both components > 2 for G < 15.5 (2.6-2.8 at G 14.5), ~1.0 (RA) / 1.3 (Dec) for G > 17:
    the 14.8 mas floor is too small for bright stars -> magnitude/band-dependent error floor (error analysis task).
- [x] F13 (2026-10-23): PM uncertainty vs magnitude (run/pm-error-model.ipynb -> writeup/plots/pm_errors_vs_mag.pdf):
      Gaia formal errors, our statistical-only and realistic (sigma_sys = 20) errors, and the robust scatter of
      ours - Gaia; 100 bricks of slice 000. Internal: run/plots/pull_near_zero_gaia.pdf (Gaia stars with |Gaia PM| < 30:
      (ours - Gaia)/sigma width 1.08 RA / 1.30 Dec for clean stars, 1.12 / 1.49 flagged; old 14.8 floor).
- [ ] F12 (NEW, James): completeness / numbers vs magnitude -- stars with PMs from us vs Gaia (with and without a Gaia
      PM), per magnitude bin, to show the added value (fainter than G ~ 21, and Gaia stars lacking PMs).
- [ ] F8: write up how the formal errors are built (per-epoch stat. error + 14.8 mas floor in quadrature -> LSQ
      covariance; Gaia variance added for Gaia stars) and what pulls mean.
- [ ] F9 past G = 21 (Dustin: the value is beyond Gaia's limit) -- needs a non-Gaia reference (e.g. galaxies/QSOs
      should have zero PM) -- design with Dustin.
- [ ] Run the catalogue figures (F7-F11) on all of 000-063 (compute node); regenerate all figures after E;
      re-derive the error floor from F6.
### I. Paper text and science
- [x] I7 (2026-10-23) Build now clean (no LaTeX/BibTeX errors): \niceurl defined, \gtrsim -> \ga, mybib.bib accents ->
      LaTeX escapes, VHS author list repaired. Dustin's comments resolved: keywords; Tycho-2 (Hog2000) + USNO-B1.0
      (Monet2003) paragraph; SDSS reworded (ugriz, 1/3 sky, r depth, Stripe 82 PMs Bramich2008, USNO-B+SDSS Munn2004);
      Pan-STARRS own paragraph (grizy, 3/4 sky, PMs, GPS1 Tian2017); J-band depths; LSST name + refocus (same systematics).
      New bib entries typed from memory -- check against ADS. \todo{verify} left on SDSS r and NIR J depths.
      OPEN \dstn: abstract (last), LSST DM pipeline paper on PMs (literature search), Gaia-vs-LS positional-error plot.
- [x] I6 (2026-10-23) main.tex methods updated to the current pipeline: DCR (two-line model eq:dcr + tab:dcr, g-z fallback,
      fig:dcr_model = dcr_amplitude_fit.pdf -- regenerate), CCD maps (B1 sample, bin-size validation), lateral colour (smooth
      poly model, c0 = 1.469, C1 validation; old binned-map text removed), measurement uncertainties (per-band sigma_sys,
      pull calibration k = 1, heavy tails, Dec-ivar bug + fix). Removed the old Results figure plots/ringmaps.png (missing
      file, duplicate fig:ringmaps label). Still stale: sec:corr_analysis numbers (floor ~22 -> ~15 mas, \todo) and
      fig:scatter_vs_mag / fig:lateral placeholders. Pre-existing build errors left: \% in \href URLs (l.80-101),
      \gtrsim (needs amssymb), bibtex mangles Ivezic/Zeljko accents in mybib.bib l.20.
- [x] I0 (2026-10-23) main.tex sec:flags replaces the old 'Sample selection and quality cuts' (N >= 5 / 180 d cuts that
      no longer exist): measurements used, when no fit is made, pm_quality table (tab:pm_quality) with threshold
      justifications, pm_flag; new sec:columns with tab:columns + header keywords. \todo's left for final-sample numbers
      and the data-release location. objid_clash omitted (v2 has no clashes). Was: Catalogue description: a table/list of all cuts and flags (pm_flag, pm_quality bits with thresholds,
      objid_clash, what is excluded from fits: unmeasured epochs, non-PSF), and possibly of all output columns
      (name, unit, meaning) -- for the paper's data-product section and a README with the catalogue.
- [ ] I1. Application section (James): research what science PM catalogues enable (disk vs halo kinematics,
      hypervelocity stars beyond escape speed, halo structure); compare our PM distribution with Gaia-based catalogues
      and fit models. Plan the section this week.
- [ ] I2. Science justification for PMs at 22-25 mag (James): what specifically do faint PMs enable? (with F12)
- [ ] I3. Note in the paper: the net positive pmra / negative pmdec of Gaia stars in this RA range is Galactic
      structure + solar reflex motion (James), not a systematic.
- [ ] I4. Parallax section: include for completeness with the explanation (Legacy Surveys observes each field at
      about the same time of year, so the parallax ellipse is barely sampled) (Dustin, James).
- [ ] I5. Sections to update: lateral (binned -> smooth model), DCR (new model), sample description, error budget (H3),
      results; captions and slots for F1, F5-F12.
### G. Still open from before
- [ ] Missing bricks to Dustin (missing_bricks.txt: 058-059 + 557 gaps); ask about CFS space for ~3.5 TB of forced
      files; ask for the (g-i) <-> BP-RP citation (with H5).
- [ ] High-PM / non-Gaia candidates (after E; feeds I1).
- [ ] Commit the figure and DCR work (notebooks, plot_style, dcr_fit, dcr_geometry, writeup/plots, run/plots).
- [ ] Low priority: MongoDB login for FireWorks.
### Meetings
- [ ] James: Tuesday 1:00 pm (send a reminder Tuesday morning). Dustin: Thursday, usual time; Slack updates meanwhile.

## Paper (writeup/main.tex)

### Step 1: accuracy fixes + methods text (no new data needed)
- [x] DCR: units of K_b (mas/mag), tan z = sqrt(X²−1), relative colour vs per-CCD reference colour, K_i = K_z
- [x] DCR: replace "averaged in some way... reset the offset" placeholder
- [x] DCR results paragraph: fix colour/wavelength reasoning
- [x] Ring maps: "461 million Gaia stars" -> measurements; state sequential order of corrections
- [x] Lateral colour section written
- [x] "Calculating Proper Motions and Parallaxes" section written (model, errors, reference epoch, cuts, pm_flag)
- [x] Survey table: σ_μ units, Gaia G, DECam σ_μ marked TODO
- [x] "to G~27" depth claims toned down
- [x] Broken `\ref{fig:hist}` replaced by the residual-vs-magnitude figure (placeholder until regenerated)
- [x] Old DRC.png (absolute colour, no colourbar units) commented out; fig:dcr is now a placeholder for the slide-15 before/after
- [x] "Analysis of Corrected Data" drafted from slide-20 numbers (~22 -> ~15 mas floor) -- re-measure after F6 regenerated
- [x] Paper compiles cleanly (see CLAUDE.md for build command); 13 `\todo`s + 4 `\figtodo`s left in text
- [ ] Nada to review all Step-1 edits (marked with `\todo` where a number/fact needs confirming)

### Step 2: figures (all from notebooks in run/, saved to writeup/plots/; shared style py_files/plot_style.py)
All figures have a title and shared axes where panels share them (2026-10-14). Decisions: F1 in mas, r 14-25,
300k-point subsample per class; F2 r band, +-50 mas, bins with >= 15 epochs; F3 sigma < 10 mas epochs, binned medians
(>= 30 per bin), error-weighted line; F5 g band (largest effect), +-15 mas, ~0.05 deg hexes with >= 10 epochs;
F6 g-band epochs only (per-epoch magnitude), bins with >= 100 epochs; F7-F10 Gaia stars with a Gaia 5-parameter
solution and pm_flag == 1, no objid clash; F11 per-brick medians.
Notebooks repointed (2026-10-12) to the combined per-slice files from slurm/combine_catalogue.py (RUNBOOK section 6).
Tested end to end on slice 058 (6 bricks): err-plots and dcr-plots run and write all their figures.
| # | Figure | Notebook / cell | Status |
|---|---|---|---|
| F1 | Positional σ vs mag, Gaia vs LS (Dustin's request) | err-plots cell 5 -> pos_err_vs_mag.png | [ ] code ready |
| F2 | DCR before/after hexbin (replace DRC.png) | dcr-plots cell 11 -> dcr_before_after.png | [ ] code ready; units bug fixed 2026-10-14 |
| F3 | K_b fit per band | dcr-plots cell 13 -> dcr_amplitude_fit.pdf | [ ] code written + tested 2026-10-14 |
| F4a | Tree-ring map, one CCD (N4, z) | run/ringmap-plots.ipynb cell 2 -> ringmap_N4_z.pdf; main.tex fig:ringmaps | [x] 2026-10-14 (2024 maps); remake after B1 |
| F4b | Tree-ring maps, whole focal plane (all 61 CCDs, z; Dustin) | run/ringmap-plots.ipynb cell 3 -> ringmap_focal_plane_z.png; main.tex fig:ringmaps_fp | [x] 2026-10-14 (2024 maps); remake after B1 |
| F5 | Lateral colour focal-plane maps, before/after | err-plots cell 26 -> lateral_colour_bins.png | [ ] code ready (2 rows: before/after LM) |
| F6 | Residual scatter vs mag per correction stage | err-plots cell 8 | [ ] labels fixed 2026-10-12; run on real data |
| F7 | PM residual (ours − Gaia) vs G | err-plots cell 31 | [ ] adapt to hexbin |
| F8 | Pull distributions PM + parallax | err-plots cell 29 | [ ] run (never executed) |
| F9 | PM precision vs mag, extended past G=21 | err-plots cell 32 | [ ] extend |
| F10 | Parallax vs Gaia | err-plots cell after F9 -> parallax_vs_gaia.pdf | [ ] code written + tested 2026-10-14 |
| F11 | Footprint + N_epochs / baseline histograms | err-plots cell 34 -> footprint.png | [ ] code ready |

### Step 3: results & discussion text
- [ ] Sample description (RA 000–063, ~81.5k bricks once the run finishes; counts of stars, PSF, with PM, Gaia/non-Gaia)
- [ ] Analysis of Corrected Data (from F6)
- [ ] Results: PM vs Gaia, uncertainty calibration (pull widths; commit msgs say ~1.5–1.7 PM, ~1.3 plx -- re-measure)
- [ ] Parallax: honest assessment (cadence-limited)
- [ ] Data product: table of output columns
- [ ] Discussion (notes at end of main.tex): empirical maps vs Bernstein parametric; limitations
- [ ] High-PM / non-Gaia candidates (epoch/baseline cuts now applied in the pipeline; needs the 000–063 run)
- [ ] Abstract, conclusion, keywords

## Open questions (for Nada / Dustin) -- answers 2026-10-12
- 14.8 mas floor: not computed in any saved cell; matches the ~15 mas bright-end plateau of F6. Note F6 plots the
  median of the *2-D* offset hypot(dra, ddec); for a 2-D Gaussian that's ≈1.18 σ per axis, so the per-axis floor
  would be ≈12.6 mas. **Nada: redo the error analysis for the paper, but only after more RA slices are processed.**
- K_i = K_z = 1 mas/mag: **Nada: negligible by design; we don't correct those bands' colours. Keep as is.**
  **NEW 2026-10-14 (F3 fit, 485 random bricks of slice 000, 2.7M Gaia-star epochs, sigma < 10 mas, before DCR):**
  K_g = 31.05 +- 0.07 (adopted 29.82), K_r = 8.97 +- 0.04 (7.91), **K_i = 4.47 +- 0.04 (adopted 1)**, K_z = 1.03 +- 0.04 (1).
  So z is right, g and r are 4% and 13% low, and i is 4.5x too small. Uncorrected i-band DCR ~3.5 mas per mag of
  relative colour at tan z = 1: small next to the ~13-15 mas per-epoch floor, but systematic in colour and airmass.
  Decide whether to update DCR_AMPLITUDES (needs a full rerun) -- Nada/Dustin.
- Ring maps "461 million measurements, ~6,400 per bin per map": **can't be verified any more**. The input star files
  are purged and ringmaps.fits stores no counts. 461e6 / (32,768 bins x ~244 CCD-filter maps) ≈ 58 per bin, so
  "6,400" is wrong unless 461M is. Suggest: drop the per-bin number (or say ~60) and keep 461M only if Nada trusts it.
- (g-i) = -0.56 + 1.34 (BP-RP) (brick_DCR.py line 103, main.tex DCR section): Nada: Dustin supplied it, probably from
  the Legacy Surveys / DECam Community Pipeline calibration. **Ask Dustin for the reference.**
- Lateral pivot c0 = 1.642 (`median_color` in brick_lateralmaps.py since the Nov 2024 initial commit, no record of how
  it was computed; also in main.tex eq. lateral + run/lateral-model.ipynb): presumably the median g-i of the 2024 Gaia
  calibration sample. In the DR10 archive (60 bricks) the median Gaia-star g-i is 1.88 per star and 1.77 per σ<10 mas
  measurement. Small effect (a constant-in-colour offset of ~0.13 x the lateral shift, < ~4 mas at the g-band field
  edge, partly absorbed by the ring maps). Optional check: refit lateral-model.ipynb with c0 = 1.77 and compare
  held-out scatter. Main.tex wording "approximately the median colour of the calibration sample" is fine.
- Imaging date range (archive sample, 60 bricks, RA 60-64): **2013-11-09 to 2021-03-26** (MJD 56605-59299).
- **DONE 2026-10-12 (Nada: option a): Gaia-star PMs are now absolute.** legacypipe forced photometry (--move-gaia)
  places every source with ref_epoch > 0 at its Gaia-predicted position (PM + parallax) in each image, so the fit gives
  residuals from Gaia. create_brick_catalogue.reference_motion() adds Gaia pmra/pmdec/parallax back (variances added in
  quadrature) and starts the position from Gaia propagated to ref_mjd (PM only). Our columns are now absolute for every
  source; tractor columns (pmra, pmdec, parallax, ref_epoch) unchanged, so ours - tractor = residual from Gaia.
  pm_analysis.py / err-plots already compare ours - Gaia, which is now the right comparison (it was wrong on the archive).
  main.tex sec:pm_method updated (todo removed).

## Bugs / suspicious things found
- FIXED 2026-10-10: create_brick_catalogue.py crashed with NameError on the first successful fit of any
  brick -- commit b0c80d0 (2025-05-02) renamed the loop variables to var_plx/var but lines 218/228 still used
  variances_plx/variances. Resolved mystery (2026-10-12): the archive was made by FireWorks in Feb-Mar 2025 (launcher
  dirs, launched from the ~/LegacyPM home clone), i.e. *before* b0c80d0; the Sept 2025 file dates are copy dates.
- FIXED 2026-10-12 (create_brick_catalogue.py; all need the rerun):
  * **Offset columns of the design matrix were unweighted** (np.ones instead of 1/sigma, since the initial commit), so the
    model fitted was d_i = mu t_i + c sigma_i. Positions (new_ra/new_dec) were off by hundreds of mas; PMs slightly
    biased where per-epoch errors change with time. Effect of the fix on PMs (4 bricks): non-Gaia median |dPM| 0.1
    (r 20-22), 0.2 (r 22-23), 0.8 mas/yr (r 23-24) vs sigma 5 / 8 / 28 -> <=3% of sigma; Gaia-star scatter unchanged.
  * Dec position ivar was missing a square (new_dec_*_ivar ~100x too small).
  * new_ra/new_dec and ref_mjd were float32 (14 mas steps at RA 60 deg) -> now float64 ('D').
  * Validation (4 archive bricks, e2e-test/catcheck/check_cat.py): Gaia positions at ref_mjd agree with Gaia to 0.5 mas
    median, 5-6 mas robust sigma (before: ~350 mas sigma); Gaia PM - ours robust sigma 3.5 / 2.9 / 4.6 mas/yr for
    G 14-18 / 18-20 / 20-21.5, pulls 1.11 / 0.88 / 1.24; parallax pulls 0.9-1.06; non-Gaia PM/plx identical apart
    from the weighting fix.
- FireWorks: my_qadapter.yaml used `rlaunch singlerun` (not a command; fixed to `singleshot` + `-l` launchpad file,
  2026-10-12, untested). **MongoDB authentication fails** with the current credentials (2026-10-12) -> FireWorks is
  blocked until NERSC resets them. The Slurm driver (slurm/) works without it. See RUNBOOK.md.
- DCR: objects with no g-i colour get g_i_colour = 0 (brick_DCR.py line 102, `colormap.get(objid, 0.)`), i.e. they are
  corrected as if g-i = 0. Check how many PSF objects this affects.
- FIXED 2026-10-12: err-plots cell 8 (slide 20) had the "DCR corrected" / "DCR+RM corrected" labels swapped.
  Slide 20 itself is therefore mislabelled.
- FIXED 2026-10-12: err-plots cell 14 / dcr-plots cell 4 gave epochs whose star had been cut from T a t_index of -1,
  which silently picked up the last row of T (wrong colours/magnitudes). Those epochs are now dropped.
- FIXED 2026-10-12: pm_analysis.gaia_star_mask included Gaia 2-parameter stars (no Gaia PM, stored as 0), 8.5% of the
  058 validation sample, compared against PM = 0. Now requires Gaia pmra/pmdec/parallax ivar > 0.
- Noted 2026-10-12, negligible: the MIN_EPOCHS / MIN_BASELINE_DAYS cut counts all of a star's rows, including epochs
  with no valid fit (ivar 0, ~60-68% of rows; zero weight in the fit). Only 9 of 19,920 PSF stars on 10 test bricks
  pass because of them. Combined files carry n_epochs / baseline_days from valid epochs for a stricter analysis cut.
- create-maps.ipynb builds lateral maps with `dc = (T.flux_g - T.flux_i)` (flux difference!) but
  brick_lateralmaps.py applies them with magnitude colour g−i around pivot 1.642. Check which colour built the
  `lateralmaps.fits` actually in use.
- DCR K_i = K_z = 1 rather than 0 (see open questions).
- **g-band CCD maps make positions worse (found 2026-10-14 from F6).** Per-axis robust sigma of Gaia-star epochs with
  sigma < 10 mas, slice 000 (raw / DCR / +CCD maps / +lateral): g 18.98 / 15.58 / 18.26 / 17.77; r 13.30 / 12.74 /
  12.20 / 12.16; i 12.51 / 12.41 / 11.63 / 11.70; z 11.30 / 11.29 / 11.00 / 11.00 mas. In g, DCR + lateral WITHOUT the
  CCD maps gives 15.03 mas. The g maps shift positions by a median 14 mas (90th pct 33). Likely cause: the Apr 2024
  maps were built after a different DCR correction, so they absorbed a g-band DCR residual (or g maps are noisy).
  Fix options: skip the CCD maps in g (cheapest), or rebuild them. Needs the RM/LM/PM steps rerun -- combine with
  the K_b and objid fixes into one rerun. Also likely behind the pmdec pull width 1.40 vs pmra 1.10 (F8).
- **objid clashes (found 2026-10-14, when combine crashed on slice 002):** ~18% of tractor catalogues have rows from
  neighbouring bricks appended at the end (other brickid, own objid numbering; e.g. large SGA galaxies), and ~3% of
  forced files include those sources' epochs. The brick's own sources always have objid == row. Two pipeline steps
  match on objid only: brick_DCR's colour lookup (a clashing neighbour overwrites an own star's g-i) and
  create_brick_catalogue's row grouping (neighbour epochs merged into the own star's fit; appended PSF rows get the
  own star's fit). Affected PSF stars per slice (002 / 030 / 061): wrong DCR colour 171 / 140 / 79, merged epochs
  18 / 17 / 21, appended PSF rows 29 / 7 / 8 -- ~0.005% of ~4.5M per slice. NOT fixed in the pipeline yet (needs a
  rerun; do it together with the K_b decision): key both lookups on (brickid, objid) and skip non-own rows.
  Mitigation now: combine_catalogue.py matches on (brickid, objid), keeps only own sources, flags objid_clash;
  pm_analysis masks drop flagged stars.
- FIXED 2026-10-14: dcr-plots F2 "before" panels plotted the forced file's dp1 column, which is in PIXELS, next to
  dp1_after in arcsec on the same colour scale (both labelled arcsec), so the before signal looked 3.8x too small.
  Slide 15 has the same problem. Now both in arcsec (dp1_before computed from full_fit_x/y).

## Data coverage (checked 2026-10-10 against DR10 south survey-bricks)
- **No RA slice is complete.** Archive 060-063 has 548 valid bricks: 060 = 238/1152, 061 = 103/1140, 062 = 124/1139,
  063 = 83/1154. Mostly Dec 0 to +13.5. 058-059 and 064-069 have 0 valid bricks.
- Scope: DR10 south = 366,912 bricks in all 360 RA slices. Only slices 020-023 and 051-069 were ever attempted
  (real data only in 060-063). `missing_bricks.txt` covers only failed bricks in those slices, not the full footprint.
- Older 2024 test run on CFS: `/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/{dcr,rm,lm}-corrected-forced-motions/`
  = RA 001-040, |Dec| < 4.7 (6,240 bricks, `forced-brickwise-*`, Sept 2024, pre-DR10 forced run). No PM catalogues.
- Request list for Dustin: `missing_bricks.txt` (repo root; regenerate if his run progresses).
- **Dustin's regeneration: delivered 2026-10-09/10.** `/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick/` now has
  060: 914, 061: 1037, 062: 1015, 063: 1071 bricks (all with tractor-forced); with the 548 archive bricks that is all
  4,585 DR10 bricks in 060-063. Also 064: 83, 065: 49, and a handful in 020-023, 051-059, 066-069 (the P1 list).
  Re-check missing_bricks.txt against this before asking Dustin for anything else.
- 29 bricks have a corrected forced file but no PM catalogue (26 of them in 063). Only 4 have their
  tractor-forced input available again (0609p067, 0623m300, 0622p065, 0630m797); the other 25 need tractor-forced from Dustin.
- Older run for RA 000-057 still exists on CFS: `/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick/`
  (76,897/77,451 DR10 bricks; Dec 2024 - Jan 2025). Not purged, because CFS isn't. Usable (DR10), so RA 000-057 needs
  nothing from Dustin except 557 bricks absent from it.
- The archived PM catalogues predate the epoch/baseline guard. Either rerun step 4 locally or apply the cut at
  analysis time; no new forced photometry is needed for this.

## Health check of archived catalogues (2026-10-10; 62 random bricks from 060-063; scripts in session scratchpad)
- Code: no logic changes to DCR/RM/LM or to the PM fit math since the archive was made; only flags and the guard are new.
- `pm_flag` == 1 for 100% of PSF objects (old bug) -> unusable. 3.1% have all-zero PMs (failed fits).
- **32.7% of PSF objects fail the epoch/baseline cut**; 23/62 bricks have median baseline < 180 d (not just the 2 known
  bricks). Per-brick median baseline ranges 3 d - 2500 d. This is an imaging-cadence fact; rerunning won't change it.
- **Gaia stars: fitted PM is relative to Gaia** (fitted vs Gaia PM: slope -0.08, corr -0.03). The forced offsets for Gaia
  stars are taken relative to Gaia's own moving predicted position, so the fit gives PM - PM_Gaia. Consequences:
  (a) the pulls in pm_analysis.py / err-plots cells 29-32 compare against the wrong reference, so the "1.5-1.7x" numbers
  are invalid; validate Gaia stars against 0 instead. (b) The catalogue's PM columns mean different things for Gaia and
  non-Gaia stars; report PM_Gaia + fitted, or document it.
  Fitted-PM robust scatter for Gaia stars ~6-8 mas/yr (G 14-21) in that sample. [Superseded: the E2E test measured
  pull widths ~1.0-1.3 with the epoch/baseline cut, so the formal errors are roughly right.]
- **RM and LM corrections don't reduce scatter on this sample** (diagnosed below) (per-epoch dRA robust sigma, Gaia G<18):
  raw 13.4 -> DCR 12.25 -> RM 12.45 -> LM 13.04 mas (red stars: RM 13.0 -> LM 14.7). This contradicts slide 20
  (different data: deg 190, median 2-D offset). Suspects: the flux-vs-magnitude colour mismatch in the lateral maps;
  maps built from a different forced run than these data; the pixel->RA/Dec sign convention.
  Check dDec too, and per CCD at the focal-plane edges.
- Verdict: don't rerun yet -- a rerun with the current code reproduces the same numbers. Diagnose LM/RM first and decide
  how to report Gaia-star PMs, then rerun corrections + PM for the whole sample in one pass.

## Lateral-colour (LM) diagnosis (2026-10-10; Gaia stars G<19 in 80 random archive bricks; scratchpad scripts lmdiag*.py, conv.py)
Root causes, in order of impact:
1. **LM maps are noise-dominated.** In lateralmaps.fits, bin-to-bin rms = 0.074 px/mag vs smooth part 0.0044 (17x).
   Star level: only 13-25% (red) / 30-41% (blue) of the applied correction tracks the residual, so per-axis scatter
   gets worse: red 13.9 -> 15.3 mas, all 13.5 -> 14.0 mas. Lateral colour is a smooth optical effect; 16-px bins
   (128x256 per CCD per filter) waste the data, and dividing by (c - c0) amplifies the noise.
2. **The smooth part is about half the right size.** Per CCD/filter, the applied colour slope correlates with the slope
   in the data (r = 0.92; g band 0.96) but k = 0.51. The maps (Sept 2024) and ringmaps (Apr 2024) used in production come
   from the *pre-DR10 2024 forced run* (CFS ls-motions/...), not the DR10 data they're applied to. The Feb 2025 DR10 maps
   from create-maps.ipynb (rm-gaia-stars/, lm-gaia-stars/) were purged.
3. **RM and LM are applied only to measurements with full_fit_dra_ivar > 1e4 (sigma < 10 mas) -- ~12% of measurements.**
   Fainter measurements get DCR only, and a star's epochs near the threshold get inconsistent corrections. The ivar cut
   makes sense for *building* maps, not for *applying* them. Needs a decision (Nada/Dustin).
4. Bug (FIXED in working tree, uncommitted): lm_full_fit_{x,y,dra,ddec} were initialised from dcr_ instead of rm_, so
   measurements outside the LM selection lost their RM correction (0.23% of measurements).
Ruled out: pixel->sky convention (empirically dra = +0.263 dy, ddec = -0.263 dx on every CCD, matching the code); sign flip of
the maps (correlation is positive); flux-vs-magnitude colour (the production maps correlate well with the magnitude-colour
signal, so the flux-difference cell in create-maps.ipynb didn't make them).
RM (ring maps): roughly neutral to slightly helpful (82-91% of the correction is signal; 13.9 -> 13.5 mas in x), also noisy
(bin-to-bin 0.154 vs smooth 0.020 px). Lower priority.
Data issues: archive forced-0630m797 has no data; forced-0623m300 lacks rm_ columns.

Done 2026-10-10 (uncommitted): build/apply cut swap in brick_ringmaps.py + brick_lateralmaps.py -- maps now *built*
from ivar > 1e4 measurements and *applied* to every measurement with a valid fit (no S/N or dqmask cut). Origin of
the old cut: copied from the tree-ring plotting cell in map-plots/err-plots (same selection block); no record of a
deliberate decision.

Prototype `run/lateral-model.ipynb` (2026-10-10, 200 bricks, 508k Gaia-star measurements, held-out half of bricks):
- Production maps: smooth radial profile matches the data, but on held-out data the applied correction is ~pure noise
  (useful fraction -0.1 to +0.3) and makes scatter worse everywhere (g red 18.8 -> 22.3 mas, z red 12.4 -> 13.9).
- New smooth models never hurt: g band improves ~0.9 mas (blue 20.4 -> 19.4, red 18.8 -> 18.0); r/i/z essentially
  neutral (the effect is small there). Radial amplitude at the field edge (R ~ 16000 px): g 27, r 9.5, i 3.3,
  z 0.5 mas per mag of colour -- sensible chromatic ordering.
- 'poly' (2-D order-3 polynomial per component, 20 params/filter) removes per-CCD colour slopes better than 'radial'
  (median 0.71 vs 1.14 mas/mag; production 0.80, none 1.23). **Recommend poly.**
- APPROVED + wired in (2026-10-10, uncommitted): notebook's [APPROVED] cell writes py_files/lateral_model_poly3.json
  (fit on all 200 bricks); brick_lateralmaps.py now uses load_lateral_model() + lateral_shift() instead of the spline
  lookup (binned-map functions kept as legacy for create-maps.ipynb). lateral_shift matches the notebook model to
  1e-17 px. run/tests.ipynb cell 5 updated to the new API (labelled [EDITED]).
- E2E test (2026-10-10; 8 held-out bricks not used in the fit, median baseline >= 1 yr; scripts in scratchpad e2e/):
  pipeline rerun from DCR columns with new RM cut + new LM, PM step run on both old (archive) and new corrections.
  * Coverage: RM applied to 9.1% -> 99.9% of valid measurements; LM 9.0% -> 93.1% (rest: no finite g-i).
  * Per-epoch scatter, Gaia stars (mas): all 15.70 old -> 14.90 new (DCR-only 15.25); sigma<10mas 12.98 -> 12.13;
    G<19 blue 16.39 -> 14.89; G<19 red 14.08 -> 12.32. New is better than both old and DCR-only.
  * Faint measurements (sigma>10mas, newly corrected): 34.90 -> 35.43. The cost comes from the RING MAPS, not LM
    (DCR 34.90 -> +RM 35.57 -> +LM 35.43); not colour-range related. Hypothesis: part of the 2024 RM "signal" is
    flux-dependent, or RM bin noise exceeds the tree-ring signal for these stars. Revisit when RM is rebuilt from DR10.
  * Gaia-star PMs (relative to Gaia; robust sigma, mas/yr): G14-18 4.36 -> 4.15; G18-20 3.57 -> 3.31;
    G20-21.5 5.06 -> 4.98. Pull widths 0.94-1.30 -> formal PM errors are roughly right once the epoch/baseline cut
    is applied (this supersedes the earlier "~2x underestimated" estimate, which wasn't a measured pull).
  * Non-Gaia PSF stars: median |dPM| new-old ~2-2.5 mas/yr per axis, well below formal sigma (6-30 mas/yr).

Proposed fix (remaining):
- [prototyped, see above] Replace binned LM maps with a smooth model fit to DR10 Gaia stars: per filter, shift per mag as a low-order function of
  focal-plane position (radial polynomial, or a low-order 2-D polynomial per CCD). Use focal-plane coords from the
  crpixes dict in the notebooks. The archive's Gaia stars should be plenty for a few dozen parameters per filter.
- [done] Apply RM/LM to all measurements.
- [deferred, Nada 2026-10-11: not worth the effort now] Rebuild ring maps from DR10 Gaia stars. See "Ring-map rebuild" below.
- Validate with the same star-level test (useful fraction, scatter before/after, split by colour).

## Ring-map rebuild -- DEFERRED (decision 2026-10-11; notes in case we come back)
- Current maps: CFS `ls-motions/rm-corrected-forced-motions/ringmaps.fits` (Apr 2024), built by `get_gaia_data()` in
  brick_ringmaps.py from `gaia-*.fits` in Dustin's old `/pscratch/sd/d/dstn/forced-motions-3/` run (pre-DR10), with the
  old build cut ivar > 1. Those input star files and `/pscratch/sd/n/nelfalou/{dcr,rm,lm}-gaia-stars/` are purged; only
  the finished maps survive. A rebuild must pull Gaia stars (ref_cat == 'GE') from the per-brick forced-* files, as
  run/lateral-model.ipynb does. 548 archive bricks give only ~60 stars per 16-px bin -> wait for more of Dustin's data.
- Why it came up: E2E faint measurements (sigma > 10 mas) get worse with RM: DCR 34.90 -> +RM 35.57 -> +LM 35.43 mas.
- Expected gain (upper bounds from E2E/diagnosis numbers):
  * Faint: a perfect rebuild at most removes the RM cost -> ~34.8 mas (~0.6 mas, ~1.7%) -> faint PM errors ~1-2% better.
  * Bright: current RM is 82-91% signal (13.9 -> 13.5 mas in x, G<18); removing the remaining noise is worth ~0.5 mas in
    quadrature on ~13 mas, <1%, further diluted by the 14.8 mas floor.
  * Net PM effect of the current maps is already positive at every magnitude, so this is a refinement, not a fix.
- Cheaper alternative if the faint cost ever matters: apply the existing RM only to sigma < 10 mas measurements
  (one-line change to the apply selection in brick_ringmaps.create_brick_corrected_data; LM stays applied to all).
  Expected faint ~34.8 mas; verify by rerunning scratchpad e2e/ (run_e2e.py, eval_e2e.py, faint.py) on the 8 test bricks.
- If rebuilt: keep the 16-px bins only if stars per bin >> 100; otherwise consider a smoother model (as for LM).
  Validate with the same held-out star-level test as lateral-model.ipynb.

## Health check: processing new sky (2026-10-29; test output /pscratch/sd/n/nelfalou/healthcheck/, deletable)
- [x] Full pipeline on 064-069 (156 bricks, outside the calibrated area): no failures. Inputs exist only up to 069 ->
      Dustin must run forced photometry for 070-359 first. Disk ~130 GB per slice (v2 quota 20 TB).
- [x] FOUND: the zero points were silently missing on new sky (pass 1: 12% of epochs; 0% in 065-069) while headers said
      ZPCORR = T. Fixed: header ZPFRAC (fraction covered; also in combine summaries as zp_frac); zeropoints.py build
      over all combined slices / several dirs; LEGACYPM_ZP_TABLE override; slurm/redo_errors.sbatch. Two-pass test
      (combine -> test table from 70 slices -> redo -> combine): ZPFRAC median 0.9996, min 0.976. Production table
      unchanged.
- [ ] On new sky: recheck pulls (error model fitted on 000/040) and the maps/lateral model at other Dec / airmass.
- RUNBOOK.md rewritten (current pipeline only; section 5 = the two-pass procedure); README quick start updated.

## Runbook (issue #1) -- see RUNBOOK.md (written 2026-10-12)
- Slurm driver `slurm/run_bricks.py` + `slurm/run_bricks.sbatch` (no FireWorks needed). Tested on a login node in the
  Shifter image (archive mode and new-brick mode). Not yet submitted as a batch job.
- FireWorks workflow `fireworks/legacypm_workflow.py` now takes slices/options on the command line (`--from-archive`,
  `--input-root`, `--out-root`, `--dry-run`) and calls the scripts by absolute path. Blocked by MongoDB auth.
- New `py_files/prep_archive_rerun.py`: rebuilds RM/LM/PM inputs from the archive (raw inputs are purged).
- End-to-end test scripts saved to `run/e2e/` (were in the session scratchpad).

## Goal: RA 000-063 processed with the current code, ready for analysis (Nada, 2026-10-12)
- Inputs (2026-10-12): 84,358 DR10 bricks in 000-063; 81,491 available (Dustin pscratch 4,114 incl. 060-063, Dustin CFS
  76,894 for 000-057, archive 548). Missing 2,867 -> missing_bricks.txt (A: 557 in 000-057, B: whole 058-059).
- slurm/run_bricks.py rewritten: picks each brick's source automatically (pscratch > CFS > archive), accepts ranges,
  loads the DCR CCD table once (was ~45 s + 14 GB per brick) and forks workers: ~15-20 s/brick on a login node.
  Tested: 2 archive + 4 new bricks; dry run over 000-063 finds all 81,491.
- [x] Debug-qos test job 59704930 (058, 2026-10-12): 6/6 bricks OK, peak 20 GB. Full bricks took 80-120 s, mostly the
  PM step (per-object full-table masks + per-epoch radec_at_mjd loop). Sped up (row grouping once, vectorised
  parallax factors, plain arrays): PM 56 -> 5.8 s and 93 -> 6.6 s on full bricks; output bit-identical in every column
  on 6 bricks (4 from 058, 2 archive incl. Gaia stars). Typical brick now ~35-45 s; sbatch now 128 workers, 6 h.
- [x] 4 jobs 59705165-59705168 (000-015, 016-031, 032-047, 048-063), 2026-10-12: all COMPLETED in 4.3-5.2 h each
  (estimate was ~2 h: a full node runs at ~100 s per brick per worker, vs ~40 s for one brick alone). 81,296 ok,
  195 failed, all in DCR: bricks with a few i-band exposures but (almost) no stars with both g and i flux, so no g-i
  colour -> empty array -> np.vectorize crash. Fixed 2026-10-14 (otypes=[float]; DCR is then skipped for those stars,
  as for bricks with no i band). Tested on 0019p172.
- [ ] Rerun the 195 failures: `sbatch --qos=debug --time=00:30:00 run_bricks.sbatch 000-063` (skips finished bricks).
- [ ] rsync tractor-pm to CFS; ask Dustin about CFS space for the ~3.5 TB of forced files.
- [ ] Send missing_bricks.txt to Dustin; resubmit the same ranges when they arrive.
- [ ] Then: repoint notebooks, figures (Step 2), error analysis / 14.8 mas floor.
