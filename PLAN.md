# LegacyPM progress tracker

Last updated: 2026-10-12. Status keys: [x] done · [~] in progress · [ ] todo · [!] blocked

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

### Step 2: figures (all from notebooks in run/, saved to writeup/plots/)
Repoint notebooks from `legacypm-catalogue/190-192` (now empty) to `archive/legacypm-catalogue/060-063`.
| # | Figure | Notebook / cell | Status |
|---|---|---|---|
| F1 | Positional σ vs mag, Gaia vs LS (Dustin's request) | err-plots "Intro plots" | [ ] rerun & save (slide 9 version exists) |
| F2 | DCR before/after hexbin (replace DRC.png) | dcr-plots | [ ] rerun & save (slide 15 version exists) |
| F3 | K_b fit per band | new cell in dcr-plots | [ ] |
| F4 | Tree-ring map | ringmaps.png | [x] |
| F5 | Lateral colour focal-plane maps, before/after | err-plots "Lateral plots" | [ ] rerun & save (slides 18-19) |
| F6 | Residual scatter vs mag per correction stage | err-plots cell 8 | [ ] **fix swapped labels first** (see Bugs) |
| F7 | PM residual (ours − Gaia) vs G | err-plots cell 31 | [ ] adapt to hexbin |
| F8 | Pull distributions PM + parallax | err-plots cell 29 | [ ] run (never executed) |
| F9 | PM precision vs mag, extended past G=21 | err-plots cell 32 | [ ] extend |
| F10 | Parallax vs Gaia | new | [ ] |
| F11 | Footprint + N_epochs / baseline histograms | new | [ ] |

### Step 3: results & discussion text
- [ ] Sample description (548 bricks, RA ≈ 60–64°; counts of stars, PSF, with PM, Gaia/non-Gaia)
- [ ] Analysis of Corrected Data (from F6)
- [ ] Results: PM vs Gaia, uncertainty calibration (pull widths; commit msgs say ~1.5–1.7 PM, ~1.3 plx -- re-measure)
- [ ] Parallax: honest assessment (cadence-limited)
- [ ] Data product: table of output columns
- [ ] Discussion (notes at end of main.tex): empirical maps vs Bernstein parametric; limitations
- [!] High-PM / non-Gaia candidates -- needs analysis-time epoch/baseline cut on the 548 bricks
- [ ] Abstract, conclusion, keywords

## Open questions (for Nada / Dustin) -- answers 2026-10-12
- 14.8 mas floor: not computed in any saved cell; matches the ~15 mas bright-end plateau of F6. Note F6 plots the
  median of the *2-D* offset hypot(dra, ddec); for a 2-D Gaussian that's ≈1.18 σ per axis, so the per-axis floor
  would be ≈12.6 mas. **Nada: redo the error analysis for the paper, but only after more RA slices are processed.**
- K_i = K_z = 1 mas/mag: **Nada: negligible by design; we don't correct those bands' colours. Keep as is.**
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
- err-plots cell 8 (= slide 20 "Error Analysis"): labels "DCR corrected" and "DCR+RM corrected" are swapped
  (scat3 = rm is labelled DCR, scat2 = dcr is labelled DCR+RM).
- create-maps.ipynb builds lateral maps with `dc = (T.flux_g - T.flux_i)` (flux difference!) but
  brick_lateralmaps.py applies them with magnitude colour g−i around pivot 1.642. Check which colour built the
  `lateralmaps.fits` actually in use.
- DCR K_i = K_z = 1 rather than 0 (see open questions).

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
  on 6 bricks (4 from 058, 2 archive incl. Gaia stars). Typical brick now ~35-45 s; sbatch now 128 workers, 6 h. [ ] 4 jobs: 000-015, 016-031, 032-047, 048-063 (RUNBOOK TL;DR).
- [ ] rsync tractor-pm to CFS; ask Dustin about CFS space for the ~3.5 TB of forced files.
- [ ] Send missing_bricks.txt to Dustin; resubmit the same ranges when they arrive.
- [ ] Then: repoint notebooks, figures (Step 2), error analysis / 14.8 mas floor.
