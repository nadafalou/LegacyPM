## F1 — Tractor positional uncertainty vs magnitude (pos_err_vs_mag.png, one column)
- Shows: the catalogue's quoted position error against r magnitude, for Gaia stars (orange) and sources too faint for Gaia (blue).
- Point: the Legacy Surveys goes about 3 magnitudes deeper than Gaia. This is the figure Dustin asked for.
- Decisions:
  - errors are in mas (√(σ_RA² + σ_Dec²)), and sources with zero or negative flux are dropped (they caused the old 0–40 mag axis);
  - r runs from 14 to 25;
  - each group is a random 300k-point subsample, to keep the file small.

## F2 — DCR before/after (dcr_before_after.png, one column, shared airmass axis)
- Shows: the median offset along the parallactic direction for r-band Gaia-star epochs, against airmass and Gaia BP−RP colour, before (top) and after (bottom) the DCR correction.
- Before: red stars shift one way, blue the other, more so at high airmass.
- After: that pattern is mostly gone.
- Decisions: r band; ±50 mas symmetric scale; bins with fewer than 15 epochs left blank.
- Note: the old version mixed pixel and arcsec units, and slide 15 has that bug.

## F3 — DCR amplitude per band (dcr_amplitude_fit.pdf, full width, shared x-axis)
- Shows: the same offset against relative colour × tan z for each band. The slope is the DCR amplitude K_b, shown next to the value the pipeline uses.
- Decisions:
  - only σ < 10 mas epochs, before correction;
  - medians in bins with at least 30 epochs;
  - a straight line weighted by each median's error.
- On slice 000: g 31.6, r 9.0, i 4.6 (pipeline uses 1), z 1.05 mas/mag.
- The g-band medians bend at large colour × tan z, so a straight line is only approximate there.

## F4 — Tree rings, one CCD (ringmap_N4_z.pdf, one column) and across the focal plane (ringmap_focal_plane_z.png, full width)
- Shows: the per-CCD distortion maps (median measured − Gaia offset in 16-pixel bins) along CCD x and y: first N4 in z, then all 61 CCDs at their real positions.
- Point: each CCD has its own rings, centred independently of where the chip sits, which is why the maps are made per CCD.
- Decisions:
  - z band, where the rings are strongest;
  - ±30 mas scale; grey means no data;
  - CCD names on the mosaic.
- Sanity checks: S7's unused half is grey and the dead N30 is missing.
- Source: these come from the 2024 map file, not this week's run.

## F5 — Lateral colour across the focal plane (lateral_colour_bins.png, full width, shared axes)
- Shows: the median radial residual of g-band Gaia-star epochs across the focal plane, for blue, middle and red stars, before (top) and after (bottom) the lateral-colour correction.
- Before: blue stars are pulled inward at the field edge (about −15 mas) and red stars pushed outward. The middle colour bin, around the pivot colour, shows almost nothing.
- After: the edge pattern is gone.
- Decisions:
  - I switched from r to g because the effect is about 3× larger there; set filt = 'r' in cell 15 if you prefer r;
  - ±15 mas scale (the old ±52 mas washed it out);
  - coarse bins (about 0.05°, at least 10 epochs each).
- Leftover: a left/right dipole remains in all three colour bins. Because it doesn't depend on colour, it isn't lateral colour.

## F6 — Per-epoch residuals vs magnitude (scatter_vs_mag.pdf, one column)
- Shows: the median 2-D offset of Gaia-star epochs from Gaia after each correction stage, plus the median statistical error, against magnitude.
- Bright stars: they sit on a floor of about 18–21 mas, far above their statistical errors.
- Faint stars (g ≳ 22): photon noise takes over and all the curves meet.
- Decisions:
  - g-band epochs only: the magnitude comes from each epoch's own flux, so mixing bands mixed magnitude scales. That caused the dip at about 22.4 and made the old "g" label wrong.
  - bins with fewer than 100 epochs are skipped.
- This is where the g-band CCD-map problem shows: the DCR-only curve is the lowest.

## F7 — Our PMs vs Gaia's (pm_vs_gaia.png, full width)
- Shows: 4-parameter-fit PMs against Gaia's for 558k Gaia stars, as a log-density plot with a 1:1 line.
- Point: the tight band along 1:1 is the agreement. The vertical spread near zero is faint stars with small true motions, where our ±5–30 mas/yr errors dominate.
- Decisions: only stars with a full Gaia astrometric solution, a successful fit and no objid clash; ±200 mas/yr range.

## F8 — Uncertainty calibration (pulls) (pulls.pdf, full width, shared axes)
- Shows: (ours − Gaia) / our quoted error, for the PM components from the 4- and 5-parameter fits and for parallax, against a unit Gaussian.
- Results: μ_α* 1.10, parallax 1.13, and μ_δ 1.40: the Dec errors are about 40% too small.
- The extra Dec-direction error fits what's leftover from uncorrected DCR, both the g-band maps and K_i, since the parallactic direction is roughly north–south for most exposures.
- Decisions: same Gaia-star selection as F7; the width is the robust σ.

What a "pull" is. For each Gaia star:

  pull = (our value − Gaia's value) / our quoted error

Gaia's PMs and parallaxes are much more precise than ours, so the difference is essentially our measurement error. If our quoted errors are right, the pulls follow a unit Gaussian: centred on 0, with width 1. That's the dashed grey curve in each panel.
- Width above 1: our errors are too small, i.e. we're overconfident.
- Width below 1: our errors are too large.
- Median away from 0: a systematic bias.

The three panels:
1. PM, 4-parameter fit: position + PM. These are the pmra_no_plx / pmdec_no_plx columns. Blue is μ_α* (RA), orange is μ_δ (Dec).
2. PM, 5-parameter fit: position + PM + parallax (pmra_plx / pmdec_plx).
3. Parallax, 5-parameter fit (parallax_new).

The "width" in each legend is a robust standard deviation (1.4826 × the median absolute deviation), so a few wild outliers don't inflate it.

Results (slices 000–001, about 558k Gaia stars):

| | Median | Width | Meaning |
| -----|---------|------------|-----------------|
| μ_α*    | −0.02  | 1.10  | errors ~10% too small; no bias  |
| μ_δ     | +0.17  | 1.40  | errors ~40% too small; small bias (0.17σ) |
| parallax | −0.10  | 1.13  | errors ~13% too small; slight bias  |

The 4- and 5-parameter panels are almost identical. Adding parallax to the fit doesn't change the PMs or their errors much, which you'd expect since parallax is weakly constrained by this cadence.

Three things to take from it:
- RA and parallax errors are nearly right. A width of 1.1 is typical for real data, and could be fixed by scaling the errors up by 1.1 or by adjusting the 14.8 mas floor.
- Dec errors are clearly underestimated. Something adds extra scatter in the north–south direction that our error model doesn't include. The likely cause is leftover DCR. DCR acts along the parallactic direction, which for most DECam exposures is close to north–south. Two pieces of uncorrected DCR are already known:
  - the g-band CCD maps, which add error in g;
  - the i-band amplitude, which is 4.5× too small.

  Fixing those and rerunning should bring the Dec width down. This figure is a good check after that rerun.
- The tails are heavier than a Gaussian (the histograms sit above the dashed curve beyond about ±3). A small fraction of stars have errors much larger than quoted. They could be bad epochs, blends or high-PM outliers. The robust width ignores them, but it's worth quantifying them for the paper, e.g. the fraction of stars with |pull| > 5.

How it's made:
- Stars: Gaia stars with a full Gaia 5-parameter solution and a successful fit (pm_flag == 1), excluding the objid_clash stars.
- Gaia's error: our quoted error already includes Gaia's PM error in quadrature, since Gaia's PM was added back to our fit. Gaia's error is tiny, so this barely matters.
- Display: histograms clipped to ±10 for display; the widths use all stars.


## F9 — PM precision vs magnitude (pm_precision_vs_mag.pdf, one column)
- Shows: the median |ours − Gaia| for each PM component in Gaia G bins.
- Results:
  - best about 1.65 (α) and 2.1 (δ) mas/yr at G ≈ 18;
  - worse for bright stars, probably saturation or brighter-fatter effects;
  - worse for faint stars, from photon noise.
- Decisions:
  - stops at G ≈ 21, where Gaia does; going fainter needs another method, not written yet;
  - I removed the per-bin star counts, which were being drawn off the axes. They now print to the notebook: 12.8k to 88k stars per bin.

## F10 — Our parallaxes vs Gaia's (parallax_vs_gaia.pdf, full width)
- Left: log density with the median in Gaia-parallax bins. It follows 1:1, but our ~8 mas errors are much larger than typical parallaxes.
- Right: the median offset, robust scatter and our quoted error against G.
  - G 17–20: scatter matches our quoted errors, so the parallax errors are honest there.
  - G < 16: the scatter jumps to 15–21 mas, with a −4 mas offset, again pointing at bright-star systematics.
- Decisions: Gaia stars with a full astrometric solution; bins need more than 20 stars.

## F11 — Footprint and temporal coverage (footprint.png, full width)
- Shows: each brick's position coloured by its number of objects with PMs, plus histograms of the per-brick median number of epochs and median time baseline, with the 180-day cut marked.
- With 000–001: the RA axis spans only 2°, so the bricks form arcs. The brick grid's RA spacing changes with Dec. With all of 000–063 it will look like a filled region.
- Sample: 2,898 bricks with 8.3M point sources that have PMs. About 140 bricks have baselines under 180 days, and those give no PMs.


# Not for writeup: DCR polyfit

Shared setup. Each point comes from Gaia-star epochs measured before any DCR correction, with position errors under 10 mas.
- y-axis (offset): how far each epoch's measured position is shifted from Gaia's, along the parallactic direction (the direction toward the zenith, where refraction pushes light). This is the shift DCR correction has to remove.
- c_rel: the star's g−i colour minus the average colour of the reference stars on that CCD. Stars bluer or redder than the reference stars are refracted differently.
- tan z: how far from the zenith the telescope was pointing, related to airmass. 0 means straight up, and the effect grows with it.
- Black dots: binned medians. Epochs are grouped into bins, and each dot is the median offset in its bin, with error bars for the uncertainty on that median.
- Held-out test: bricks were split randomly in half. Each model is fitted on one half and checked on the other. A model that only memorised the noise in its half does worse on the other half, so this is the check against overfitting.

1. dcr_polyfit.pdf: Dustin's suggestion

Top row: offset against x = c_rel × tan z, one panel per band. The coloured curves are polynomial fits in x of order 1 (a straight line, which is the current pipeline model), 3, 4 and 5.

Bottom row: held-out residuals, i.e. the other half's binned medians minus each fit. A good model sits on the zero line.

What it shows:
- In g, the straight line misses badly at both ends, with residuals up to ±10 mas. Higher orders follow the curve better; order 5 gives the flattest residuals, and the held-out check shows no overfitting.
- In r, i and z, all the orders look about the same. The straight line already fits to within a few mas.
- The weakness: the higher-order curves bend sharply at the edges (g order 4 at the right end, z order 5), where there's little data. That's the extrapolation risk Dustin was worried about.

2. dcr_curvature_g_r.pdf: where the curvature comes from

This figure tests the assumption behind figure 1: that DCR depends only on the product c_rel × tan z. Data are split into three airmass ranges (blue: tan z 0.05–0.3, orange: 0.3–0.6, green: 0.6–1.2). Top row is g, bottom row is r.

Left column: offset against x = c_rel × tan z. If DCR were a function of x alone, the three curves would lie on top of each other. They don't. At negative x the low-airmass curves hook upward while the high-airmass one keeps going down.

Right column: offset ÷ tan z against c_rel. If DCR = tan z × (some function of colour), these three curves would coincide. They do, closely, over the whole colour range.

What it means: DCR scales with tan z, as refraction theory says. The non-linearity is in colour: the offset per unit tan z is a curved function of c_rel, flattening for very blue and very red stars. That's physically sensible: what sets refraction is a star's effective wavelength in the filter, and g−i doesn't map onto that linearly, especially at the extremes. A polynomial in x is the wrong shape. It mixes airmass and colour, which is why its curves have to bend oddly at the ends.

3. dcr_tanz_polyfit.pdf: the model that follows

Top row: offset ÷ tan z against c_rel, for epochs with 0.3 < tan z < 1.3 (one airmass range, so each colour bin gives one point).
- Grey dotted line: the current linear model.
- Coloured curves: DCR = tan z × polynomial in colour, of order 1, 2 and 3.

Bottom row: held-out residuals for each, in mas.

What it shows:
- Over −1 < c_rel < 2, where nearly all stars are, the quadratic (orange) and cubic (green) follow the data well, and the residuals are close to zero in every band. The linear models (blue, grey) leave a systematic curve, most visibly in g.
- At the extremes (c_rel < −1.5 or > 2.2), the data flatten. Every polynomial misses there: the quadratic and cubic overshoot in r and i at the red end, and everything misses g at the blue end. Few stars have those colours.
- The order-1 line in r and i sits slightly off the grey "current" line even near c_rel = 0. That's the constant term: stars at the reference colour still shift a little with airmass, which points to a small offset in the reference-colour conversion.

How the models compare (held-out RMS in mas, scored in cells of colour × airmass, which tells the shapes apart):

| Band | Current | x, order 5 | tan z × order 2 | tan z × order 4 |
| -----|---------|------------|-----------------|-----------------|
| g    | 2.57    | 1.24       | 1.53            | 1.15            |
| r    | 1.18    | 1.15       | 0.63            | 0.60            |
| i    | 0.78    | 0.76       | 0.46            | 0.41            |
| z    | 0.51    | 0.49       | 0.43            | 0.42            |


The tan z × colour model wins in every band. Going from order 2 to 4 buys little except in g.
