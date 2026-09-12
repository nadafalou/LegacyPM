"""
Analysis helpers for comparing pipeline-derived proper motions / parallax
against Gaia, using the archived per-brick tractor-pm-*.fits catalogues.

These are written to operate over "however many bricks are currently valid" --
pass `degrees=None` (the default) to use everything found under `archive_dir`,
so re-running once more bricks come back from Dustin needs no code changes,
just a bigger archive.
"""
import glob
import os
import logging

import numpy as np
from astrometry.util.fits import fits_table, merge_tables

logger = logging.getLogger(__name__)

# Columns needed for PM/parallax-vs-Gaia validation; kept narrow so loading
# hundreds of bricks stays fast and light on memory (each catalogue has ~255
# columns total, most irrelevant here).
_COLUMNS = [
    'brickid', 'brickname', 'objid', 'type', 'ra', 'dec', 'ref_cat',
    'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag',
    'pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar',
    'pmra_plx', 'pmra_plx_ivar', 'pmdec_plx', 'pmdec_plx_ivar',
    'parallax_new', 'parallax_new_ivar',
    'pmra_no_plx', 'pmra_no_plx_ivar', 'pmdec_no_plx', 'pmdec_no_plx_ivar',
    'pm_flag', 'ref_mjd',
]


def _is_complete(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _load_one(args):
    path, columns = args
    try:
        return path, fits_table(path, columns=columns)
    except Exception as e:
        return path, None


def load_catalogue(archive_dir, degrees=None, columns=_COLUMNS, n_workers=8):
    """Load and concatenate all valid tractor-pm-*.fits catalogues under archive_dir.

    degrees: optional list of degree-directory names to restrict to (default: all
        present under archive_dir).
    Skips zero-byte/corrupt files automatically so a partially-processed archive
    (or one with bricks still awaiting re-run) doesn't break the load.
    n_workers: reads files in parallel (this is Lustre I/O bound, hundreds of
        bricks serially takes minutes; set to 1 to disable).
    Returns (table, skipped_paths).
    """
    if degrees is None:
        degrees = sorted(d for d in os.listdir(archive_dir) if os.path.isdir(os.path.join(archive_dir, d)))

    paths = []
    skipped = []
    for deg in degrees:
        deg_dir = os.path.join(archive_dir, deg)
        if not os.path.isdir(deg_dir):
            continue
        for path in sorted(glob.glob(os.path.join(deg_dir, "tractor-pm-*.fits"))):
            if not _is_complete(path):
                skipped.append(path)
                continue
            paths.append(path)

    tables = []
    if n_workers > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
            for path, T in ex.map(_load_one, [(p, columns) for p in paths]):
                if T is None:
                    skipped.append(path)
                else:
                    tables.append(T)
    else:
        for path in paths:
            path, T = _load_one((path, columns))
            if T is None:
                skipped.append(path)
            else:
                tables.append(T)

    if not tables:
        raise RuntimeError(f"no valid catalogue files found under {archive_dir} for degrees={degrees}")

    T = merge_tables(tables, columns='fillzero')
    if skipped:
        print(f"skipped {len(skipped)} corrupt/incomplete catalogue file(s) (see returned list)")
    return T, skipped


# Sanity bound on |PM|, mas/yr. The fastest known real stars (e.g. Barnard's
# Star) are ~10,000 mas/yr; anything beyond that in this dataset has -- every
# case checked -- been a numerically degenerate least-squares fit (very few or
# time-clustered epochs giving an ill-conditioned design matrix in pm_lsq/
# pm_lsq_plx) that still "succeeds" without raising, rather than a real star.
# This is a stopgap analysis-side cut, not a physical claim; the real fix
# belongs in calculate_pm (e.g. a minimum-epoch-count or condition-number
# check) -- flagged separately, not applied here since it touches core fit
# math and wasn't part of the already-agreed bug list.
_MAX_SANE_PM_MAS_PER_YR = 1e4


def gaia_star_mask(T, model='no_plx'):
    """Boolean mask: Gaia-reference PSF stars with a genuinely successful,
    numerically-sane fit.

    `model`: 'no_plx' (plain proper-motion fit) or 'plx' (simultaneous PM+parallax
    fit) -- selects which pair of pmra/pmdec columns to check.

    The nonzero check works around a bug in the pipeline (fixed in code, but the
    currently-archived catalogues predate the fix): pm_flag was previously set to
    1 even when the fit failed and left pmra/pmdec at their zero-initialized
    value. Once catalogues are regenerated with the fixed pipeline, pm_flag==1
    alone will be reliable and this extra check becomes a no-op.

    The |PM| sanity bound guards against the numerical-degeneracy issue above --
    see _MAX_SANE_PM_MAS_PER_YR.
    """
    pmra_col = 'pmra_no_plx' if model == 'no_plx' else 'pmra_plx'
    pmdec_col = 'pmdec_no_plx' if model == 'no_plx' else 'pmdec_plx'
    pmra = getattr(T, pmra_col)
    pmdec = getattr(T, pmdec_col)
    return (
        (T.ref_cat == 'GE') &
        (T.type == 'PSF') &
        (T.pm_flag == 1) &
        ((pmra != 0) | (pmdec != 0)) &
        (np.abs(pmra) < _MAX_SANE_PM_MAS_PER_YR) &
        (np.abs(pmdec) < _MAX_SANE_PM_MAS_PER_YR)
    )


def non_gaia_psf_mask(T, model='no_plx'):
    """Boolean mask: non-Gaia-reference PSF stars with a genuinely successful,
    numerically-sane fit -- the pool to search for high-PM discoveries in."""
    pmra_col = 'pmra_no_plx' if model == 'no_plx' else 'pmra_plx'
    pmdec_col = 'pmdec_no_plx' if model == 'no_plx' else 'pmdec_plx'
    pmra = getattr(T, pmra_col)
    pmdec = getattr(T, pmdec_col)
    return (
        (T.ref_cat != 'GE') &
        (T.type == 'PSF') &
        (T.pm_flag == 1) &
        ((pmra != 0) | (pmdec != 0)) &
        (np.abs(pmra) < _MAX_SANE_PM_MAS_PER_YR) &
        (np.abs(pmdec) < _MAX_SANE_PM_MAS_PER_YR)
    )


def pm_pulls(T, mask, model='no_plx'):
    """(pull_ra, pull_dec, pull_parallax) = (measured - gaia truth) / formal_sigma."""
    suffix = '_no_plx' if model == 'no_plx' else '_plx'
    pmra = getattr(T, 'pmra' + suffix)[mask]
    pmdec = getattr(T, 'pmdec' + suffix)[mask]
    pmra_ivar = getattr(T, 'pmra' + suffix + '_ivar')[mask]
    pmdec_ivar = getattr(T, 'pmdec' + suffix + '_ivar')[mask]

    pull_ra = (pmra - T.pmra[mask]) * np.sqrt(pmra_ivar)
    pull_dec = (pmdec - T.pmdec[mask]) * np.sqrt(pmdec_ivar)

    pull_plx = None
    if model == 'plx':
        plx = T.parallax_new[mask]
        plx_ivar = T.parallax_new_ivar[mask]
        pull_plx = (plx - T.parallax[mask]) * np.sqrt(plx_ivar)

    return pull_ra, pull_dec, pull_plx


def precision_vs_magnitude(T, mask, model='no_plx', mag_col='gaia_phot_g_mean_mag',
                            mag_bins=np.linspace(15, 21, 13)):
    """Median |residual vs Gaia| in bins of magnitude, for pmra/pmdec.

    Returns (mag_centers, median_abs_resid_ra, median_abs_resid_dec, counts).
    This is the "precision vs magnitude" figure -- extend to non-Gaia stars later
    (item 5 in the paper plan) once there's a way to establish ground truth for
    them (e.g. repeat-visit scatter), which the current single-epoch-per-brick
    Gaia-anchored comparison can't do on its own.
    """
    suffix = '_no_plx' if model == 'no_plx' else '_plx'
    pmra = getattr(T, 'pmra' + suffix)[mask]
    pmdec = getattr(T, 'pmdec' + suffix)[mask]
    mag = getattr(T, mag_col)[mask]
    gaia_pmra = T.pmra[mask]
    gaia_pmdec = T.pmdec[mask]

    resid_ra = np.abs(pmra - gaia_pmra)
    resid_dec = np.abs(pmdec - gaia_pmdec)

    centers, med_ra, med_dec, counts = [], [], [], []
    for lo, hi in zip(mag_bins[:-1], mag_bins[1:]):
        J = np.flatnonzero((mag > lo) & (mag <= hi))
        centers.append((lo + hi) / 2)
        counts.append(len(J))
        med_ra.append(np.nanmedian(resid_ra[J]) if len(J) else np.nan)
        med_dec.append(np.nanmedian(resid_dec[J]) if len(J) else np.nan)

    return np.array(centers), np.array(med_ra), np.array(med_dec), np.array(counts)


def robust_stats(x):
    """Median and a robust (outlier-resistant) sigma estimate via normalized MAD.

    Astrometric pull distributions from a large automated pipeline like this one
    reliably include a small tail of catastrophic outliers (near-degenerate
    fits that technically succeeded but shouldn't be trusted quantitatively) --
    plain mean/std is dominated by those, so summarize with median + MAD instead,
    and report the outlier count/fraction separately rather than hiding it.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    robust_sigma = 1.4826 * mad  # normal-consistent scaling
    n_outlier = int(np.sum(np.abs(x - med) > 10 * robust_sigma)) if robust_sigma > 0 else 0
    return {
        'n': len(x),
        'median': med,
        'robust_sigma': robust_sigma,
        'mean': np.mean(x),
        'std': np.std(x),
        'n_outlier_10sigma': n_outlier,
        'frac_outlier_10sigma': n_outlier / len(x) if len(x) else np.nan,
    }
