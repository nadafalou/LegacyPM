from astropy.io import fits
import numpy as np
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u
from astrometry.util.fits import os, fits_table
from legacypipe.survey import radec_at_mjd, mjd_to_year
import sys
import logging
import time
try:
    from .error_model import total_error, SIGMA_SYS_MAS
except ImportError:
    from error_model import total_error, SIGMA_SYS_MAS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DAYSPERYEAR = 365.2425

# Quality is FLAGGED, not cut (Nada, 2026-10-23): every PSF object gets a PM whenever a fit is numerically possible,
# and pm_quality (bitmask below) says why it may not be trustworthy. Analyses and figures use pm_quality == 0
# ("clean"); the catalogue keeps everything.
#   MIN_EPOCHS_FIT / baseline > 0: the minimum for a fit to exist at all (5 parameters from 2 x N equations).
#   FEW_EPOCHS, SHORT_BASELINE_DAYS: below these, PMs are unreliable (PLAN H1: outlier rates 10% / 26% vs 0.13% for
#     clean stars; a ~36-day baseline gave |PM| ~ 1e16 mas/yr in the old archive).
#   MAX_COND_NUMBER: numerically singular design matrix (duplicate epochs) -> fit failed.
MIN_EPOCHS_FIT = 3
FEW_EPOCHS = 10
SHORT_BASELINE_DAYS = 730
MAX_COND_NUMBER = 1e8

# pm_quality bits
PMQ_FEW_EPOCHS = 1        # < FEW_EPOCHS measured epochs
PMQ_SHORT_BASELINE = 2    # measured epochs span < SHORT_BASELINE_DAYS
PMQ_NO_COLOUR = 4         # no g - i colour (flux_g or flux_i <= 0): no DCR or lateral-colour correction applied
PMQ_BRIGHT = 8            # Tractor maskbits BRIGHT or MEDIUM (near / is a bright star)
PMQ_SATURATED = 16        # Tractor maskbits SATUR_G, SATUR_R, SATUR_Z or SATUR_I
PMQ_FIT_FAILED = 32       # fewer than MIN_EPOCHS_FIT measured epochs, zero baseline, or a singular fit
MASKBITS_BRIGHT = (1 << 1) | (1 << 11)
MASKBITS_SATURATED = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 14)

BRICK_WARN_INSUFFICIENT_FRACTION = 0.5


def pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, B):
    A = np.zeros((2 * N, 5))
    A[:N, 0] = plx_dra / dra_error
    A[N:, 0] = plx_ddec / ddec_error
    A[:N, 1] = dt / dra_error
    A[N:, 2] = dt / ddec_error
    A[:N, 3] = 1 / dra_error  # offset columns must be weighted like the data (B = d/sigma)
    A[N:, 4] = 1 / ddec_error

    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > MAX_COND_NUMBER:
        raise np.linalg.LinAlgError(f"ill-conditioned design matrix (cond={cond:.3g} > {MAX_COND_NUMBER:.3g})")

    X, resid, rank, s = np.linalg.lstsq(A, B, rcond=None) # in arcsec
    X = X * 1000 # arcsec to mas
    # parallax, pmra_plx, pmdec_plx, ra_offset_plx, dec_offset_plx = X[0], X[1], X[2], X[3], X[4] # mas, mas/yr, mas/yr
    # ra_offset_plx = ra_offset_plx / 1000 # to arcsec
    # dec_offset_plx = dec_offset_plx / 1000 # to arcsec

    cov_matrix = np.linalg.inv(A.T @ A)
    variances = np.diag(cov_matrix) * 1000**2
    # parallax_var, pmra_plx_var, pmdec_plx_var, ra_offset_plx_var, dec_offset_plx_var = variances

    return X, variances  # all in mas


def pm_lsq(N, dt, dra_error, ddec_error, B):
    A = np.zeros((2 * N, 4))
    A[:N, 0] = dt / dra_error
    A[N:, 1] = dt / ddec_error
    A[:N, 2] = 1 / dra_error  # offset columns must be weighted like the data (B = d/sigma)
    A[N:, 3] = 1 / ddec_error

    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > MAX_COND_NUMBER:
        raise np.linalg.LinAlgError(f"ill-conditioned design matrix (cond={cond:.3g} > {MAX_COND_NUMBER:.3g})")

    X, resid, rank, s = np.linalg.lstsq(A, B, rcond=None) # in arcsec
    X = X * 1000
    # pmra, pmdec, ra_offset, dec_offset = X[0], X[1], X[2] # mas, mas/yr, mas/yr

    cov_matrix = np.linalg.inv(A.T @ A)
    variances = np.diag(cov_matrix) * 1000**2
    # pmra_var, pmdec_var, ra_offset_var, dec_offset_var = variances

    return X, variances  # all in mas


def calculate_pm(mjd, dra, ddec, dra_error, ddec_error, ra, dec):
    """Fit one object. mjd, dra, ddec (arcsec) and the errors are that object's per-epoch measurements;
    ra, dec its Tractor position (deg)."""
    ref_year = 0 # bc dra, ddec around 0, not actual ra, decs
    N = len(mjd)
    dra_ivar = 1 / dra_error ** 2
    ref_mjd = np.sum(dra_ivar * mjd) / np.sum(dra_ivar) # ref_mjd for this dataset is different from Gaia's ref_mjd
                                                        # theoretically, dra_ivar ~= ddec_ivar
    baseline_days = mjd.max() - mjd.min() if N > 0 else 0
    if N < MIN_EPOCHS_FIT or baseline_days <= 0:   # no fit possible; data-quality limits are flagged by the caller
        return None, None, None, None, ref_mjd

    # change in ra, dec due to parallax = 1 arcsec, at each epoch (radec_at_mjd needs all-vector arguments)
    one = np.ones(N)
    plx_ra, plx_dec = radec_at_mjd(ra * one, dec * one, ref_year * one, 0 * one, 0 * one, 1000 * one, mjd)
    plx_dra = (plx_ra - ra) * np.cos(np.deg2rad(dec)) * 3600
    plx_ddec = (plx_dec - dec) * 3600
    dt = (mjd - ref_mjd) / DAYSPERYEAR
    B = np.append(dra / dra_error, ddec / ddec_error)

    try:
        X_plx, variances_plx = pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, B)
    except np.linalg.LinAlgError as err:
        X_plx, variances_plx = None, None

    try:
        X, variances = pm_lsq(N, dt, dra_error, ddec_error, B)
    except np.linalg.LinAlgError as err:
        X, variances = None, None

    return X_plx, variances_plx, X, variances, ref_mjd


def reference_motion(tractor, objid, ref_mjd):
    """The forced photometry moves every source with ref_epoch > 0 (Gaia stars) to each image's epoch using the
    catalogue's pmra, pmdec and parallax (legacypipe forced_photom, --move-gaia), so for those sources the fitted
    dra/ddec trends are residuals from the Gaia motion. Returns the reference position at ref_mjd (proper motion
    only, no parallax displacement), and the motion and its variances to add back to the fit so that our PM and
    parallax columns are absolute for every source. For sources without a reference motion: tractor ra/dec, zeros.
    """
    ra, dec = tractor['ra'][objid], tractor['dec'][objid]
    zero = {'pmra': 0., 'pmdec': 0., 'parallax': 0.}
    if not tractor['ref_epoch'][objid] > 0:
        return ra, dec, zero, dict(zero)
    pm = {k: float(tractor[k][objid]) for k in ('pmra', 'pmdec', 'parallax')}
    var = {k: (1 / float(tractor[k + '_ivar'][objid]) if tractor[k + '_ivar'][objid] > 0 else 0.)
           for k in ('pmra', 'pmdec', 'parallax')}
    ra0, dec0 = radec_at_mjd(ra, dec, float(tractor['ref_epoch'][objid]), pm['pmra'], pm['pmdec'], 0., ref_mjd)
    return float(np.atleast_1d(ra0)[0]), float(np.atleast_1d(dec0)[0]), pm, var


def create_brick_catalogue(filename, new_dir, old_dir, tractor_dir, original_data=False):
    logger.info(f"Starting PM catalogue generation for brick: {filename}")
    start_time = time.time()

    try:
        f = os.path.join(old_dir, filename)
        forced_table = fits.open(f)
        tractor_table = fits.open(tractor_dir + "tractor-forced-" + f[-13:])
        pm_table = tractor_table.copy()
    except Exception as e:
        logger.error(f"Failed to open FITS files for {filename}: {e}")
        raise

    pmra_plx = np.zeros(len(tractor_table[1].data))
    pmra_plx_ivar = np.zeros(len(tractor_table[1].data))
    pmdec_plx = np.zeros(len(tractor_table[1].data))
    pmdec_plx_ivar = np.zeros(len(tractor_table[1].data))
    parallax = np.zeros(len(tractor_table[1].data))
    parallax_ivar = np.zeros(len(tractor_table[1].data))
    new_ra_plx = np.zeros(len(tractor_table[1].data))
    new_ra_plx_ivar = np.zeros(len(tractor_table[1].data))
    new_dec_plx = np.zeros(len(tractor_table[1].data))
    new_dec_plx_ivar = np.zeros(len(tractor_table[1].data))
    pmra = np.zeros(len(tractor_table[1].data))
    pmra_ivar = np.zeros(len(tractor_table[1].data))
    pmdec = np.zeros(len(tractor_table[1].data))
    pmdec_ivar = np.zeros(len(tractor_table[1].data))
    new_ra = np.zeros(len(tractor_table[1].data))
    new_ra_ivar = np.zeros(len(tractor_table[1].data))
    new_dec = np.zeros(len(tractor_table[1].data))
    new_dec_ivar = np.zeros(len(tractor_table[1].data))
    pm_flag = np.zeros(len(tractor_table[1].data))
    ref_mjd = np.zeros(len(tractor_table[1].data))
    pm_n_epochs = np.zeros(len(tractor_table[1].data), dtype=np.int16)
    pm_baseline_days = np.zeros(len(tractor_table[1].data), dtype=np.float32)

    dra_ivar = forced_table[1].data.full_fit_dra_ivar
    ddec_ivar = forced_table[1].data.full_fit_ddec_ivar
    # Per-epoch total error = statistical (forced photometry) + systematic floor, in quadrature (error_model.py).
    # Use the total ivars the lateral-colour step wrote into the corrected forced file, so the data product is the
    # single source of the fit weights; fall back to computing them (older files, or original_data runs).
    fcols = forced_table[1].columns.names
    if 'full_fit_dra_ivar_total' in fcols and 'full_fit_ddec_ivar_total' in fcols:
        tot_ra = np.asarray(forced_table[1].data.full_fit_dra_ivar_total, dtype=float)
        tot_dec = np.asarray(forced_table[1].data.full_fit_ddec_ivar_total, dtype=float)
        with np.errstate(divide='ignore'):
            dra_error = np.where(tot_ra > 0, 1 / np.sqrt(np.where(tot_ra > 0, tot_ra, 1)), np.inf)
            ddec_error = np.where(tot_dec > 0, 1 / np.sqrt(np.where(tot_dec > 0, tot_dec, 1)), np.inf)
        sigma_sys = float(forced_table[1].header.get('SIGSYS', SIGMA_SYS_MAS))
    else:
        dra_error = total_error(dra_ivar)
        ddec_error = total_error(ddec_ivar)
        sigma_sys = SIGMA_SYS_MAS

    psf_mask = tractor_table[1].data.type == 'PSF'
    J = np.flatnonzero(psf_mask)
    pm_flag[~psf_mask] = -1  # not a PSF source, PM was never attempted

    # object-level quality flags from the Tractor catalogue (see the PMQ_* bits); epoch-based bits are set per object
    tdat = tractor_table[1].data
    pm_quality = np.zeros(len(tdat), dtype=np.int16)
    with np.errstate(all='ignore'):
        has_colour = (np.asarray(tdat.flux_g) > 0) & (np.asarray(tdat.flux_i) > 0)
    maskbits = np.asarray(tdat.maskbits).astype(np.int64)
    pm_quality[~has_colour] |= PMQ_NO_COLOUR
    pm_quality[(maskbits & MASKBITS_BRIGHT) != 0] |= PMQ_BRIGHT
    pm_quality[(maskbits & MASKBITS_SATURATED) != 0] |= PMQ_SATURATED
    pm_quality[~psf_mask] = 0   # no PM attempted: no quality to report

    # Plain numpy copies of the columns used per object (FITS_rec column access is slow), and each object's
    # measurement rows found once (a stable sort keeps them in file order) instead of a full-table mask per object.
    F = forced_table[1].data
    f_objid = np.asarray(F.objid)
    f_mjd = np.asarray(F.mjd, dtype=float)
    f_dra = np.asarray(F.full_fit_dra if original_data else F.lm_full_fit_dra)
    f_ddec = np.asarray(F.full_fit_ddec if original_data else F.lm_full_fit_ddec)
    f_raw_dra = np.asarray(F.full_fit_dra)
    # Only epochs the forced photometry actually measured: an error (ivar > 0) AND an offset. ~5% of epochs have
    # ivar > 0 but full_fit_dra == 0 (no offset measured); the correction steps already skip them, and after the
    # corrections they would carry fake offsets. Epochs with ivar == 0 carried zero weight but still counted towards
    # the epoch / baseline counts. Both are excluded, so FEW_EPOCHS / SHORT_BASELINE_DAYS count measured epochs only.
    measured = (np.asarray(F.full_fit_dra_ivar) > 0) & (np.asarray(F.full_fit_ddec_ivar) > 0) & (f_raw_dra != 0)
    order = np.flatnonzero(measured)[np.argsort(f_objid[measured], kind='stable')]
    uniq, starts = np.unique(f_objid[order], return_index=True)
    rows_of = dict(zip(uniq.tolist(), np.split(order, starts[1:])))
    no_rows = np.zeros(0, dtype=int)
    T = {k: np.asarray(tractor_table[1].data[k]) for k in
         ('ra', 'dec', 'ref_epoch', 'pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar')}

    fails = []
    n_insufficient_data = 0  # objects flagged FEW_EPOCHS or SHORT_BASELINE (unreliable time coverage)

    for i, objid in zip(J, tractor_table[1].data.objid[J]):
        rows = rows_of.get(int(objid), no_rows)
        n_ep = len(rows)
        base = float(f_mjd[rows].max() - f_mjd[rows].min()) if n_ep else 0.
        pm_n_epochs[i], pm_baseline_days[i] = n_ep, base
        if n_ep < FEW_EPOCHS:
            pm_quality[i] |= PMQ_FEW_EPOCHS
        if base < SHORT_BASELINE_DAYS:
            pm_quality[i] |= PMQ_SHORT_BASELINE
        if pm_quality[i] & (PMQ_FEW_EPOCHS | PMQ_SHORT_BASELINE):
            n_insufficient_data += 1
        try:
            X_plx, var_plx, X, var, rmjd = calculate_pm(
                f_mjd[rows], f_dra[rows], f_ddec[rows], dra_error[rows], ddec_error[rows], T['ra'][objid], T['dec'][objid]
            )
        except Exception as e:
            logger.warning(f"Error calculating PM for objid={objid}: {e}")
            fails.append(objid)
            pm_flag[i] = 0
            pm_quality[i] |= PMQ_FIT_FAILED
            continue

        success = True

        # reference motion the forced photometry already moved this source with (zero unless it is a Gaia star)
        ra0, dec0, ref_pm, ref_var = reference_motion(T, objid, rmjd)
        cosdec = np.cos(np.deg2rad(dec0))

        if X_plx is None:
            success = False
            dat = f_raw_dra[rows]
            if dat.size != 0 and sum(dat) != 0:
                fails.append(objid)
        else:
            dplx, dpmra, dpmdec, ra_offset, dec_offset = X_plx # in mas, mas/yr; relative to the reference motion
            parallax[i] = dplx + ref_pm['parallax']
            pmra_plx[i] = dpmra + ref_pm['pmra']
            pmdec_plx[i] = dpmdec + ref_pm['pmdec']
            new_ra_plx[i] = ra0 + ra_offset / 1000 / cosdec / 3600
            new_dec_plx[i] = dec0 + dec_offset / 1000 / 3600
            v_plx, v_pmra, v_pmdec, ra_offset_var, dec_offset_var = var_plx
            parallax_ivar[i] = 1 / (v_plx + ref_var['parallax'])
            pmra_plx_ivar[i] = 1 / (v_pmra + ref_var['pmra'])
            pmdec_plx_ivar[i] = 1 / (v_pmdec + ref_var['pmdec'])
            new_ra_plx_ivar[i] = 1 / (np.sqrt(ra_offset_var) / 1000 / cosdec / 3600)**2
            new_dec_plx_ivar[i] = 1 / (np.sqrt(dec_offset_var) / 1000 / 3600)**2

        if X is None:
            success = False
        else:
            dpmra, dpmdec, ra_offset, dec_offset = X # in mas, mas/yr; relative to the reference motion
            pmra[i] = dpmra + ref_pm['pmra']
            pmdec[i] = dpmdec + ref_pm['pmdec']
            new_ra[i] = ra0 + ra_offset / 1000 / cosdec / 3600
            new_dec[i] = dec0 + dec_offset / 1000 / 3600
            v_pmra, v_pmdec, ra_offset_var, dec_offset_var = var
            pmra_ivar[i] = 1 / (v_pmra + ref_var['pmra'])
            pmdec_ivar[i] = 1 / (v_pmdec + ref_var['pmdec'])
            new_ra_ivar[i] = 1 / (np.sqrt(ra_offset_var) / 1000 / cosdec / 3600)**2
            new_dec_ivar[i] = 1 / (np.sqrt(dec_offset_var) / 1000 / 3600)**2

        ref_mjd[i] = rmjd
        pm_flag[i] = 1 if success else 0  # only mark success when both fits (with & without parallax) succeeded
        if not success:
            pm_quality[i] |= PMQ_FIT_FAILED

    if len(J) > 0:
        frac_insufficient = n_insufficient_data / len(J)
        if frac_insufficient >= BRICK_WARN_INSUFFICIENT_FRACTION:
            logger.warning(
                f"{filename}: {n_insufficient_data}/{len(J)} PSF objects ({100 * frac_insufficient:.0f}%) "
                f"are flagged for too few epochs or too short a time baseline -- this brick likely "
                f"has inadequate temporal coverage for proper motion measurement, not just a handful of bad stars"
            )

    pm_table[1].columns.add_col(fits.Column(name="pmra_plx", array=pmra_plx, format='E', unit='mas/yr'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_plx_ivar", array=pmra_plx_ivar, format='E', unit='(mas/yr)^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_plx", array=pmdec_plx, format='E', unit='mas/yr'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_plx_ivar", array=pmdec_plx_ivar, format='E', unit='(mas/yr)^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="parallax_new", array=parallax, format='E', unit='mas'))
    pm_table[1].columns.add_col(fits.Column(name="parallax_new_ivar", array=parallax_ivar, format='E', unit='mas^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_plx", array=new_ra_plx, format='D', unit='deg'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_plx", array=new_dec_plx, format='D', unit='deg'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_plx_ivar", array=new_ra_plx_ivar, format='E', unit='deg^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_plx_ivar", array=new_dec_plx_ivar, format='E', unit='deg^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_no_plx", array=pmra, format='E', unit='mas/yr'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_no_plx_ivar", array=pmra_ivar, format='E', unit='(mas/yr)^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_no_plx", array=pmdec, format='E', unit='mas/yr'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_no_plx_ivar", array=pmdec_ivar, format='E', unit='(mas/yr)^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_no_plx", array=new_ra, format='D', unit='deg'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_no_plx", array=new_dec, format='D', unit='deg'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_no_plx_ivar", array=new_ra_ivar, format='E', unit='deg^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_no_plx_ivar", array=new_dec_ivar, format='E', unit='deg^(-2)'))
    pm_table[1].columns.add_col(fits.Column(name="pm_flag", array=pm_flag, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="ref_mjd", array=ref_mjd, format='D'))
    pm_table[1].columns.add_col(fits.Column(name="pm_quality", array=pm_quality, format='I'))
    pm_table[1].columns.add_col(fits.Column(name="pm_n_epochs", array=pm_n_epochs, format='I'))
    pm_table[1].columns.add_col(fits.Column(name="pm_baseline_days", array=pm_baseline_days, format='E', unit='d'))

    print("done! now writing to file...")

    output_name = f"tractor-{'original-' if original_data else ''}pm-{filename[-13:]}"
    # record the error model that produced these PMs
    pm_table[1].header['SIGSYS'] = (sigma_sys, 'systematic per-epoch error added in quadrature (mas)')
    output_path = f"{new_dir}{output_name}"
    tmp_path = output_path + ".tmp"
    try:
        pm_table.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, output_path)  # atomic: never leaves a truncated file at output_path
        logger.info(f"Saved PM catalogue to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing PM file: {e}")
        raise

    logger.info(f"Finished PM for {filename} in {time.time() - start_time:.2f} sec, failed on {len(fails)} object(s).")
    return fails


def _output_is_complete(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def create_catalogues(new_dir, old_dir, tractor_dir, original_data=False, cont=True):
    fails = []
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced':
            continue
        output_name = f"tractor-{'original-' if original_data else ''}pm-{filename[-13:]}"
        if cont is True and _output_is_complete(os.path.join(new_dir, output_name)):
            continue
        print(filename[-13:])
        try:
            fails.extend(create_brick_catalogue(filename, new_dir, old_dir, tractor_dir, original_data))
        except Exception as e:
            logger.error(f"Skipping brick {filename} due to error: {e}")
            continue
    return fails

    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        logger.error("Usage: <filename> <new_dir> <old_dir> <tractor_dir>")
        sys.exit(1)

    filename, new_dir, old_dir, tractor_dir = sys.argv[1:]
    logger.info(f"Running create_brick_catalogue for {filename}")
    create_brick_catalogue(filename, new_dir, old_dir, tractor_dir)