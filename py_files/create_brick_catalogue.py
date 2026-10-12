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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DAYSPERYEAR = 365.2425

# A handful of stars in the archived catalogues came back with |PM| up to
# ~1e16 mas/yr -- numerically "successful" (no LinAlgError) but physically
# nonsense. Traced this to specific real cases (e.g. brick 0628p105, objid
# 1499): NOT a matrix-conditioning problem -- cond(A) for these fits is
# perfectly ordinary, and the reported variance is honestly large, matching
# the wild point estimate. The actual cause is that *every* object in that
# brick has a time baseline of ~36 days (vs. ~1200-1800 days for a normal,
# well-covered brick, checked across several) -- nowhere near enough leverage
# to constrain a mas/yr-level linear trend, so the least-squares point
# estimate is essentially noise amplified by a tiny baseline, occasionally
# landing "significant" by chance. This is a data-completeness issue for that
# brick (probably imaged in only one short observing run so far), not a
# per-star fluke, so it needs a baseline requirement, not just an epoch count.
#   MIN_EPOCHS: below this there's no way to meaningfully constrain a linear
#     trend (or a trend + parallax) against per-point noise.
#   MIN_BASELINE_DAYS: below this, even many epochs can't meaningfully
#     constrain a mas/yr-level trend -- chosen well inside the large gap
#     observed between pathological (~30-50d) and normal (~1200-1800d)
#     bricks, but worth revisiting if a future brick falls in between.
#   MAX_COND_NUMBER: kept as a cheap backstop against true numerical
#     singularity (duplicate/near-duplicate mjd values), though it did not
#     catch the case above -- the baseline check is what actually matters.
# None of these are derived from first principles -- worth sanity-checking
# against more known-good vs. known-bad stars as more data comes back from
# Dustin.
MIN_EPOCHS = 5
MIN_BASELINE_DAYS = 180
MAX_COND_NUMBER = 1e8

# Some bricks have *every* object fail the epoch/baseline check above (e.g.
# only one short observing run covers that patch of sky so far) -- the
# catalogue file for such a brick is technically valid but scientifically
# empty. Warn loudly rather than let that pass silently as just another
# "finished" brick; see the per-brick log line in create_brick_catalogue.
BRICK_WARN_INSUFFICIENT_FRACTION = 0.5


def pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, obj_filt, B):
    A = np.zeros((2 * N, 5))
    A[:N, 0] = plx_dra / dra_error[obj_filt]
    A[N:, 0] = plx_ddec / ddec_error[obj_filt]
    A[:N, 1] = dt / dra_error[obj_filt]
    A[N:, 2] = dt / ddec_error[obj_filt]
    A[:N, 3] = 1 / dra_error[obj_filt]  # offset columns must be weighted like the data (B = d/sigma)
    A[N:, 4] = 1 / ddec_error[obj_filt]

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


def pm_lsq(N, dt, dra_error, ddec_error, obj_filt, B):
    A = np.zeros((2 * N, 4))
    A[:N, 0] = dt / dra_error[obj_filt]
    A[N:, 1] = dt / ddec_error[obj_filt]
    A[:N, 2] = 1 / dra_error[obj_filt]  # offset columns must be weighted like the data (B = d/sigma)
    A[N:, 3] = 1 / ddec_error[obj_filt]

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


def calculate_pm(forced, tractor, objid, dra_error, ddec_error, original_data=False):
    ref_year = 0 # bc dra, ddec around 0, not actual ra, decs
    obj_filt = forced.objid == objid

    plx_ra, plx_dec = [], []
    for date in forced.mjd[obj_filt]:
        ra, dec = radec_at_mjd(tractor.ra[objid], tractor.dec[objid], ref_year, 0, 0, 1000, date) # mas/ye and mas, returns deg
        plx_ra.append(ra)
        plx_dec.append(dec)
    plx_dra, plx_ddec = (np.array(plx_ra) - tractor.ra[objid]) * np.cos(np.deg2rad(tractor.dec[objid])) * 3600, (np.array(plx_dec) - tractor.dec[objid]) * 3600 # change in ra, dec due to parallax = 1 arcsec
    dra_ivar = 1 / dra_error ** 2
    ref_mjd = np.sum((dra_ivar * forced.mjd)[obj_filt]) / np.sum(dra_ivar[obj_filt]) # ref_mjd for this dataset is different from Gaia's ref_mjd
                                                                                     # theoretically, dra_ivar ~= ddec_ivar
    dt = (forced.mjd[obj_filt] - ref_mjd) / DAYSPERYEAR
    N = len(forced.full_fit_dra[obj_filt])
    if original_data:
        B = np.append((forced.full_fit_dra / dra_error)[obj_filt], (forced.full_fit_ddec / ddec_error)[obj_filt])
    else:
        B = np.append((forced.lm_full_fit_dra / dra_error)[obj_filt], (forced.lm_full_fit_ddec / ddec_error)[obj_filt])

    mjd_obj = forced.mjd[obj_filt]
    baseline_days = mjd_obj.max() - mjd_obj.min() if N > 0 else 0
    if N < MIN_EPOCHS or baseline_days < MIN_BASELINE_DAYS:
        return None, None, None, None, ref_mjd

    try:
        X_plx, variances_plx = pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, obj_filt, B)
    except np.linalg.LinAlgError as err:
        X_plx, variances_plx = None, None

    try:
        X, variances = pm_lsq(N, dt, dra_error, ddec_error, obj_filt, B)
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
    ra, dec = tractor.ra[objid], tractor.dec[objid]
    zero = {'pmra': 0., 'pmdec': 0., 'parallax': 0.}
    if not tractor.ref_epoch[objid] > 0:
        return ra, dec, zero, dict(zero)
    pm = {k: float(tractor[k][objid]) for k in ('pmra', 'pmdec', 'parallax')}
    var = {k: (1 / float(tractor[k + '_ivar'][objid]) if tractor[k + '_ivar'][objid] > 0 else 0.)
           for k in ('pmra', 'pmdec', 'parallax')}
    ra0, dec0 = radec_at_mjd(ra, dec, float(tractor.ref_epoch[objid]), pm['pmra'], pm['pmdec'], 0., ref_mjd)
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

    dra_ivar = forced_table[1].data.full_fit_dra_ivar
    ddec_ivar = forced_table[1].data.full_fit_ddec_ivar
    dra_error = 1 / np.sqrt(dra_ivar)
    ddec_error = 1 / np.sqrt(ddec_ivar)
    dra_error = np.hypot(dra_error, 0.014806002)
    ddec_error = np.hypot(ddec_error, 0.014806002)
    # dra_ivar = 1 / dra_error ** 2 
    # ddec_ivar = 1 / ddec_error ** 2 

    psf_mask = tractor_table[1].data.type == 'PSF'
    J = np.flatnonzero(psf_mask)
    pm_flag[~psf_mask] = -1  # not a PSF source, PM was never attempted

    fails = []
    n_insufficient_data = 0  # objects where calculate_pm rejected both fits for having too few epochs/too short a baseline (see MIN_EPOCHS/MIN_BASELINE_DAYS) -- distinct from a per-object numerical failure (an exception)

    for i, objid in zip(J, tractor_table[1].data.objid[J]):
        try:
            X_plx, var_plx, X, var, rmjd = calculate_pm(
                forced_table[1].data, tractor_table[1].data, objid, dra_error, ddec_error, original_data
            )
        except Exception as e:
            logger.warning(f"Error calculating PM for objid={objid}: {e}")
            fails.append(objid)
            pm_flag[i] = 0
            continue

        if X_plx is None and X is None:
            n_insufficient_data += 1

        success = True

        # reference motion the forced photometry already moved this source with (zero unless it is a Gaia star)
        ra0, dec0, ref_pm, ref_var = reference_motion(tractor_table[1].data, objid, rmjd)
        cosdec = np.cos(np.deg2rad(dec0))

        if X_plx is None:
            success = False
            dat = forced_table[1].data.full_fit_dra[forced_table[1].data.objid == objid]
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

    if len(J) > 0:
        frac_insufficient = n_insufficient_data / len(J)
        if frac_insufficient >= BRICK_WARN_INSUFFICIENT_FRACTION:
            logger.warning(
                f"{filename}: {n_insufficient_data}/{len(J)} PSF objects ({100 * frac_insufficient:.0f}%) "
                f"had no usable PM fit (too few epochs or too short a time baseline) -- this brick likely "
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

    print("done! now writing to file...")

    output_name = f"tractor-{'original-' if original_data else ''}pm-{filename[-13:]}"
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