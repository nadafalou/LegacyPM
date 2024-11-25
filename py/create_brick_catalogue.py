from astropy.io import fits
import numpy as np
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u
from astrometry.util.fits import os, fits_table
from legacypipe.py.legacypipe.survey import radec_at_mjd, mjd_to_year

DAYSPERYEAR = 365.2425

    
def pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, obj_filt, B):
    A = np.zeros((2 * N, 5))
    A[:N, 0] = plx_dra / dra_error[obj_filt] 
    A[N:, 0] = plx_ddec / ddec_error[obj_filt]
    A[:N, 1] = dt / dra_error[obj_filt]
    A[N:, 2] = dt / ddec_error[obj_filt]
    A[:N, 3] = np.ones(N)
    A[N:, 4] = np.ones(N)

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
    A[:N, 2] = np.ones(N)
    A[N:, 3] = np.ones(N)

    X, resid, rank, s = np.linalg.lstsq(A, B, rcond=None) # in arcsec
    X = X * 1000
    # pmra, pmdec, ra_offset, dec_offset = X[0], X[1], X[2] # mas, mas/yr, mas/yr

    cov_matrix = np.linalg.inv(A.T @ A)
    variances = np.diag(cov_matrix) * 1000**2
    # pmra_var, pmdec_var, ra_offset_var, dec_offset_var = variances
    
    return X, variances  # all in mas
    
    
def calculate_pm(forced, tractor, objid, dra_error, ddec_error):
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
    B = np.append((forced.lm_full_fit_dra / dra_error)[obj_filt], (forced.lm_full_fit_ddec / ddec_error)[obj_filt])
    
    try:
        X_plx, variances_plx = pm_lsq_plx(N, dt, plx_dra, plx_ddec, dra_error, ddec_error, obj_filt, B)
    except np.linalg.LinAlgError as err:
        X_plx, variances_plx = None, None

    try:
        X, variances = pm_lsq(N, dt, dra_error, ddec_error, obj_filt, B)
    except np.linalg.LinAlgError as err:
        X, variances = None, None
        
    return X_plx, variances_plx, X, variances, ref_mjd        
    
    
def create_brick_catalogue(filename, new_dir, old_dir, tractor_dir):
    f = os.path.join(old_dir, filename)
    forced_table = fits.open(f)
    tractor_table = fits.open(tractor_dir + "tractor-forced-" + f[-13:])
    pm_table = tractor_table.copy()

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

    J = np.flatnonzero(tractor_table[1].data.type == 'PSF')
    pm_flag[~J] = -1

    fails = []

    for i, objid in zip(J, tractor_table[1].data.objid[J]):
        X_plx, variances_plx, X, variances, rmjd = calculate_pm(forced_table[1].data, tractor_table[1].data, objid, dra_error, ddec_error)
        if X_plx is None:
            pm_flag[i] = 0
            dat = forced_table[1].data.full_fit_dra[forced_table[1].data.objid == objid]
            if dat.size != 0 and sum(dat) != 0:
                fails.append(objid)
        else:
            parallax[i], pmra_plx[i], pmdec_plx[i], ra_offset, dec_offset = X_plx # in mas, mas/yr
            new_ra_plx[i] = tractor_table[1].data.ra[objid] + ra_offset / 1000 / np.cos(np.deg2rad(tractor_table[1].data.dec[objid])) / 3600
            new_dec_plx[i] = tractor_table[1].data.dec[objid] + dec_offset / 1000 / 3600
            parallax_ivar[i], pmra_plx_ivar[i], pmdec_plx_ivar[i], ra_offset_ivar, dec_offset_ivar = 1 / variances_plx
            new_ra_plx_ivar[i] = 1 / ((1 / np.sqrt(ra_offset_ivar)) / 1000 / np.cos(np.deg2rad(tractor_table[1].data.dec[objid])) / 3600)**2
            new_dec_plx_ivar[i] = 1 / ((1 / np.sqrt(dec_offset_ivar)) / 1000 / 3600)

        if X is None:
            pm_flag[i] = 0
        else:
            pmra[i], pmdec[i], ra_offset, dec_offset = X # in mas, mas/yr
            new_ra[i] = tractor_table[1].data.ra[objid] + ra_offset / 1000 / np.cos(np.deg2rad(tractor_table[1].data.dec[objid])) / 3600
            new_dec[i] = tractor_table[1].data.dec[objid] + dec_offset / 1000 / 3600
            pmra_ivar[i], pmdec_ivar[i], ra_offset_ivar, dec_offset_ivar = 1 / variances
            new_ra_ivar[i] = 1 / ((1 / np.sqrt(ra_offset_ivar)) / 1000 / np.cos(np.deg2rad(tractor_table[1].data.dec[objid])) / 3600)**2
            new_dec_ivar[i] = 1 / ((1 / np.sqrt(dec_offset_ivar)) / 1000 / 3600)

        ref_mjd[i] = rmjd
        pm_flag[i] = 1  

    pm_table[1].columns.add_col(fits.Column(name="pmra_plx", array=pmra_plx, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_plx_ivar", array=pmra_plx_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_plx", array=pmdec_plx, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_plx_ivar", array=pmdec_plx_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="parallax_new", array=parallax, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="parallax_new_ivar", array=parallax_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_plx", array=new_ra_plx, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_plx", array=new_dec_plx, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_plx_ivar", array=new_ra_plx_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_plx_ivar", array=new_dec_plx_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_no_plx", array=pmra, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmra_no_plx_ivar", array=pmra_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_no_plx", array=pmdec, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pmdec_no_plx_ivar", array=pmdec_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_no_plx", array=new_ra, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_no_plx", array=new_dec, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_ra_no_plx_ivar", array=new_ra_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="new_dec_no_plx_ivar", array=new_dec_ivar, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="pm_flag", array=pm_flag, format='E'))
    pm_table[1].columns.add_col(fits.Column(name="ref_mjd", array=ref_mjd, format='E'))

    print("done! now writing to file...")

    pm_table.writeto(f"{ new_dir }tractor-pm-{ filename[-13:] }", overwrite=True)
    return fails
    
    
def create_catalogues(new_dir, old_dir, tractor_dir, cont=True):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced' or (cont is True and filename in os.listdir(new_dir)):
            continue
        print(filename[-13:])
        
        return create_brick_catalogue(filename, new_dir, old_dir, tractor_dir)
