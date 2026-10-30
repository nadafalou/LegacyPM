"""Vetting tests for the white-dwarf candidates from the RPM pilot (see wd-vetting.ipynb).

Per-epoch refits use the pipeline's own fit (create_brick_catalogue.calculate_pm) on the fully corrected positions
(zp_full_fit_* after the exposure zero points if present, else lm_full_fit_*), the total errors
(full_fit_*_ivar_total) and their RA-Dec correlation (full_fit_radec_cov_total, if present), with the same epoch
selection, so a refit of all epochs reproduces the catalogue PM. Nothing here writes to the pipeline or the catalogue.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'py_files'))
import create_brick_catalogue as cbc   # noqa: E402

DAYS = cbc.DAYSPERYEAR


def measured(E):
    """Epochs the PM fit uses: an error and a measured offset (as create_brick_catalogue)."""
    return (E['full_fit_dra_ivar'] > 0) & (E['full_fit_ddec_ivar'] > 0) & (E['full_fit_dra'] != 0) \
        & (E['full_fit_dra_ivar_total'] > 0) & (E['full_fit_ddec_ivar_total'] > 0)


def _pos(E):
    """Positions the PM fit uses (arcsec), as create_brick_catalogue: zero-point stage if present, else lateral."""
    st = 'zp' if 'zp_full_fit_dra' in E.dtype.names else 'lm'
    return E[f'{st}_full_fit_dra'].astype(float), E[f'{st}_full_fit_ddec'].astype(float)


def _errors(E):
    sra = 1 / np.sqrt(E['full_fit_dra_ivar_total'])
    sdec = 1 / np.sqrt(E['full_fit_ddec_ivar_total'])
    rho = (E['full_fit_radec_cov_total'] / (sra * sdec) if 'full_fit_radec_cov_total' in E.dtype.names
           else np.zeros(len(E)))
    return sra, sdec, np.nan_to_num(rho)


def fit(E, ra, dec):
    """4-parameter fit of one object's epochs E. Returns dict pmra, pmdec, var_ra, var_dec, chi (whitened residuals,
    RA then Dec), n, baseline; None if no fit."""
    if len(E) < cbc.MIN_EPOCHS_FIT or np.ptp(E['mjd']) <= 0:
        return None
    sra, sdec, rho = _errors(E)
    dra, ddec = _pos(E)
    try:
        _, _, X, V, ref, _ = cbc.calculate_pm(E['mjd'].astype(float), dra, ddec, sra, sdec, ra, dec, rho)
    except Exception:
        return None
    if X is None:
        return None
    dt = (E['mjd'] - ref) / DAYS
    # residuals in units of sigma (X is mas, offsets are arcsec); per axis, ignoring the RA-Dec correlation
    rra = (dra - (X[0] * dt + X[2]) / 1000) / sra
    rdec = (ddec - (X[1] * dt + X[3]) / 1000) / sdec
    return dict(pmra=X[0], pmdec=X[1], var_ra=V[0], var_dec=V[1], chi=np.concatenate([rra, rdec]),
                rra=rra, rdec=rdec, n=len(E), baseline=np.ptp(E['mjd']))


def clipped_fit(E, ra, dec, nsig=5., n_iter=5):
    """Fit with iterative rejection of epochs > nsig from the fit in RA or Dec."""
    keep = np.ones(len(E), bool)
    f = None
    for _ in range(n_iter):
        f = fit(E[keep], ra, dec)
        if f is None:
            return None, keep
        bad = np.maximum(np.abs(f['rra']), np.abs(f['rdec'])) > nsig
        if not bad.any():
            break
        idx = np.flatnonzero(keep)
        keep[idx[bad]] = False
    return f, keep


def band_fits(E, ra, dec, min_epochs=5, min_baseline=730.):
    """Separate fits per filter (needs min_epochs and min_baseline days). {band: fit}."""
    out = {}
    for b in np.unique(np.char.strip(E['filter'].astype(str))):
        m = np.char.strip(E['filter'].astype(str)) == b
        if m.sum() >= min_epochs and np.ptp(E['mjd'][m]) >= min_baseline:
            f = fit(E[m], ra, dec)
            if f is not None:
                out[b] = f
    return out


def season_jackknife(E, ra, dec, ref):
    """Largest PM change (in units of the full-fit error) when one observing season (calendar year) is dropped."""
    yr = np.floor(2000 + (E['mjd'] - 51544.5) / DAYS).astype(int)
    worst = 0.
    for y in np.unique(yr):
        f = fit(E[yr != y], ra, dec)
        if f is None or f['baseline'] < 730:
            continue
        d = np.hypot((f['pmra'] - ref['pmra']) / np.sqrt(ref['var_ra']),
                     (f['pmdec'] - ref['pmdec']) / np.sqrt(ref['var_dec']))
        worst = max(worst, d)
    return worst


def mu_snr(pmra, pmdec, var_ra, var_dec):
    """Significance of the total PM with the proper propagation sigma_mu = sqrt(pmra^2 var_ra + pmdec^2 var_dec)/mu."""
    mu = np.hypot(pmra, pmdec)
    return mu / (np.sqrt(pmra**2 * var_ra + pmdec**2 * var_dec) / mu)


def distance_pc(m, M):
    return 10 ** ((m - M + 5) / 5)


def vtan_kms(mu_masyr, d_pc):
    return 4.74047 * mu_masyr / 1000 * d_pc
