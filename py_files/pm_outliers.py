"""PLAN H1: the vertical feature in the PM-vs-Gaia plot (F7) -- Gaia stars whose PM from our fit is far from Gaia's.

For Gaia stars the forced photometry measures offsets from Gaia's predicted position, so the corrected offsets
(lm_full_fit_dra/ddec) of each epoch are residuals from Gaia, and our fitted PM minus Gaia's is the trend in them.
This module builds, per Gaia star, the catalogue properties plus summaries of its epochs, flags outliers, and
refits the PM with and without rejecting outlying epochs (same weighting as create_brick_catalogue: statistical
error plus the 14.8 mas floor in quadrature, reference epoch = weighted mean MJD).
"""
import os
import numpy as np
import fitsio

from .pm_analysis import COMBINED_DIR, _degree_list, _MAX_SANE_PM_MAS_PER_YR

FLOOR_MAS = 14.806002
DAYS_PER_YEAR = 365.2425
OBJ_COLS = ['brickid', 'brickname', 'objid', 'ra', 'dec', 'type', 'ref_cat', 'gaia_phot_g_mean_mag',
            'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'pmra', 'pmdec',
            'parallax', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar', 'pmra_no_plx', 'pmdec_no_plx', 'pmra_no_plx_ivar',
            'pmdec_no_plx_ivar', 'pm_flag', 'ref_mjd', 'n_epochs', 'baseline_days', 'objid_clash', 'maskbits',
            'fitbits', 'nobs_g', 'nobs_r', 'nobs_i', 'nobs_z']
EPOCH_COLS = ['brickid', 'objid', 'filter', 'mjd', 'airmass', 'x', 'y', 'dqmask', 'fracin', 'full_fit_dra',
              'full_fit_dra_ivar', 'full_fit_ddec_ivar', 'full_fit_x', 'full_fit_y', 'dcr_full_fit_x',
              'dcr_full_fit_y', 'rm_full_fit_x', 'rm_full_fit_y', 'lm_full_fit_x', 'lm_full_fit_y',
              'lm_full_fit_dra', 'lm_full_fit_ddec']


def _key(brickid, objid):
    return (np.asarray(brickid).astype(np.int64) << 32) | np.asarray(objid).astype(np.int64)


def star_sample(T, threshold=30.):
    """Mask of Gaia stars in the F7 selection, and the outlier flag |ours - Gaia| > threshold (mas/yr)."""
    ok = ((np.char.strip(T['ref_cat'].astype(str)) == 'GE') & (np.char.strip(T['type'].astype(str)) == 'PSF')
          & (T['pm_flag'] == 1) & (T['pmra_ivar'] > 0) & (T['pmdec_ivar'] > 0) & (T['parallax_ivar'] > 0)
          & ~T['objid_clash'].astype(bool)
          & (np.abs(T['pmra_no_plx']) < _MAX_SANE_PM_MAS_PER_YR) & (np.abs(T['pmdec_no_plx']) < _MAX_SANE_PM_MAS_PER_YR))
    dpm = np.hypot(T['pmra_no_plx'] - T['pmra'], T['pmdec_no_plx'] - T['pmdec'])
    return ok, dpm, ok & (dpm > threshold)


def _fit_trend(t, y, w):
    """Weighted straight line y = a + b t; returns slope b and its variance."""
    sw, st, sy = w.sum(), (w * t).sum(), (w * y).sum()
    stt, sty = (w * t * t).sum(), (w * t * y).sum()
    det = sw * stt - st * st
    if det <= 0:
        return np.nan, np.nan
    return (sw * sty - st * sy) / det, sw / det


def refit(mjd, dra, ddec, sig, clip=None, max_iter=5):
    """Our 4-parameter PM fit of one star (RA and Dec trends are independent), optionally with iterative rejection of
    epochs whose residual from the fit exceeds `clip` sigma (on either axis). Offsets and sigma in mas. Returns
    (dpmra, dpmdec, n_used) in mas/yr -- relative to Gaia for Gaia stars."""
    keep = np.ones(len(mjd), bool)
    for _ in range(max_iter if clip else 1):
        if keep.sum() < 5:
            return np.nan, np.nan, int(keep.sum())
        w = 1 / sig[keep]**2
        ref = np.sum(w * mjd[keep]) / w.sum()
        t = (mjd - ref) / DAYS_PER_YEAR
        bra, _ = _fit_trend(t[keep], dra[keep], w)
        bde, _ = _fit_trend(t[keep], ddec[keep], w)
        if not clip:
            break
        # residuals about the fitted lines (offsets fitted as weighted means at the reference epoch)
        ara = np.sum(w * (dra[keep] - bra * t[keep])) / w.sum()
        ade = np.sum(w * (ddec[keep] - bde * t[keep])) / w.sum()
        bad = (np.abs(dra - ara - bra * t) > clip * sig) | (np.abs(ddec - ade - bde * t) > clip * sig)
        new_keep = ~bad
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return bra, bde, int(keep.sum())


def build_table(degrees=('000-007',), threshold=30., control_per_outlier=3, seed=0, combined_dir=COMBINED_DIR,
                clip=5.):
    """Per-star table of outliers and a random control sample of non-outliers (control_per_outlier x as many; None =
    all stars of the F7 selection),
    with catalogue properties, epoch summaries and PM refits (no rejection / `clip`-sigma rejection)."""
    rng = np.random.default_rng(seed)
    rows = []
    for deg in _degree_list(degrees):
        fo = os.path.join(combined_dir, f'gaia-objects-{deg}.fits')
        fe = os.path.join(combined_dir, f'gaia-epochs-{deg}.fits')
        if not (os.path.exists(fo) and os.path.exists(fe)):
            continue
        T = fitsio.read(fo, columns=OBJ_COLS)
        ok, dpm, out = star_sample(T, threshold)
        if control_per_outlier is None:          # every star of the F7 selection
            pick = np.flatnonzero(ok)
        else:
            ctrl_pool = np.flatnonzero(ok & ~out)
            ctrl = rng.choice(ctrl_pool, min(len(ctrl_pool), control_per_outlier * out.sum()), replace=False)
            pick = np.concatenate([np.flatnonzero(out), ctrl])
        T, dpm, is_out = T[pick], dpm[pick], out[pick]
        F = fitsio.read(fe, columns=EPOCH_COLS)
        fk, tk = _key(F['brickid'], F['objid']), _key(T['brickid'], T['objid'])
        F = F[np.isin(fk, tk)]
        fk = _key(F['brickid'], F['objid'])
        order = np.argsort(fk, kind='stable')
        F, fk = F[order], fk[order]
        uk, start = np.unique(fk, return_index=True)
        bounds = dict(zip(uk.tolist(), zip(start, np.r_[start[1:], len(fk)])))
        for i in range(len(T)):
            a, b = bounds.get(int(tk[i]), (0, 0))
            E = F[a:b]
            valid = (E['full_fit_dra_ivar'] > 0) & (E['full_fit_dra'] != 0)
            E = E[valid]
            n = len(E)
            row = {k: T[k][i] for k in OBJ_COLS}
            row.update(deg=deg, is_outlier=bool(is_out[i]), dpm_total=dpm[i],
                       dpmra=T['pmra_no_plx'][i] - T['pmra'][i], dpmdec=T['pmdec_no_plx'][i] - T['pmdec'][i])
            if n:
                sig = np.hypot(1000 / np.sqrt(E['full_fit_dra_ivar']), FLOOR_MAS)   # mas, as in the PM fit
                dra, dde = E['lm_full_fit_dra'] * 1000, E['lm_full_fit_ddec'] * 1000
                chi = np.maximum(np.abs(dra), np.abs(dde)) / sig
                filt = np.char.strip(E['filter'].astype(str))
                years = np.floor((E['mjd'] - 51544.5) / DAYS_PER_YEAR)
                edge = (E['x'] < 50) | (E['x'] > 1998) | (E['y'] < 50) | (E['y'] > 4046)
                col = lambda p, ax: E[(p + '_' if p else '') + 'full_fit_' + ax]
                shift = lambda p, q: np.hypot(col(p, 'x') - col(q, 'x'), col(p, 'y') - col(q, 'y')) * 262.
                rr, rd, nr = refit(E['mjd'], dra, dde, sig)
                cr, cd, nc = refit(E['mjd'], dra, dde, sig, clip=clip)
                row.update(n_valid=n, chi_max=chi.max(), n_chi_gt5=int((chi > 5).sum()), frac_chi_gt5=(chi > 5).mean(),
                           resid_rms_mas=np.sqrt(np.mean(dra**2 + dde**2) / 2), n_bands=len(set(filt)),
                           n_years=len(set(years)), mjd_first=E['mjd'].min(), mjd_last=E['mjd'].max(),
                           frac_edge=edge.mean(), frac_dqmask=(E['dqmask'] != 0).mean(), min_fracin=E['fracin'].min(),
                           airmass_max=E['airmass'].max(), airmass_med=np.median(E['airmass']),
                           dcr_shift_max=shift('dcr', '').max(), rm_shift_max=shift('rm', 'dcr').max(),
                           lm_shift_max=shift('lm', 'rm').max(),
                           refit_dpmra=rr, refit_dpmdec=rd, clip_dpmra=cr, clip_dpmdec=cd, clip_n_used=nc,
                           clip_n_rejected=n - nc)
            rows.append(row)
    return rows


def to_columns(rows):
    """List of per-star dicts -> dict of numpy arrays (missing keys -> NaN)."""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in rows]
        out[k] = np.array(vals)
    return out


def refit_variants(deg, combined_dir=COMBINED_DIR, clip=5.):
    """Refit every Gaia star of the F7 selection in one slice in several ways, to test fixes for the outliers:
      'pipeline'   all epochs with ivar > 0 (what create_brick_catalogue does -- should reproduce the catalogue)
      'nozero'     also drop epochs with full_fit_dra == 0 (no offset measured, but ivar > 0)
      'nozero_clip' as 'nozero', plus iterative `clip`-sigma rejection of outlying epochs
    Returns the catalogue table rows (selection), and a dict variant -> (dpmra, dpmdec) relative to Gaia."""
    T = fitsio.read(os.path.join(combined_dir, f'gaia-objects-{deg}.fits'), columns=OBJ_COLS)
    ok, _, _ = star_sample(T)
    T = T[ok]
    F = fitsio.read(os.path.join(combined_dir, f'gaia-epochs-{deg}.fits'),
                    columns=['brickid', 'objid', 'mjd', 'full_fit_dra', 'full_fit_dra_ivar', 'lm_full_fit_dra', 'lm_full_fit_ddec'])
    F = F[F['full_fit_dra_ivar'] > 0]
    fk = _key(F['brickid'], F['objid'])
    order = np.argsort(fk, kind='stable')
    F, fk = F[order], fk[order]
    uk, start = np.unique(fk, return_index=True)
    bounds = dict(zip(uk.tolist(), zip(start, np.r_[start[1:], len(fk)])))
    tk = _key(T['brickid'], T['objid'])
    res = {v: np.full((len(T), 2), np.nan) for v in ('pipeline', 'nozero', 'nozero_clip')}
    for i in range(len(T)):
        a, b = bounds.get(int(tk[i]), (0, 0))
        E = F[a:b]
        if len(E) == 0:
            continue
        sig = np.hypot(1000 / np.sqrt(E['full_fit_dra_ivar']), FLOOR_MAS)
        dra, dde = E['lm_full_fit_dra'] * 1000, E['lm_full_fit_ddec'] * 1000
        res['pipeline'][i] = refit(E['mjd'], dra, dde, sig)[:2]
        nz = E['full_fit_dra'] != 0
        if nz.sum() >= 5:
            res['nozero'][i] = refit(E['mjd'][nz], dra[nz], dde[nz], sig[nz])[:2]
            res['nozero_clip'][i] = refit(E['mjd'][nz], dra[nz], dde[nz], sig[nz], clip=clip)[:2]
    return T, res
