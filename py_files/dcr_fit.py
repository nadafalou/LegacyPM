"""Build the DCR calibration sample from the combined Gaia-star files and fit DCR models to it.

The quantity brick_DCR.py corrects: the offset of each Gaia-star epoch along the parallactic direction (before any
correction), as a function of x = c_rel * tan z, with c_rel = (g - i) - (g - i)_ref of the CCD/exposure and
(g - i)_ref = -0.56 + 1.34 * (BP - RP of the astrometric reference stars on that CCD).
Same definitions and selection as run/dcr-plots.ipynb (F3), but without brick_DCR's 14 GB dictionary: the CCD
reference colour is looked up with a sorted-key search, so tens of millions of epochs fit in memory.
"""
import os
import functools
import numpy as np
import fitsio

from . import dcr_geometry
from .pm_analysis import COMBINED_DIR, _degree_list

CCD_TABLE = '/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/survey-ccds-decam-dr10-bprp-kd.fits'
PIXSCALE = 262.   # mas per pixel
EPOCH_COLS = ['brickid', 'objid', 'filter', 'full_fit_dra_ivar', 'full_fit_x', 'full_fit_y', 'x', 'y',
              'ra', 'dec', 'mjd', 'exptime', 'expnum', 'ccdname']


def _key(expnum, ccd_index):
    return expnum.astype(np.int64) * 256 + ccd_index


@functools.lru_cache(maxsize=1)
def ccd_reference_colour():
    """Sorted (keys, (g-i)_ref, ccdname -> index map) for every DR10 CCD/exposure."""
    C = fitsio.read(CCD_TABLE, columns=['expnum', 'ccdname', 'ccdbprp'])
    names = np.char.strip(C['ccdname'].astype(str))
    uniq = np.unique(names)
    index = {n: i for i, n in enumerate(uniq)}
    keys = _key(C['expnum'], np.searchsorted(uniq, names))
    order = np.argsort(keys)
    return keys[order], (-0.56 + 1.34 * C['ccdbprp'].astype(float))[order], index


def load_sample(degrees=('000-063',), brick_fraction=0.25, seed=1, combined_dir=COMBINED_DIR, max_sigma_mas=10.):
    """Gaia-star epochs with sigma < max_sigma_mas from a random fraction of bricks in each slice.
    Returns a dict of arrays: band, x (= c_rel tan z), y (offset along the parallactic direction, mas), c_rel,
    tanz, brickid (for splitting train/test by brick), airmass."""
    keys, giref, ccd_index = ccd_reference_colour()
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ('band', 'x', 'y', 'c_rel', 'tanz', 'brickid')}
    for deg in _degree_list(degrees):
        fe = os.path.join(combined_dir, f'gaia-epochs-{deg}.fits')
        fo = os.path.join(combined_dir, f'gaia-objects-{deg}.fits')
        if not (os.path.exists(fe) and os.path.exists(fo)):
            continue
        F = fitsio.read(fe, columns=EPOCH_COLS)
        bricks = np.unique(F['brickid'])
        keep = rng.choice(bricks, max(1, int(round(brick_fraction * len(bricks)))), replace=False)
        F = F[np.isin(F['brickid'], keep) & (F['full_fit_dra_ivar'] > 1 / (max_sigma_mas / 1000.)**2)   # ivar in arcsec^-2
              & (F['full_fit_x'] != F['x'])]
        T = fitsio.read(fo, columns=['brickid', 'objid', 'flux_g', 'flux_i'])
        # (brickid, objid) join
        tkey = (T['brickid'].astype(np.int64) << 32) | T['objid'].astype(np.int64)
        fkey = (F['brickid'].astype(np.int64) << 32) | F['objid'].astype(np.int64)
        srt = np.argsort(tkey)
        pos = np.clip(np.searchsorted(tkey[srt], fkey), 0, len(T) - 1)
        row = np.where(tkey[srt][pos] == fkey, srt[pos], -1)
        F, row = F[row >= 0], row[row >= 0]
        with np.errstate(all='ignore'):
            gi = -2.5 * np.log10(T['flux_g'][row] / T['flux_i'][row])
        # CCD reference colour
        names = np.char.strip(F['ccdname'].astype(str))
        ci = np.array([ccd_index.get(n, -1) for n in np.unique(names)])
        cidx = ci[np.searchsorted(np.unique(names), names)]
        k = _key(F['expnum'], cidx)
        p = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        ref = np.where((keys[p] == k) & (cidx >= 0), giref[p], np.nan)
        c_rel = gi - ref
        # geometry, exactly as BrickDCR: airmass and parallactic angle from the exposure time and position
        ha = dcr_geometry.hour_angle(F['ra'], F['mjd'] + 0.5 * F['exptime'] / 86400.)
        am, pa = dcr_geometry.parallactic(F['dec'], ha)
        tanz = np.sqrt(np.clip(am**2 - 1, 0, None))
        dx, dy = F['full_fit_x'] - F['x'], F['full_fit_y'] - F['y']
        y = (dx * np.cos(pa) - dy * np.sin(pa)) * PIXSCALE
        ok = np.isfinite(c_rel) & (np.abs(c_rel) < 3) & (am >= 1) & np.isfinite(y)
        out['band'].append(np.char.strip(F['filter'].astype(str))[ok])
        out['x'].append((c_rel * tanz)[ok]); out['y'].append(y[ok]); out['c_rel'].append(c_rel[ok])
        out['tanz'].append(tanz[ok]); out['brickid'].append(F['brickid'][ok])
    return {k: np.concatenate(v) for k, v in out.items()}


def binned_medians(x, y, edges, min_count=200):
    """(centre = median x, median y, error on the median, count) in each bin with >= min_count points."""
    idx = np.digitize(x, edges)
    xc, ym, ye, n = [], [], [], []
    for k in range(1, len(edges)):
        m = idx == k
        if m.sum() < min_count:
            continue
        v = y[m]
        med = np.median(v)
        xc.append(np.median(x[m])); ym.append(med); n.append(m.sum())
        ye.append(1.2533 * 1.4826 * np.median(np.abs(v - med)) / np.sqrt(m.sum()))
    return np.array(xc), np.array(ym), np.array(ye), np.array(n)


def fit_poly(xc, ym, ye, order):
    """Polynomial in x fitted to binned medians, weighted by their errors. Returns coefficients (highest power
    first, as np.polyval)."""
    return np.polyfit(xc, ym, order, w=1 / ye)


def binned_2d(c_rel, tanz, y, c_edges, t_edges, min_count=100):
    """Medians of y in (c_rel, tan z) cells with >= min_count epochs: (median c_rel, median tan z, median y, error)."""
    ic, it = np.digitize(c_rel, c_edges), np.digitize(tanz, t_edges)
    cell = ic * (len(t_edges) + 1) + it
    order = np.argsort(cell)
    cell_s = cell[order]
    starts = np.flatnonzero(np.r_[True, cell_s[1:] != cell_s[:-1]])
    out = []
    for a, b in zip(starts, np.r_[starts[1:], len(cell_s)]):
        if b - a < min_count:
            continue
        j = order[a:b]
        med = np.median(y[j])
        out.append((np.median(c_rel[j]), np.median(tanz[j]), med,
                    1.2533 * 1.4826 * np.median(np.abs(y[j] - med)) / np.sqrt(b - a)))
    return np.array(out).T


def fit_tanz_poly(c, t, ym, ye, order):
    """Fit y = tan z * sum_k a_k c_rel^k (k = 0..order) to cell medians, weighted. Returns a (highest power first)."""
    A = np.stack([t * c**k for k in range(order, -1, -1)], axis=1)
    w = 1 / ye
    return np.linalg.lstsq(A * w[:, None], ym * w, rcond=None)[0]


def predict(model, c, t):
    """model = ('x', coef) for a polynomial in c_rel tan z, or ('tanz', coef) for tan z * polynomial in c_rel."""
    kind, coef = model
    return np.polyval(coef, c * t) if kind == 'x' else t * np.polyval(coef, c)


# --- Two-line (piecewise-linear) models, suggested by James (meeting notes, 2026-10) ---------------------------------

def _wlstsq(A, y, e):
    w = 1 / e
    coef = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    return coef, np.sum(((A @ coef - y) * w)**2)


def fit_two_lines(xc, ym, ye, breaks):
    """Two independent straight lines, for x < x_b and x >= x_b, with x_b chosen from `breaks` to minimise the
    weighted chi^2 on these medians. Returns ('two', (x_b, (a_lo, b_lo), (a_hi, b_hi))), y = a + b x on each side."""
    best = None
    for xb in breaks:
        lo, hi = xc < xb, xc >= xb
        if lo.sum() < 3 or hi.sum() < 3:
            continue
        clo, chi_lo = _wlstsq(np.stack([np.ones(lo.sum()), xc[lo]], 1), ym[lo], ye[lo])
        chi_, chi_hi = _wlstsq(np.stack([np.ones(hi.sum()), xc[hi]], 1), ym[hi], ye[hi])
        if best is None or chi_lo + chi_hi < best[0]:
            best = (chi_lo + chi_hi, (xb, tuple(clo), tuple(chi_)))
    return ('two', best[1])


def fit_broken_line(xc, ym, ye, breaks):
    """Continuous broken line: y = a + b_lo min(x - x_b, 0) + b_hi max(x - x_b, 0), x_b from `breaks`.
    Returns ('broken', (x_b, a, b_lo, b_hi))."""
    best = None
    for xb in breaks:
        A = np.stack([np.ones_like(xc), np.minimum(xc - xb, 0), np.maximum(xc - xb, 0)], 1)
        coef, chi = _wlstsq(A, ym, ye)
        if best is None or chi < best[0]:
            best = (chi, (xb,) + tuple(coef))
    return ('broken', best[1])


def fit_tanz_broken(c, t, ym, ye, breaks):
    """y = tan z * broken line in c_rel (break c_b from `breaks`), fitted to (c_rel, tan z) cell medians.
    Returns ('tanz_broken', (c_b, a, b_lo, b_hi))."""
    best = None
    for cb in breaks:
        A = t[:, None] * np.stack([np.ones_like(c), np.minimum(c - cb, 0), np.maximum(c - cb, 0)], 1)
        coef, chi = _wlstsq(A, ym, ye)
        if best is None or chi < best[0]:
            best = (chi, (cb,) + tuple(coef))
    return ('tanz_broken', best[1])


def eval_piecewise(model, x):
    """Evaluate a 'two' or 'broken' model at x."""
    kind, p = model
    if kind == 'two':
        xb, (a0, b0), (a1, b1) = p
        return np.where(x < xb, a0 + b0 * x, a1 + b1 * x)
    xb, a, blo, bhi = p
    return a + blo * np.minimum(x - xb, 0) + bhi * np.maximum(x - xb, 0)


def predict_any(model, c, t):
    """predict() extended to the piecewise models: 'two'/'broken' are functions of x = c_rel tan z,
    'tanz_broken' is tan z times a broken line in c_rel."""
    kind = model[0]
    if kind in ('x', 'tanz'):
        return predict(model, c, t)
    if kind in ('two', 'broken'):
        return eval_piecewise(model, c * t)
    cb, a, blo, bhi = model[1]
    return t * (a + blo * np.minimum(c - cb, 0) + bhi * np.maximum(c - cb, 0))


# --- PLAN H2: offsets after each correction stage, split along / across the parallactic direction ------------------

STAGES = ['', 'dcr_', 'rm_', 'lm_']   # raw forced photometry, + DCR, + CCD maps, + lateral colour


def load_stage_sample(degrees=('000',), brick_fraction=1.0, seed=1, combined_dir=COMBINED_DIR, max_sigma_mas=10.):
    # also returns 'clean' (star-level quality, pm_quality == 0 equivalent), 'G' and 'sig_stat' (statistical error, mas)
    """Gaia-star epochs (sigma < max_sigma_mas, valid fit) with, for every correction stage, the offset from Gaia
    along (par_<stage>) and across (perp_<stage>) the parallactic direction, in mas, plus band, c_rel and tan z.
    Stage keys: 'raw', 'dcr', 'rm', 'lm', and 'zp' (exposure-CCD zero points) when the files have zp_full_fit_*."""
    keys, giref, ccd_index = ccd_reference_colour()
    rng = np.random.default_rng(seed)
    cols = (EPOCH_COLS + ['full_fit_dra'] + [f'{s}full_fit_{a}' for s in STAGES[1:] for a in 'xy'])
    out = {}
    for deg in _degree_list(degrees):
        fe = os.path.join(combined_dir, f'gaia-epochs-{deg}.fits')
        fo = os.path.join(combined_dir, f'gaia-objects-{deg}.fits')
        if not (os.path.exists(fe) and os.path.exists(fo)):
            continue
        have = set(fitsio.FITS(fe)[1].get_colnames())
        zp_cols = ['lm_full_fit_dra', 'lm_full_fit_ddec', 'zp_full_fit_dra', 'zp_full_fit_ddec']
        has_zp = all(c in have for c in zp_cols)
        F = fitsio.read(fe, columns=cols + (zp_cols if has_zp else []))
        if brick_fraction < 1:
            bricks = np.unique(F['brickid'])
            F = F[np.isin(F['brickid'], rng.choice(bricks, max(1, int(brick_fraction * len(bricks))), replace=False))]
        F = F[(F['full_fit_dra_ivar'] > 1 / (max_sigma_mas / 1000.)**2) & (F['full_fit_dra'] != 0)]
        T = fitsio.read(fo, columns=['brickid', 'objid', 'flux_g', 'flux_i', 'maskbits', 'n_epochs', 'baseline_days',
                                     'pm_flag', 'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag',
                                     'gaia_phot_rp_mean_mag', 'pmra', 'pmdec', 'ra', 'dec']
                    + (['pm_quality'] if 'pm_quality' in fitsio.FITS(fo)[1].get_colnames() else []))
        tkey = (T['brickid'].astype(np.int64) << 32) | T['objid'].astype(np.int64)
        fkey = (F['brickid'].astype(np.int64) << 32) | F['objid'].astype(np.int64)
        srt = np.argsort(tkey)
        pos = np.clip(np.searchsorted(tkey[srt], fkey), 0, len(T) - 1)
        row = np.where(tkey[srt][pos] == fkey, srt[pos], -1)
        F, row = F[row >= 0], row[row >= 0]
        with np.errstate(all='ignore'):
            gi = -2.5 * np.log10(T['flux_g'][row] / T['flux_i'][row])
        mb = T['maskbits'][row].astype(np.int64)
        # star-level "clean" (pm_quality == 0 equivalent, see pm_analysis.quality_bits)
        if 'pm_quality' in T.dtype.names:   # the catalogue's own quality bits (all flags, incl. BAD_EPOCHS, colour fallback)
            clean = (T['pm_quality'][row] == 0) & (T['pm_flag'][row] == 1)
        else:
            clean = ((T['n_epochs'][row] >= 10) & (T['baseline_days'][row] >= 730) & np.isfinite(gi) & (T['pm_flag'][row] == 1)
                     & ((mb & ((1 << 1) | (1 << 11) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 14))) == 0))
        names = np.char.strip(F['ccdname'].astype(str))
        un = np.unique(names)
        cidx = np.array([ccd_index.get(n, -1) for n in un])[np.searchsorted(un, names)]
        k = _key(F['expnum'], cidx)
        p = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        c_rel = gi - np.where((keys[p] == k) & (cidx >= 0), giref[p], np.nan)
        ha = dcr_geometry.hour_angle(F['ra'], F['mjd'] + 0.5 * F['exptime'] / 86400.)
        am, pa = dcr_geometry.parallactic(F['dec'], ha)
        d = {'band': np.char.strip(F['filter'].astype(str)), 'c_rel': c_rel,
             'tanz': np.sqrt(np.clip(am**2 - 1, 0, None)), 'brickid': F['brickid'], 'clean': clean,
             'G': T['gaia_phot_g_mean_mag'][row], 'sig_stat': 1000 / np.sqrt(F['full_fit_dra_ivar']),
             'pa': pa, 'dec': F['dec'],   # parallactic angle (rad, from north through east) and Dec of the epoch
             'objid': F['objid'], 'bp_rp': T['gaia_phot_bp_mean_mag'][row] - T['gaia_phot_rp_mean_mag'][row],
             'gaia_pm': np.hypot(T['pmra'][row], T['pmdec'][row]), 'star_ra': T['ra'][row], 'star_dec': T['dec'][row]}
        for s, name in zip(STAGES, ['raw', 'dcr', 'rm', 'lm']):
            dx = F[f'{s}full_fit_x'] - F['x']
            dy = F[f'{s}full_fit_y'] - F['y']
            d[f'par_{name}'] = (dx * np.cos(pa) - dy * np.sin(pa)) * PIXSCALE
            d[f'perp_{name}'] = (dx * np.sin(pa) + dy * np.cos(pa)) * PIXSCALE
        if has_zp:   # zero-point stage: the lm_ pixel offsets shifted by the zero point (CCD x = -Dec, y = +RA)
            # dDec = -0.262 dx  ->  dx = -dDec / 0.262 (sign fixed 2026-10-28: it doubled the Dec zero point)
            dx = F['lm_full_fit_x'] - F['x'] - (F['zp_full_fit_ddec'] - F['lm_full_fit_ddec']) * 1000 / PIXSCALE
            dy = F['lm_full_fit_y'] - F['y'] + (F['zp_full_fit_dra'] - F['lm_full_fit_dra']) * 1000 / PIXSCALE
            d['par_zp'] = (dx * np.cos(pa) - dy * np.sin(pa)) * PIXSCALE
            d['perp_zp'] = (dx * np.sin(pa) + dy * np.cos(pa)) * PIXSCALE
        for kk, v in d.items():
            out.setdefault(kk, []).append(v)
    return {kk: np.concatenate(v) for kk, v in out.items()}
