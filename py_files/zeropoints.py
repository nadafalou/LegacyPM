"""Per-exposure, per-CCD astrometric zero points measured from Gaia stars (decided 2026-10-26, Nada).

After the DCR, CCD-map and lateral-colour corrections, all stars on one CCD in one exposure still share a common offset
from their Gaia-predicted positions: a "zero point" of 2 mas (RA) and 4 mas (Dec) rms, larger in g, mostly varying from
CCD to CCD within an exposure (atmospheric turbulence coherent over arcminutes, plus residual errors of each CCD's
astrometric solution), and drifting slowly with time (mean -0.2 mas/yr in RA, +0.1 mas/yr in Dec: a fake proper
motion common to every star). Shared by all stars in a field, it made the PMs of a brick's stars err together
(~0.5 mas/yr) and produced a mean PM offset from Gaia. Diagnostic: legacypm-catalogue-v2/diag/zeropoints.py.

The correction: for each (expnum, ccdname), the median of the corrected offsets (lm_full_fit_dra/ddec) of bright Gaia
stars (sigma < 10 mas, measured, dqmask == 0, with a full Gaia astrometric solution, not BRIGHT/SATURATED) is subtracted
from every object on that CCD in that exposure. With fewer than N_MIN such stars, the exposure's median zero point is
used (if at least EXP_MIN_CCDS of its CCDs have one), else no correction. The table also holds zero points from two
disjoint halves of the Gaia stars (by a hash of brickid, objid) for validation without circularity.

  python zeropoints.py build [--workers 32]      # compute node; reads all v2 combined gaia-epochs / gaia-objects
"""
import os
import sys
import argparse
import numpy as np
import fitsio

DEFAULT_TABLE = '/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/zeropoints/zeropoints-dr10-v2.fits'
N_MIN = 10
EXP_MIN_CCDS = 3
MASKBITS_BAD = (1 << 1) | (1 << 11) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 14)   # BRIGHT, MEDIUM, SATUR_*
_PREFIX = {'N': 0, 'S': 1, 'FN': 2, 'FS': 3}


def ccd_code(ccdname):
    """Integer code for DECam CCD names (N1-N31, S1-S31, FN1-4, FS1-4): prefix * 100 + number."""
    names = np.char.strip(np.asarray(ccdname).astype(str))
    out = np.full(names.shape, -1, dtype=np.int64)
    for p, c in _PREFIX.items():
        m = np.char.startswith(names, p) & np.char.isdigit(np.char.lstrip(names, 'NFS'))
        if p in ('N', 'S'):
            m &= ~np.char.startswith(names, 'F')
        num = np.char.lstrip(names[m], 'NFS')
        out[m] = c * 100 + num.astype(np.int64)
    return out


def exposure_ccd_key(expnum, ccdname):
    return np.asarray(expnum).astype(np.int64) * 1000 + ccd_code(ccdname)


def _group_median(key, v):
    o = np.lexsort((v, key))
    ks, vs = key[o], v[o]
    st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    n = np.diff(np.r_[st, len(ks)])
    return ks[st], 0.5 * (vs[st + (n - 1) // 2] + vs[st + n // 2]), n


def _read_slice(args):
    combined_dir, deg = args
    fe, fo = (os.path.join(combined_dir, f'gaia-{k}-{deg}.fits') for k in ('epochs', 'objects'))
    if not (os.path.exists(fe) and os.path.exists(fo)):
        return None
    F = fitsio.read(fe, columns=['brickid', 'objid', 'expnum', 'ccdname', 'full_fit_dra', 'full_fit_dra_ivar',
                                 'lm_full_fit_dra', 'lm_full_fit_ddec', 'dqmask'])
    F = F[(F['full_fit_dra_ivar'] > 1e4) & (F['full_fit_dra'] != 0) & (F['dqmask'] == 0)]
    T = fitsio.read(fo, columns=['brickid', 'objid', 'maskbits', 'pmra_ivar', 'parallax_ivar'])
    good = ((T['maskbits'].astype(np.int64) & MASKBITS_BAD) == 0) & (T['pmra_ivar'] > 0) & (T['parallax_ivar'] > 0)
    tk = (T['brickid'].astype(np.int64) << 32) | T['objid'].astype(np.int64)
    fk = (F['brickid'].astype(np.int64) << 32) | F['objid'].astype(np.int64)
    s = np.argsort(tk)
    pos = np.clip(np.searchsorted(tk[s], fk), 0, len(tk) - 1)
    ok = (tk[s][pos] == fk) & good[s][pos]
    F, fk = F[ok], fk[ok]
    half = ((fk * 2654435761) >> 7) & 1
    return (exposure_ccd_key(F['expnum'], F['ccdname']), (F['lm_full_fit_dra'] * 1000).astype(np.float32),
            (F['lm_full_fit_ddec'] * 1000).astype(np.float32), half.astype(np.int8))


def _zp_with_fallback(key, med_ra, med_de, n):
    """CCD zero point where n >= N_MIN; else the exposure median of its CCDs with n >= N_MIN (if >= EXP_MIN_CCDS)."""
    good = n >= N_MIN
    exp = key // 1000
    ue, ie = np.unique(exp[good], return_inverse=True)
    _, e_ra, ne = _group_median(exp[good], med_ra[good])
    _, e_de, _ = _group_median(exp[good], med_de[good])
    ok_e = ne >= EXP_MIN_CCDS
    j = np.searchsorted(ue, exp)
    has_e = (j < len(ue)) & (ue[np.clip(j, 0, len(ue) - 1)] == exp) & ok_e[np.clip(j, 0, len(ue) - 1)]
    zr = np.where(good, med_ra, np.where(has_e, e_ra[np.clip(j, 0, len(ue) - 1)], 0.))
    zd = np.where(good, med_de, np.where(has_e, e_de[np.clip(j, 0, len(ue) - 1)], 0.))
    src = np.where(good, 1, np.where(has_e, 2, 0)).astype(np.int8)   # 1 CCD, 2 exposure, 0 none
    return zr, zd, src


def build(combined_dir, out_path, workers=32, degrees=None):
    from concurrent.futures import ProcessPoolExecutor
    degrees = degrees or [f'{d:03d}' for d in range(64)]
    with ProcessPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(_read_slice, [(combined_dir, d) for d in degrees]) if p is not None]
    key = np.concatenate([p[0] for p in parts]); ra = np.concatenate([p[1] for p in parts])
    de = np.concatenate([p[2] for p in parts]); half = np.concatenate([p[3] for p in parts]); del parts
    print(f'{len(key):,} bright Gaia-star epochs', flush=True)
    cols = {}
    u, mra, n = _group_median(key, ra); _, mde, _ = _group_median(key, de)
    cols['key'], cols['n'] = u, n.astype(np.int32)
    cols['zp_ra'], cols['zp_dec'], cols['zp_src'] = _zp_with_fallback(u, mra, mde, n)
    for h, lab in ((0, 'a'), (1, 'b')):
        m = half == h
        uh, hra, nh = _group_median(key[m], ra[m]); _, hde, _ = _group_median(key[m], de[m])
        j = np.searchsorted(uh, u); hit = (j < len(uh)) & (uh[np.clip(j, 0, len(uh) - 1)] == u)
        jj = np.clip(j, 0, len(uh) - 1)
        nn = np.where(hit, nh[jj], 0); r_ = np.where(hit, hra[jj], 0.); d_ = np.where(hit, hde[jj], 0.)
        cols[f'n_{lab}'] = nn.astype(np.int32)
        cols[f'zp_ra_{lab}'], cols[f'zp_dec_{lab}'], cols[f'zp_src_{lab}'] = _zp_with_fallback(u, r_, d_, nn)
    cols['expnum'] = (u // 1000).astype(np.int64)
    out = np.zeros(len(u), dtype=[(k, np.asarray(v).dtype) for k, v in cols.items()])
    for k, v in cols.items():
        out[k] = v
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    fitsio.write(tmp, out, clobber=True, header={'NMIN': N_MIN, 'EXPMIN': EXP_MIN_CCDS})
    os.replace(tmp, out_path)
    print(f'{len(u):,} exposure-CCDs -> {out_path}; source: CCD {np.mean(out["zp_src"] == 1):.1%}, exposure '
          f'{np.mean(out["zp_src"] == 2):.1%}, none {np.mean(out["zp_src"] == 0):.1%}', flush=True)


_TABLE = {}


def table_id(path=DEFAULT_TABLE):
    """Short ID of the zero-point table (file size + mtime), recorded in headers (ZPTABLE); '' if no table."""
    if not os.path.exists(path):
        return ''
    st = os.stat(path)
    return f'{int(st.st_mtime) % 10**8:08d}{st.st_size % 1000:03d}'


def zp_values(D, path=DEFAULT_TABLE):
    """({column: array}, header cards) for the zero-point-corrected positions of a corrected forced table D (any
    record array with expnum, ccdname, lm_full_fit_dra/ddec): zp_full_fit_* = lm_full_fit_* - zero point (arcsec).
    No table: ({}, [])."""
    if not os.path.exists(path):
        return {}, []
    zr, zd = lookup(D['expnum'], D['ccdname'], path=path)
    vals = {'zp_full_fit_dra': (D['lm_full_fit_dra'] - zr / 1000.).astype(np.float32),
            'zp_full_fit_ddec': (D['lm_full_fit_ddec'] - zd / 1000.).astype(np.float32)}
    cards = [('ZPTABLE', table_id(path), 'exposure-CCD zero-point table ID (zeropoints.py)'),
             ('ZPCORR', True, 'zp_full_fit_* = lm_full_fit_* - exposure-CCD zero point')]
    return vals, cards


def add_zp_columns(hdul, path=DEFAULT_TABLE):
    """Write zp_full_fit_dra/ddec (arcsec) into an astropy HDUList of a corrected forced file (add or overwrite; the
    table HDU is rebuilt when columns are added), plus header cards. No table: nothing is written."""
    from astropy.io import fits
    vals, cards = zp_values(hdul[1].data, path)
    if not vals:
        return
    names = hdul[1].columns.names
    for k, v in vals.items():
        if k in names:
            hdul[1].data[k] = v
    new = [fits.Column(name=k, format='E', unit='arcsec', array=v) for k, v in vals.items() if k not in names]
    if new:
        hdr = hdul[1].header.copy()
        hdul[1] = fits.BinTableHDU.from_columns(hdul[1].columns + fits.ColDefs(new), header=hdr)
    for key, val, comment in cards:
        hdul[1].header[key] = (val, comment)


def lookup(expnum, ccdname, which='', path=DEFAULT_TABLE):
    """Zero points (RA*, Dec) in mas to subtract from each epoch's offsets; which = '' (all Gaia stars), 'a' or 'b'
    (halves, for validation). Exposure-CCDs not in the table get 0."""
    sfx = f'_{which}' if which else ''
    if (path, sfx) not in _TABLE:   # only the columns needed (the full table is ~170 MB; workers share it after fork)
        _TABLE[(path, sfx)] = fitsio.read(path, columns=['key', f'zp_ra{sfx}', f'zp_dec{sfx}'])
    Z = _TABLE[(path, sfx)]
    k = exposure_ccd_key(expnum, ccdname)
    j = np.clip(np.searchsorted(Z['key'], k), 0, len(Z) - 1)
    hit = Z['key'][j] == k
    return np.where(hit, Z[f'zp_ra{sfx}'][j], 0.), np.where(hit, Z[f'zp_dec{sfx}'][j], 0.)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['build'])
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', default=DEFAULT_TABLE)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from py_files.pm_analysis import COMBINED_DIR
    build(COMBINED_DIR, args.out, args.workers)
