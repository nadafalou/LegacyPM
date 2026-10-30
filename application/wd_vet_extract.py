"""Extract what the white-dwarf candidate vetting needs from the v2 catalogues (read-only). Heavy I/O: run on a compute
node, e.g.  srun -A desi -q interactive -C cpu -N 1 -t 00:30:00 python wd_vet_extract.py

Inputs : output/hpm_mu20_n200.fits (from rpm-diagram.ipynb; candidates = wd_candidate & not Gaia)
Outputs: output/wd_vet_cands.fits   catalogue rows of the candidates (many columns)
         output/wd_vet_epochs.fits  their per-epoch measurements (fully corrected positions, as the PM fit uses them)
         output/wd_vet_gaia.fits    clean Gaia PSF stars in the same bricks (Gaia parallax, our parallax_new) --
                                    for the subdwarf absolute-magnitude relation and the parallax zero-point check
         output/wd_vet_gaiahpm_epochs.fits  per-epoch data of clean Gaia stars with mu > 20 mas/yr and r > 20: real
                                    movers near the candidates' magnitudes, to calibrate the astrometric tests
         output/wd_vet_redhpm.fits / wd_vet_redhpm_epochs.fits  non-Gaia HPM objects on the red sequence (g - z > 2,
                                    r > 21, same HPM cuts incl. S/N_r > 30): real movers per the PM-direction test, at the candidates'
                                    magnitudes -- the magnitude-matched calibration of the astrometric tests
         output/wd_vet_control.fits clean non-Gaia PSF objects, r 21-23, g - z < 1.5, no significant PM -- mostly
                                    distant stars and QSOs, true parallax ~ 0: the parallax control sample
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import fitsio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rpm

OUT = os.path.join(HERE, 'output')
CAT_COLS = rpm.COLUMNS + ['brickid', 'parallax', 
    'flux_w1', 'flux_w2', 'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_i', 'mw_transmission_i', 'mw_transmission_w1',
    'mw_transmission_w2', 'parallax_ivar',
    'pmra_plx', 'pmdec_plx', 'pmra_plx_ivar', 'pmdec_plx_ivar', 'pm_n_bad_epochs', 'ref_mjd',
    'fracflux_g', 'fracflux_r', 'fracflux_z', 'nobs_g', 'nobs_r', 'nobs_z', 'rchisq_g', 'rchisq_r', 'rchisq_z',
    'dchisq']
EPOCH_COLS = ['brickid', 'objid', 'mjd', 'filter', 'expnum', 'ccdname', 'airmass', 'psfsize', 'exptime',
              'full_fit_dra', 'full_fit_ddec', 'full_fit_dra_ivar', 'full_fit_ddec_ivar',
              'lm_full_fit_dra', 'lm_full_fit_ddec', 'full_fit_dra_ivar_total', 'full_fit_ddec_ivar_total',
              'flux', 'flux_ivar', 'fracmasked', 'rchisq', 'ra', 'dec']
EPOCH_COLS_NEW = ['zp_full_fit_dra', 'zp_full_fit_ddec', 'full_fit_radec_cov_total']   # error model v3 (redo-errors)


def _s(a):
    return np.char.strip(np.asarray(a).astype(str))


def one_brick(args):
    brick, objids = args
    root = os.path.join(rpm.CATALOGUE_ROOT, brick[:3])
    T = fitsio.read(os.path.join(root, f'tractor-pm-{brick}.fits'), columns=CAT_COLS)
    d = rpm.derive(T)
    psf = (_s(T['brickname']) == brick) & T['brick_primary'] & (_s(T['type']) == 'PSF') & (T['pm_quality'] == 0) & (T['pm_flag'] == 1)
    own = _s(T['brickname']) == brick   # Tractor files can end with neighbouring bricks' rows (repeated objids)
    cand = np.isin(T['objid'], objids) & psf
    gaia = psf & d['gaia'] & (T['parallax_ivar'] > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = d['mu'] / d['mu_err']
    ctrl = (psf & ~d['gaia'] & (d['r'] > 21) & (d['r'] < 23) & (d['g_z'] < 1.5) & np.isfinite(d['z'])
            & (snr < 3) & (T['parallax_new_ivar'] > 0))
    # per-epoch rows of the candidates (grouped on brickid, objid as in create_brick_catalogue)
    fn = os.path.join(root, f'forced-{brick}.fits')
    key = fitsio.read(fn, columns=['brickid', 'objid'])
    gaia_hpm = gaia & (d['mu'] > 20) & (d['r'] > 20)
    red_hpm = rpm.hpm_mask(T, d) & ~d['gaia'] & (d['r'] > 21) & (d['g_z'] > 2) & own
    rows = np.flatnonzero(np.isin(key['objid'], T['objid'][cand | gaia_hpm | red_hpm]))
    have = fitsio.FITS(fn)[1].get_colnames()
    cols = EPOCH_COLS + [c for c in EPOCH_COLS_NEW if c in have]
    if len(cols) != len(EPOCH_COLS + EPOCH_COLS_NEW):
        raise RuntimeError(f'{brick}: forced file is not on the current error model (no zp_/cov columns)')
    E = fitsio.read(fn, columns=cols, rows=rows) if len(rows) else None
    if E is not None:   # keep only this brick's sources (forced files can hold neighbouring bricks' epochs)
        E = E[E['brickid'] == T['brickid'][own][0]]
        EG = E[np.isin(E['objid'], T['objid'][gaia_hpm])]
        ER = E[np.isin(E['objid'], T['objid'][red_hpm])]
        E = E[np.isin(E['objid'], T['objid'][cand])]
    else:
        EG = ER = None
    extra = lambda m: {'g': d['g'][m], 'r': d['r'][m], 'z': d['z'][m], 'g_z': d['g_z'][m], 'mu': d['mu'][m],
                       'mu_err': d['mu_err'][m], 'H_r': d['H_r'][m]}
    return brick, (T[cand], extra(cand)), E, (T[gaia], extra(gaia)), (T[ctrl], extra(ctrl)), EG, (T[red_hpm], extra(red_hpm)), ER


def _stack(parts):
    T = np.concatenate([p[0] for p in parts])
    out = {k: T[k] for k in T.dtype.names}
    for k in parts[0][1]:
        out[k] = np.concatenate([p[1][k] for p in parts])
    return out


def main():
    H = fitsio.read(os.path.join(OUT, 'hpm_mu20_n200.fits'))
    gaia = _s(H['ref_cat']) == 'GE'
    C = H[H['wd_candidate'] & ~gaia]
    bricks = _s(C['brickname'])
    jobs = [(b, C['objid'][bricks == b]) for b in np.unique(bricks)]
    print(f'{len(C)} candidates in {len(jobs)} bricks', flush=True)
    with Pool(min(32, len(jobs))) as pool:
        res = pool.map(one_brick, jobs)
    cands = _stack([r[1] for r in res])
    epochs = np.concatenate([r[2] for r in res if r[2] is not None])
    gaia_t = _stack([r[3] for r in res])
    ctrl = _stack([r[4] for r in res])
    print(f'{len(cands["objid"])} candidates, {len(epochs)} epochs; {len(gaia_t["objid"])} Gaia stars; '
          f'{len(ctrl["objid"])} control objects', flush=True)
    for name, data in [('cands', cands), ('gaia', gaia_t), ('control', ctrl)]:
        fitsio.write(os.path.join(OUT, f'wd_vet_{name}.fits'), data, names=list(data), clobber=True)
    fitsio.write(os.path.join(OUT, 'wd_vet_epochs.fits'), epochs, clobber=True)
    eg = np.concatenate([r[5] for r in res if r[5] is not None])
    fitsio.write(os.path.join(OUT, 'wd_vet_gaiahpm_epochs.fits'), eg, clobber=True)
    red = _stack([r[6] for r in res])
    fitsio.write(os.path.join(OUT, 'wd_vet_redhpm.fits'), red, names=list(red), clobber=True)
    er = np.concatenate([r[7] for r in res if r[7] is not None])
    fitsio.write(os.path.join(OUT, 'wd_vet_redhpm_epochs.fits'), er, clobber=True)
    print(f'{len(red["objid"])} red non-Gaia HPM stars, {len(er)} epochs', flush=True)
    print(f'{len(np.unique(eg["brickid"].astype(np.int64) * 100000 + eg["objid"]))} Gaia HPM stars with epochs', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
