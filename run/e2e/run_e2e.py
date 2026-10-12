"""End-to-end test of the new RM apply cut + smooth lateral model on held-out archive bricks.
Writes only under $LEGACYPM_E2E_DIR (default /pscratch/sd/n/nelfalou/e2e-test). Usage: python run_e2e.py
Then: python run_pm.py (only needed if the PM step has to be redone), python eval_e2e.py, python faint.py.
Results of the 2026-10-10 run are in PLAN.md (LM section, "E2E test")."""
import os, sys, glob, logging
import numpy as np, fitsio
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor

REPO = '/pscratch/sd/n/nelfalou/LegacyPM'
sys.path.insert(0, REPO)
from py_files import brick_ringmaps, brick_lateralmaps, create_brick_catalogue

ARCHIVE = '/pscratch/sd/n/nelfalou/archive/legacypm-catalogue'
HERE = os.environ.get('LEGACYPM_E2E_DIR', '/pscratch/sd/n/nelfalou/e2e-test')
DIRS = {k: os.path.join(HERE, k) + '/' for k in ('in', 'rm', 'lm', 'pm_new', 'pm_old')}
PM_COLS = ['pmra_plx', 'pmra_plx_ivar', 'pmdec_plx', 'pmdec_plx_ivar', 'parallax_new', 'parallax_new_ivar',
           'new_ra_plx', 'new_dec_plx', 'new_ra_plx_ivar', 'new_dec_plx_ivar', 'pmra_no_plx', 'pmra_no_plx_ivar',
           'pmdec_no_plx', 'pmdec_no_plx_ivar', 'new_ra_no_plx', 'new_dec_no_plx', 'new_ra_no_plx_ivar',
           'new_dec_no_plx_ivar', 'pm_flag', 'ref_mjd']


def strip_columns(src, dst, drop):
    with fits.open(src) as h:
        keep = [c for c in h[1].columns if c.name not in drop]
        hdu = fits.BinTableHDU.from_columns(fits.ColDefs(keep), header=h[1].header)
        fits.HDUList([h[0].copy(), hdu]).writeto(dst, overwrite=True)


def pick_bricks(n=8, seed=7):
    # exclude the 200 bricks the lateral model was fit on (same selection as run/lateral-model.ipynb)
    paths = sorted(p for d in ['060', '061', '062', '063'] for p in glob.glob(f'{ARCHIVE}/{d}/tractor-pm-*.fits')
                   if os.path.getsize(p) > 0)
    used = set(np.random.default_rng(42).choice(paths, 200, replace=False))
    cand = [p for p in paths if p not in used]
    rng = np.random.default_rng(seed)
    out = []
    for p in rng.permutation(cand):
        b = os.path.basename(p)[11:19]
        fp = os.path.join(os.path.dirname(p), f'forced-{b}.fits')
        try:
            F = fitsio.read(fp, columns=['objid', 'mjd', 'x', 'rm_full_fit_x'])
        except Exception:
            continue
        F = F[F['x'] != 0]
        if len(F) == 0:
            continue
        o = F['objid']; mn = np.full(o.max() + 1, np.inf); mx = np.full(o.max() + 1, -np.inf)
        np.minimum.at(mn, o, F['mjd']); np.maximum.at(mx, o, F['mjd'])
        base = (mx - mn)[np.isfinite(mn)]
        if np.median(base) >= 365:
            out.append(p)
        if len(out) == n:
            break
    return out


def run_brick(pm_path):
    logging.getLogger().setLevel(logging.WARNING)
    b = os.path.basename(pm_path)[11:19]
    fn = f'forced-{b}.fits'
    src_forced = os.path.join(os.path.dirname(pm_path), fn)
    # inputs: forced with DCR columns only; tractor table without the PM columns
    strip_columns(src_forced, DIRS['in'] + fn,
                  {c for c in fitsio.FITS(src_forced)[1].get_colnames() if c.startswith(('rm_', 'lm_'))})
    strip_columns(pm_path, DIRS['in'] + f'tractor-forced-{b}.fits', set(PM_COLS))
    # new pipeline: RM (new apply cut) -> LM (smooth model, new apply cut) -> PM
    rmaps = brick_ringmaps.get_ringmaps('/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/rm-corrected-forced-motions/ringmaps.fits')
    brick_ringmaps.create_brick_corrected_data(fn, DIRS['rm'], DIRS['in'], brick_ringmaps.make_splines(rmaps))
    brick_lateralmaps.create_brick_corrected_data(fn, DIRS['lm'], DIRS['rm'], DIRS['in'], brick_lateralmaps.load_lateral_model())
    create_brick_catalogue.create_brick_catalogue(fn, DIRS['pm_new'], DIRS['lm'], DIRS['in'])
    # old corrections (archive), same PM code
    create_brick_catalogue.create_brick_catalogue(fn, DIRS['pm_old'], os.path.dirname(pm_path) + '/', DIRS['in'])
    return b


if __name__ == '__main__':
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    bricks = pick_bricks()
    print('bricks:', [os.path.basename(p)[11:19] for p in bricks], flush=True)
    with ProcessPoolExecutor(len(bricks)) as ex:
        for b in ex.map(run_brick, bricks):
            print('done', b, flush=True)
