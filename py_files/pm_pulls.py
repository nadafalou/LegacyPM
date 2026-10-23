"""Per-epoch residuals from the PM fit, for testing the error model (pulls vs magnitude; run/pull-vs-mag.ipynb).

For every PSF object of a brick: the straight-line (4-parameter, no parallax) fit of the corrected offsets
lm_full_fit_dra/ddec over its measured epochs, weighted with the pipeline's errors (error_model.total_error). That
fit splits into independent weighted line fits in RA and in Dec, so it is done for all objects at once with bincounts.
Kept per epoch: the residuals (mas), the statistical errors (mas), the band and the object's magnitude in that band,
the number of epochs, the epoch's Dec, and whether the object is clean (pm_quality == 0 apart from BAD_EPOCHS) and a Gaia star.
Pulls for any error model sigma^2 = (k sigma_stat)^2 + sigma_sys^2 are then residual / sigma * sqrt(N / (N - 2)).

  python pm_pulls.py 000 --fraction 0.25 --workers 64     # writes <out-root>/combined/pulls-000-0.25.npz
"""
import os
import argparse
import numpy as np
import fitsio
from concurrent.futures import ProcessPoolExecutor

try:
    from .colours import g_minus_i, COLOUR_SOURCE_NONE
except ImportError:
    from colours import g_minus_i, COLOUR_SOURCE_NONE
try:
    from .error_model import total_error, corrected_ddec_ivar
except ImportError:
    from error_model import total_error, corrected_ddec_ivar

DEFAULT_OUT_ROOT = "/pscratch/sd/n/nelfalou/legacypm-catalogue"
EPOCH_COLS = ['brickid', 'objid', 'filter', 'mjd', 'dec', 'expnum', 'ccdname', 'full_fit_dra', 'full_fit_dra_ivar', 'full_fit_ddec_ivar',
              'lm_full_fit_dra', 'lm_full_fit_ddec']
OBJ_COLS = ['brickid', 'objid', 'type', 'ref_epoch', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'maskbits']
MASKBITS_BRIGHT = (1 << 1) | (1 << 11)
MASKBITS_SATURATED = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 14)
BANDS = 'griz'


def line_residuals(obj, t, d, err, n_obj):
    """Weighted straight-line fit d = a + b t per object (obj = index 0..n_obj-1); returns residuals d - a - b t."""
    w = 1 / err**2
    bc = lambda v: np.bincount(obj, v, minlength=n_obj)
    S, St, Stt, Sd, Std = bc(w), bc(w * t), bc(w * t * t), bc(w * d), bc(w * t * d)
    det = S * Stt - St**2
    with np.errstate(all='ignore'):
        b = (S * Std - St * Sd) / det
        a = (Sd - b * St) / S
    return d - a[obj] - b[obj] * t


def brick_residuals(args):
    deg_dir, brick, inpdate = args
    try:
        T = fitsio.read(os.path.join(deg_dir, f"tractor-pm-{brick}.fits"), columns=OBJ_COLS)
        F = fitsio.read(os.path.join(deg_dir, f"forced-{brick}.fits"), columns=EPOCH_COLS)
    except Exception as e:
        return None
    own_id = np.bincount(T['brickid'] - T['brickid'].min()).argmax() + T['brickid'].min()
    T = T[(T['brickid'] == own_id) & (np.char.strip(T['type'].astype(str)) == 'PSF')]
    F = F[(F['brickid'] == own_id) & (F['full_fit_dra_ivar'] > 0) & (F['full_fit_ddec_ivar'] > 0)
          & (F['full_fit_dra'] != 0)]
    if len(T) == 0 or len(F) == 0:
        return None
    srt = np.argsort(T['objid'])
    pos = np.clip(np.searchsorted(T['objid'][srt], F['objid']), 0, len(T) - 1)
    row = np.where(T['objid'][srt][pos] == F['objid'], srt[pos], -1)
    F, row = F[row >= 0], row[row >= 0]
    n_obj = len(T)
    n_ep = np.bincount(row, minlength=n_obj)
    first = np.full(n_obj, np.inf); last = np.full(n_obj, -np.inf)
    np.minimum.at(first, row, F['mjd']); np.maximum.at(last, row, F['mjd'])
    keep = n_ep[row] >= 3
    F, row = F[keep], row[keep]
    t = (F['mjd'] - np.bincount(row, F['mjd'], n_obj)[row] / n_ep[row]) / 365.25
    out = {}
    band = np.char.strip(F['filter'].astype(str))
    # statistical ivars as the pipeline uses them (Dec corrected for the cos^2(Dec) bug where the brick has it)
    ivar = {'ra': F['full_fit_dra_ivar'].astype(float)}
    hdr = fitsio.read_header(os.path.join(deg_dir, f"forced-{brick}.fits"), 1)   # the pipeline's record, if any
    ivar['dec'], cards = corrected_ddec_ivar(F['full_fit_dra_ivar'], F['full_fit_ddec_ivar'], F['dec'],
                                             inpdate or hdr.get('INPDATE'), hdr.get('DDIVFIX'))
    out['decfix'] = np.full(len(F), dict((k, v) for k, v, _ in cards)['DECFIX'])
    for ax, dcol in (('ra', 'lm_full_fit_dra'), ('dec', 'lm_full_fit_ddec')):
        out[f'res_{ax}'] = (line_residuals(row, t, F[dcol].astype(float), total_error(ivar[ax], band), n_obj) * 1000).astype(np.float32)
        out[f'sig_{ax}'] = (1000 / np.sqrt(ivar[ax])).astype(np.float32)
    flux = np.full(len(F), np.nan)
    for b in BANDS:
        m = band == b
        flux[m] = T[f'flux_{b}'][row[m]]
    with np.errstate(all='ignore'):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
    mb = T['maskbits'].astype(np.int64)
    has_colour = g_minus_i(T['flux_g'], T['flux_i'], T['flux_z'])[1] != COLOUR_SOURCE_NONE
    clean = ((n_ep >= 10) & (last - first >= 730) & has_colour
             & ((mb & (MASKBITS_BRIGHT | MASKBITS_SATURATED)) == 0))
    out['expnum'] = F['expnum'].astype(np.int32)
    out['dec'] = F['dec'].astype(np.float32)
    out.update(band=np.array([BANDS.find(b) for b in band], dtype=np.int8), mag=mag.astype(np.float32),
               n=n_ep[row].astype(np.int16), clean=clean[row], gaia=(T['ref_epoch'] > 0)[row],
               obj=(np.int64(own_id) << 32 | T['objid'][row].astype(np.int64)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("degree")
    ap.add_argument("--fraction", type=float, default=0.25, help="random fraction of bricks")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--input-date", default=None, help="YYYY-MM-DD the inputs were written, for outputs made before "
                    "brick_DCR recorded INPDATE (e.g. 2025-01-01 for CFS inputs: Dec-ivar bug corrected)")
    args = ap.parse_args()
    deg_dir = os.path.join(args.out_root, args.degree)
    bricks = sorted(f[11:19] for f in os.listdir(deg_dir) if f.startswith("tractor-pm-"))
    rng = np.random.default_rng(1)
    bricks = sorted(rng.choice(bricks, max(1, int(args.fraction * len(bricks))), replace=False))
    parts = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(brick_residuals, [(deg_dir, b, args.input_date) for b in bricks], chunksize=2):
            if r is not None:
                parts.append(r)
    res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    path = os.path.join(args.out_root, "combined", f"pulls-{args.degree}-{args.fraction}.npz")
    np.savez(path, **res)
    print(f"{len(parts)} bricks, {len(res['mag']):,} epochs -> {path}")


if __name__ == "__main__":
    main()
