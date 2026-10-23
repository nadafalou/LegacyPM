"""Combine the per-brick pipeline output of whole RA slices into a few files per slice for analysis and figures.
Reading ~80k tractor-pm / forced files in a notebook is too slow, so this reads them once and writes, per slice,
into <out-root>/combined/:

  pm-<deg>.fits           PSF objects with a successful PM fit (pm_flag == 1): identifiers, positions, fluxes, Gaia
                          columns, our PM/parallax columns, plus n_epochs / baseline_days (epochs with a valid fit) and
                          objid_clash (see read_brick; drop these, ~0.005%). Only each brick's own sources are kept.
  gaia-objects-<deg>.fits all Gaia stars (ref_cat GE), same columns, any pm_flag
  gaia-epochs-<deg>.fits  per-epoch forced-photometry rows of those Gaia stars, raw + DCR/RM/LM-corrected positions
  bricks-<deg>.fits       one row per brick: centre, object counts, median epochs and baseline

Only bricks with a finished tractor-pm file are included; rerun after more bricks finish (files are overwritten).
  python combine_catalogue.py 000-063 --workers 32
Runs in the desi-pm env or the Shifter image (needs fitsio + numpy only).
"""
import os
import sys
import time
import argparse
import numpy as np
import fitsio
from concurrent.futures import ProcessPoolExecutor

DEFAULT_OUT_ROOT = "/pscratch/sd/n/nelfalou/legacypm-catalogue"

OBJ_COLS = ['brickid', 'brickname', 'objid', 'type', 'ra', 'dec', 'ra_ivar', 'dec_ivar', 'ref_cat', 'ref_id',
            'ref_epoch', 'flux_g', 'flux_r', 'flux_i', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_i',
            'flux_ivar_z', 'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_i', 'mw_transmission_z',
            'nobs_g', 'nobs_r', 'nobs_i', 'nobs_z', 'maskbits', 'fitbits',
            'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag',
            'pmra', 'pmdec', 'parallax', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar',
            'pmra_plx', 'pmra_plx_ivar', 'pmdec_plx', 'pmdec_plx_ivar', 'parallax_new', 'parallax_new_ivar',
            'new_ra_plx', 'new_dec_plx', 'new_ra_plx_ivar', 'new_dec_plx_ivar',
            'pmra_no_plx', 'pmra_no_plx_ivar', 'pmdec_no_plx', 'pmdec_no_plx_ivar',
            'new_ra_no_plx', 'new_dec_no_plx', 'new_ra_no_plx_ivar', 'new_dec_no_plx_ivar', 'pm_flag', 'ref_mjd']
QUALITY_COLS = ['pm_quality', 'pm_n_epochs', 'pm_baseline_days']
EPOCH_COLS = ['brickid', 'brickname', 'objid', 'expnum', 'ccdname', 'filter', 'mjd', 'exptime', 'airmass',
              'ra', 'dec', 'x', 'y', 'dqmask', 'flux', 'flux_ivar', 'fracin',
              'full_fit_dra', 'full_fit_ddec', 'full_fit_dra_ivar', 'full_fit_ddec_ivar', 'full_fit_x', 'full_fit_y',
              'dcr_full_fit_x', 'dcr_full_fit_y', 'dcr_full_fit_dra', 'dcr_full_fit_ddec', 'dp1', 'colour_airmass',
              'rm_full_fit_x', 'rm_full_fit_y', 'rm_full_fit_dra', 'rm_full_fit_ddec',
              'lm_full_fit_x', 'lm_full_fit_y', 'lm_full_fit_dra', 'lm_full_fit_ddec']


def parse_degrees(items):
    out = []
    for item in items:
        lo, _, hi = item.partition("-")
        out += [f"{d:03d}" for d in range(int(lo), int(hi or lo) + 1)]
    return out


def strip(a):
    return np.char.strip(a.astype(str))


def read_brick(args):
    deg_dir, brick = args
    try:
        tf = os.path.join(deg_dir, f"tractor-pm-{brick}.fits")
        have = set(fitsio.FITS(tf)[1].get_colnames())   # quality columns exist from 2026-10-23 on
        T = fitsio.read(tf, columns=[c for c in OBJ_COLS + QUALITY_COLS if c in have])
        F = fitsio.read(os.path.join(deg_dir, f"forced-{brick}.fits"), columns=EPOCH_COLS)
    except Exception as e:
        return brick, None, f"{type(e).__name__}: {e}"
    # Tractor catalogues can have rows from neighbouring bricks appended at the end (e.g. large SGA galaxies), with
    # their own brickid and objid; forced files can include those sources' epochs too. Match on (brickid, objid).
    n = len(T)
    own_id = np.bincount(T['brickid'] - T['brickid'].min()).argmax() + T['brickid'].min()
    own = T['brickid'] == own_id
    tkey = (T['brickid'].astype(np.int64) << 32) | T['objid'].astype(np.int64)
    fkey = (F['brickid'].astype(np.int64) << 32) | F['objid'].astype(np.int64)
    srt = np.argsort(tkey)
    pos = np.clip(np.searchsorted(tkey[srt], fkey), 0, max(n - 1, 0))
    frow = np.where(tkey[srt][pos] == fkey, srt[pos], -1) if n else np.full(len(F), -1)
    # Own sources the pipeline (before the (brickid, objid) fix) may have mixed up with a neighbour sharing their
    # objid: wrong g-i in the DCR colour lookup, and/or the neighbour's epochs merged into the PM fit.
    clash_ids = np.union1d(T['objid'][~own], F['objid'][F['brickid'] != own_id])
    objid_clash = own & np.isin(T['objid'], clash_ids)
    # epochs with a valid fit, per object
    v = (F['full_fit_dra_ivar'] > 0) & (frow >= 0)
    o = frow[v]
    n_epochs = np.bincount(o, minlength=n)[:n]
    first = np.full(n, np.inf)
    last = np.full(n, -np.inf)
    np.minimum.at(first, o, F['mjd'][v])
    np.maximum.at(last, o, F['mjd'][v])
    baseline = np.where(n_epochs > 0, last - first, 0.)
    is_psf = (strip(T['type']) == 'PSF') & own
    is_gaia = (strip(T['ref_cat']) == 'GE') & own
    ok = is_psf & (T['pm_flag'] == 1)
    extra = {'n_epochs': n_epochs.astype(np.int16), 'baseline_days': baseline.astype(np.float32),
             'objid_clash': objid_clash}
    gaia_rows = (frow >= 0) & is_gaia[np.maximum(frow, 0)]
    summary = {'brickname': brick, 'ra': float(np.median(T['ra'])), 'dec': float(np.median(T['dec'])),
               'n_objects': int(own.sum()), 'n_appended': int((~own).sum()), 'n_clash_psf': int((objid_clash & is_psf).sum()), 'n_psf': int(is_psf.sum()), 'n_pm_ok': int(ok.sum()),
               'n_gaia': int(is_gaia.sum()), 'n_gaia_pm_ok': int((ok & is_gaia).sum()),
               'median_epochs_psf': float(np.median(n_epochs[is_psf])) if is_psf.any() else 0.,
               'median_baseline_psf': float(np.median(baseline[is_psf])) if is_psf.any() else 0.}
    out = {'pm': (T[ok], {k: v[ok] for k, v in extra.items()}),
           'gaia_objects': (T[is_gaia], {k: v[is_gaia] for k, v in extra.items()}),
           'gaia_epochs': (F[gaia_rows], {}),
           'summary': summary}
    return brick, out, None


def stack(parts):
    """Concatenate (structured array, extra columns) parts column by column (string widths may differ by file)."""
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return None
    names = list(parts[0][0].dtype.names) + list(parts[0][1])
    cols = [np.concatenate([p[0][c] for p in parts]) for c in parts[0][0].dtype.names]
    cols += [np.concatenate([p[1][c] for p in parts]) for c in parts[0][1]]
    return cols, names


def write(path, stacked):
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    if stacked is None:
        return
    cols, names = stacked
    fitsio.write(tmp, cols, names=names, clobber=True)
    os.replace(tmp, path)


def combine_degree(degree, out_root, workers):
    deg_dir = os.path.join(out_root, degree)
    bricks = sorted(f[11:19] for f in os.listdir(deg_dir)
                    if f.startswith("tractor-pm-") and os.path.getsize(os.path.join(deg_dir, f)) > 0) \
        if os.path.isdir(deg_dir) else []
    if not bricks:
        print(f"degree {degree}: no finished bricks", flush=True)
        return
    results, errors = [], []
    with ProcessPoolExecutor(workers) as ex:
        for brick, out, err in ex.map(read_brick, [(deg_dir, b) for b in bricks], chunksize=4):
            (errors.append((brick, err)) if err else results.append(out))
    cdir = os.path.join(out_root, "combined")
    os.makedirs(cdir, exist_ok=True)
    for kind, fname in (('pm', f"pm-{degree}.fits"), ('gaia_objects', f"gaia-objects-{degree}.fits"),
                        ('gaia_epochs', f"gaia-epochs-{degree}.fits")):
        write(os.path.join(cdir, fname), stack([r[kind] for r in results]))
    S = [r['summary'] for r in results]
    write(os.path.join(cdir, f"bricks-{degree}.fits"),
          ([np.array([s[k] for s in S]) for k in S[0]], list(S[0])))
    print(f"degree {degree}: {len(results)} bricks combined, {sum(s['n_pm_ok'] for s in S)} PSF objects with PMs, "
          f"{sum(s['n_gaia'] for s in S)} Gaia stars; {len(errors)} unreadable" +
          "".join(f"\n  {b}: {e}" for b, e in errors), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("degrees", nargs="+", help="RA degree slices or ranges, e.g. 060 or 000-063")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    for degree in parse_degrees(args.degrees):
        t0 = time.time()
        combine_degree(degree, args.out_root, args.workers)
        print(f"  ({time.time() - t0:.0f} s)", flush=True)
