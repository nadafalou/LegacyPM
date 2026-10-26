"""Rebuild the CCD (tree-ring) maps from the 000-063 Gaia stars, with the current DCR model applied first (PLAN B1).

The maps brick_ringmaps.py applies are, per CCD x band, the median offset (measured - Gaia-predicted position, pixels)
of high-S/N Gaia-star epochs after DCR, in square bins, interpolated bilinearly. The current file (Apr 2024) was built
from the old pre-DR10 forced-motions-3 run and makes g worse (PLAN H3). This rebuilds it from the combined Gaia-epoch
files of our own run, in three steps:

  sample    per slice: Gaia-star epochs with sigma < 10 mas, measured (full_fit_dra != 0), dqmask == 0, a g - i colour
            (colours.py), not BRIGHT/SATURATED, with the DCR shift of the current model (dcr_geometry.dcr_shift_mas)
            removed exactly as brick_DCR does -> <combined>/ringmap-sample/<deg>.npz
  validate  maps from 80% of the bricks (by brickid), applied to the other 20%: per-band robust per-axis scatter of the
            held-out offsets with no map, the 2024 maps, and new maps with 16/32/64-px bins -> ringmap-validation.json
  build     maps from all bricks for each bin size -> ringmaps-dr10-b<bin>.fits (same format as the 2024 file:
            one (ny, nx, 2) image per 'CCD-band' extension, dx and dy in pixels) + counts per bin (.npz)

  python ringmap_build.py sample validate build --workers 32        # on a compute node (~100 GB of input)
"""
import os
import sys
import json
import argparse
import numpy as np
import fitsio
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_files import dcr_geometry
from py_files.dcr_fit import ccd_reference_colour, _key, PIXSCALE
from py_files.colours import g_minus_i
from py_files.pm_analysis import V1_COMBINED_DIR as COMBINED_DIR   # the v1 combined files the adopted maps were built from

OLD_MAPS = '/global/cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/rm-corrected-forced-motions/ringmaps.fits'
BANDS = 'griz'
BINNINGS = (16, 32, 64)
NX, NY = 2048, 4096
MASKBITS_BAD = (1 << 1) | (1 << 11) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 14)   # BRIGHT, MEDIUM, SATUR_*
EPOCH_COLS = ['brickid', 'objid', 'filter', 'full_fit_dra', 'full_fit_dra_ivar', 'full_fit_x', 'full_fit_y', 'x', 'y',
              'ra', 'dec', 'mjd', 'exptime', 'expnum', 'ccdname', 'dqmask']
SAMPLE_DIR = os.path.join(COMBINED_DIR, 'ringmap-sample')
REF = {}   # CCD reference colours, loaded once before forking


def sample_slice(deg):
    fe, fo = (os.path.join(COMBINED_DIR, f'gaia-{k}-{deg}.fits') for k in ('epochs', 'objects'))
    if not (os.path.exists(fe) and os.path.exists(fo)):
        return deg, 0
    F = fitsio.read(fe, columns=EPOCH_COLS)
    F = F[(F['full_fit_dra_ivar'] > 1e4) & (F['full_fit_dra'] != 0) & (F['dqmask'] == 0)]   # sigma < 10 mas
    T = fitsio.read(fo, columns=['brickid', 'objid', 'flux_g', 'flux_i', 'flux_z', 'maskbits'])
    tkey = (T['brickid'].astype(np.int64) << 32) | T['objid'].astype(np.int64)
    fkey = (F['brickid'].astype(np.int64) << 32) | F['objid'].astype(np.int64)
    srt = np.argsort(tkey)
    pos = np.clip(np.searchsorted(tkey[srt], fkey), 0, len(T) - 1)
    row = np.where(tkey[srt][pos] == fkey, srt[pos], -1)
    F, row = F[row >= 0], row[row >= 0]
    gi, _ = g_minus_i(T['flux_g'][row], T['flux_i'][row], T['flux_z'][row])
    keys, giref, ccd_index = REF['keys'], REF['giref'], REF['index']
    names = np.char.strip(F['ccdname'].astype(str))
    un, inv = np.unique(names, return_inverse=True)
    cidx = np.array([ccd_index.get(n, -1) for n in un])[inv]
    k = _key(F['expnum'], cidx)
    p = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    c_rel = gi - np.where((keys[p] == k) & (cidx >= 0), giref[p], np.nan)
    # brick_DCR corrects only -2 < c_rel < 3; keep only those (others would carry uncorrected DCR into the maps)
    ok = np.isfinite(c_rel) & (c_rel > -2) & (c_rel < 3) & ((T['maskbits'][row].astype(np.int64) & MASKBITS_BAD) == 0)
    F, c_rel, names = F[ok], c_rel[ok], names[ok]
    ha = dcr_geometry.hour_angle(F['ra'], F['mjd'] + 0.5 * F['exptime'] / 86400.)
    am, pa = dcr_geometry.parallactic(F['dec'], ha)
    band = np.char.strip(F['filter'].astype(str))
    corr = dcr_geometry.dcr_shift_mas(band, c_rel, np.sqrt(np.clip(am**2 - 1, 0, None))) / PIXSCALE   # pixels
    out = dict(ccd=names.astype('U3'), band=np.array([BANDS.find(b) for b in band], np.int8),
               x=F['x'].astype(np.float32), y=F['y'].astype(np.float32),
               dx=(F['full_fit_x'] - F['x'] - np.cos(pa) * corr).astype(np.float32),     # as brick_DCR: x - dcrx
               dy=(F['full_fit_y'] - F['y'] + np.sin(pa) * corr).astype(np.float32),     # y - dcry, dcry = -sin * corr
               brickid=F['brickid'].astype(np.int32), gi=gi[ok].astype(np.float32))   # gi: for the lateral fit (C1)
    np.savez(os.path.join(SAMPLE_DIR, f'{deg}.npz'), **out)
    return deg, len(F)


def load_samples(brick_fraction=1., seed=0):
    """All sample files (or a random fraction of the bricks), with ccd / map indices added."""
    rng = np.random.default_rng(seed)
    parts = []
    for f in sorted(os.listdir(SAMPLE_DIR)):
        if not f.endswith('.npz'):
            continue
        p = dict(np.load(os.path.join(SAMPLE_DIR, f)))
        if brick_fraction < 1:
            b = np.unique(p['brickid'])
            keep = np.isin(p['brickid'], rng.choice(b, max(1, int(round(brick_fraction * len(b)))), replace=False))
            p = {k: v[keep] for k, v in p.items()}
        parts.append(p)
    S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ccds = np.unique(S['ccd'])
    S['ccd_i'] = np.searchsorted(ccds, S['ccd']).astype(np.int16)
    S['ccds'] = ccds
    S['map_i'] = (S['ccd_i'].astype(np.int32) * 4 + S['band'])
    return S, ccds


def binned_medians(S, m, binning):
    """Median dx, dy and counts per (map, bin) of the selected epochs: arrays (n_maps, ny, nx)."""
    nx, ny = NX // binning, NY // binning
    xb = np.clip((S['x'][m] // binning).astype(np.int64), 0, nx - 1)
    yb = np.clip((S['y'][m] // binning).astype(np.int64), 0, ny - 1)
    n_maps = int(S['map_i'].max()) + 1
    key = (S['map_i'][m].astype(np.int64) * ny + yb) * nx + xb
    size = n_maps * ny * nx
    count = np.bincount(key, minlength=size)
    out = []
    for comp in ('dx', 'dy'):
        order = np.lexsort((S[comp][m], key))
        ks, vs = key[order], S[comp][m][order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        n = np.diff(np.r_[starts, len(ks)])
        lo, hi = starts + (n - 1) // 2, starts + n // 2
        med = np.zeros(size, np.float32)
        med[ks[starts]] = 0.5 * (vs[lo] + vs[hi])
        out.append(med.reshape(n_maps, ny, nx))
    return out[0], out[1], count.reshape(n_maps, ny, nx)


def splines(dxmap, dymap, binning):
    """Bilinear splines exactly as brick_ringmaps.make_splines (1-indexed bin centres)."""
    ny, nx = dxmap.shape
    xv = binning * np.arange(nx) + 0.5 * binning + 1
    yv = binning * np.arange(ny) + 0.5 * binning + 1
    bbox = [1, nx * binning + 1, 1, ny * binning + 1]
    return (RectBivariateSpline(xv, yv, dxmap.T, bbox=bbox, kx=1, ky=1),
            RectBivariateSpline(xv, yv, dymap.T, bbox=bbox, kx=1, ky=1))


def apply_maps(S, m, maps_by_key, binning):
    """Correction (dx, dy) in pixels for the selected epochs; maps_by_key[(ccd, band)] = (dxmap, dymap)."""
    cx, cy = np.zeros(m.sum(), np.float32), np.zeros(m.sum(), np.float32)
    x, y = S['x'][m], S['y'][m]
    mi = S['map_i'][m]
    order = np.argsort(mi, kind='stable')
    u, starts = np.unique(mi[order], return_index=True)
    rows = dict(zip(u.tolist(), np.split(order, starts[1:])))
    ccds = S['ccds']
    for (c, b), (dxm, dym) in maps_by_key.items():
        i = np.searchsorted(ccds, c)
        if i >= len(ccds) or ccds[i] != c:
            continue
        j = rows.get(int(i) * 4 + BANDS.find(b))
        if j is None:
            continue
        sx, sy = splines(dxm, dym, binning)
        cx[j], cy[j] = sx(x[j], y[j], grid=False), sy(x[j], y[j], grid=False)
    return cx, cy


def as_dict(dx, dy, ccds):
    return {(c, b): (dx[i * 4 + j], dy[i * 4 + j]) for i, c in enumerate(ccds) for j, b in enumerate(BANDS)
            if i * 4 + j < len(dx)}


def old_maps(path=OLD_MAPS):
    """{(ccd, band): (dxmap, dymap)} from a map file (default: the Apr 2024 maps)."""
    import astropy.io.fits as pf
    out = {}
    with pf.open(path) as h:
        for hdu in h:
            if hdu.data is None:
                continue
            out[(hdu.name[:-2], hdu.name[-1:].lower())] = (hdu.data[:, :, 0].astype(np.float32), hdu.data[:, :, 1].astype(np.float32))
    return out


def robust(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def validate(S, ccds):
    test = (S['brickid'].astype(np.int64) * 2654435761 % 2**32) % 5 == 0   # 20% of bricks, fixed
    train = ~test
    res = {'n_train': int(train.sum()), 'n_test': int(test.sum()), 'scatter_mas': {}}
    versions = {'none': None, 'maps 2024': (old_maps(), 16)}
    for b in BINNINGS:
        dx, dy, _ = binned_medians(S, train, b)
        versions[f'new {b} px'] = (as_dict(dx, dy, ccds), b)
    for name, v in versions.items():
        cx, cy = (np.zeros(test.sum()), np.zeros(test.sum())) if v is None else apply_maps(S, test, *v)
        rx, ry = (S['dx'][test] - cx) * PIXSCALE, (S['dy'][test] - cy) * PIXSCALE
        res['scatter_mas'][name] = {b: float(np.sqrt((robust(rx[S['band'][test] == j])**2 + robust(ry[S['band'][test] == j])**2) / 2))
                                    for j, b in enumerate(BANDS)}
        print(name, res['scatter_mas'][name], flush=True)
    json.dump(res, open(os.path.join(COMBINED_DIR, 'ringmap-validation.json'), 'w'), indent=1)


def build(S, ccds):
    allm = np.ones(len(S['x']), bool)
    for b in BINNINGS:
        dx, dy, n = binned_medians(S, allm, b)
        path = os.path.join(COMBINED_DIR, f'ringmaps-dr10-b{b}.fits')
        tmp = path + '.tmp'
        first = True
        for i, c in enumerate(ccds):
            for j, band in enumerate(BANDS):
                k = i * 4 + j
                if k >= len(dx) or n[k].sum() == 0:
                    continue
                fitsio.write(tmp, np.dstack((dx[k], dy[k])), extname=f'{c}-{band}', clobber=first,
                             header={'BINNING': b, 'NSTARS': int(n[k].sum())})
                first = False
        os.replace(tmp, path)
        np.savez(os.path.join(COMBINED_DIR, f'ringmaps-dr10-b{b}-counts.npz'), counts=n, ccds=ccds)
        nb = n.reshape(len(n), -1)
        print(f'{b} px: wrote {path}; median epochs per non-empty bin by band: ' +
              ', '.join(f'{band} {np.median(nb[j::4][nb[j::4] > 0]):.0f}' for j, band in enumerate(BANDS)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='+', choices=['sample', 'validate', 'build'])
    ap.add_argument('--degrees', default='000-063')
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    if 'sample' in args.steps:
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        REF['keys'], REF['giref'], REF['index'] = ccd_reference_colour()
        lo, _, hi = args.degrees.partition('-')
        degs = [f'{d:03d}' for d in range(int(lo), int(hi or lo) + 1)]
        with ProcessPoolExecutor(args.workers) as ex:
            for deg, n in ex.map(sample_slice, degs):
                print(f'sample {deg}: {n:,} epochs', flush=True)
    if 'validate' in args.steps or 'build' in args.steps:
        S, ccds = load_samples()
        print(f'{len(S["x"]):,} epochs, {len(ccds)} CCDs; by band: ' +
              ', '.join(f'{b} {np.sum(S["band"] == j):,}' for j, b in enumerate(BANDS)), flush=True)
        if 'validate' in args.steps:
            validate(S, ccds)
        if 'build' in args.steps:
            build(S, ccds)


if __name__ == '__main__':
    main()
