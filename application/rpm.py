"""Reduced proper motion (RPM) diagram pilot (Application A).

Reads the v2 tractor-pm-<brick>.fits catalogues (read-only) for a set of bricks, computes
    mu_total = sqrt(pmra_no_plx^2 + pmdec_no_plx^2)                 mas/yr
    mu_err   = sqrt(1/pmra_no_plx_ivar + 1/pmdec_no_plx_ivar)       mas/yr (per-axis errors combined; conservative)
    H_r      = r + 5 log10(mu_total / 1000) + 5                      mu in arcsec/yr
    colour   = g - z
with extinction-corrected (mw_transmission) Tractor magnitudes, and gives white-dwarf cooling tracks from the
Montreal models (Bergeron et al.; Bedard et al. 2020) at fixed tangential velocity.

Nothing here touches the pipeline; it only reads the catalogues.
"""
import glob
import json
import os

import numpy as np
import fitsio

CATALOGUE_ROOT = '/pscratch/sd/n/nelfalou/legacypm-catalogue-v2'
HERE = os.path.dirname(os.path.abspath(__file__))
WD_TABLE = os.path.join(HERE, 'data', 'bergeron_Table_Mass_0.6.txt')

COLUMNS = ['brickname', 'objid', 'brick_primary', 'type', 'ra', 'dec', 'ref_cat', 'ref_id', 'maskbits',
           'gaia_phot_g_mean_mag', 'gaia_phot_bp_mean_mag', 'gaia_phot_rp_mean_mag', 'ebv', 'pmra', 'pmdec',
           'flux_g', 'flux_r', 'flux_i', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
           'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z',
           'pmra_no_plx', 'pmra_no_plx_ivar', 'pmdec_no_plx', 'pmdec_no_plx_ivar',
           'parallax_new', 'parallax_new_ivar',
           'pm_flag', 'pm_quality', 'pm_n_epochs', 'pm_baseline_days', 'pm_colour_source']

K_VTAN = 4.74047   # km/s per (arcsec/yr * pc)


def brick_centre(name):
    """RA, Dec (deg) of a brick from its name, e.g. '0300m335' -> (30.0, -33.5)."""
    ra = int(name[:4]) / 10.
    dec = int(name[5:]) / 10. * (-1 if name[4] == 'm' else 1)
    return ra, dec


def find_bricks(ra_range=(25, 35), dec_range=(-40, -30), root=CATALOGUE_ROOT):
    """All non-empty tractor-pm catalogues whose brick centre lies in the box."""
    paths = []
    for deg in range(int(np.floor(ra_range[0])), int(np.ceil(ra_range[1])) + 1):
        for p in sorted(glob.glob(os.path.join(root, '%03d' % deg, 'tractor-pm-*.fits'))):
            ra, dec = brick_centre(os.path.basename(p)[len('tractor-pm-'):-len('.fits')])
            if (ra_range[0] <= ra < ra_range[1] and dec_range[0] <= dec < dec_range[1]
                    and os.path.getsize(p) > 0):
                paths.append(p)
    return paths


def pick_bricks(paths, n, seed=1):
    """n bricks spread over the box: random draw (fixed seed), sorted by name."""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(paths, size=min(n, len(paths)), replace=False))


def load(paths, columns=COLUMNS):
    """Concatenate the requested columns of the given catalogues (structured array)."""
    return np.concatenate([fitsio.read(p, columns=columns) for p in paths])


def _mag(flux, transmission):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(flux > 0, 22.5 - 2.5 * np.log10(flux / transmission), np.nan)


def derive(T):
    """Dict of derived quantities: dereddened g, r, z, colour g - z, mu_total, mu_err, H_r, gaia flag."""
    d = {}
    for b in 'grz':
        d[b] = _mag(T['flux_' + b], T['mw_transmission_' + b])
    d['g_z'] = d['g'] - d['z']
    with np.errstate(divide='ignore', invalid='ignore'):
        d['mu'] = np.hypot(T['pmra_no_plx'], T['pmdec_no_plx'])
        d['mu_err'] = np.sqrt(1. / T['pmra_no_plx_ivar'] + 1. / T['pmdec_no_plx_ivar'])
        d['H_r'] = d['r'] + 5 * np.log10(d['mu'] / 1000.) + 5
    d['gaia'] = np.char.strip(T['ref_cat'].astype(str)) == 'GE'
    d['snr_r'] = T['flux_r'] * np.sqrt(T['flux_ivar_r'])
    return d


def hpm_mask(T, d, mu_min=20., snr_min=4., snr_r_min=30., r_max=None):
    """Clean, primary PSF objects with a significant high proper motion, measured g, r, z, r-band S/N > snr_r_min
    (and r < r_max if given).

    snr_r_min: at low S/N the PM errors are underestimated (pull width up to ~2 for faint non-Gaia PSF objects, mostly
    zero-PM galaxies and QSOs; pm_quality == 0 does not flag low S/N), so a 4-sigma 'HPM' selection there is noise: its
    PM directions are isotropic, with no solar reflex. By the direction test the non-Gaia HPM objects are real at
    S/N_r > 30, 70-80% real at 20-30, and mostly noise below 20. The limit is in S/N rather than magnitude because it
    carries over between survey depths: r < 23 (the first choice) matches S/N_r > 30 in the DES-depth pilot field but
    lets noise into the candidate list of a DECaLS-depth field (RPM_PILOT.md section 3.4)."""
    return ((T['brick_primary']) & (np.char.strip(T['type'].astype(str)) == 'PSF')
            & (T['pm_quality'] == 0) & (T['pm_flag'] == 1)
            & (T['pmra_no_plx_ivar'] > 0) & (T['pmdec_no_plx_ivar'] > 0)
            & (d['mu'] > mu_min) & (d['mu'] / d['mu_err'] > snr_min)
            & np.isfinite(d['g']) & np.isfinite(d['r']) & np.isfinite(d['z']) & (d['snr_r'] > snr_r_min)
            & ((d['r'] < r_max) if r_max is not None else True))


def wd_tracks(path=WD_TABLE):
    """Montreal 0.6 Msun cooling sequences: {'DA': (Teff, g - z, M_r), 'DB': ...}.

    Uses the SDSS-system g, r, z columns (the first ugriz block), not DECam: the DECam - SDSS differences in g - z are
    a few hundredths of a mag for these colours, below what matters for this pilot.
    """
    out, block, name = {}, [], None
    with open(path) as f:
        lines = f.readlines()
    for line in lines + ['END']:
        s = line.split()
        if line == 'END' or (s and ('grid' in line)):
            if block:
                out[name] = np.array(block)
            block = []
            name = 'DA' if 'hydrogen' in line else 'DB'
            continue
        if not s or s[0] == 'Teff':
            continue
        block.append([float(x) for x in s])
    # columns: Teff logg Mbol BC U B V R I J H Ks Y J H K W1 W2 W3 W4 S3.6 S4.5 S5.8 S8.0 u g r i z (SDSS) ...
    tracks = {}
    for k, a in out.items():
        teff, g, r, z = a[:, 0], a[:, 25], a[:, 26], a[:, 28]
        tracks[k] = (teff, g - z, r)
    return tracks


def wd_track_H(g_z, vtan, kind='DA', teff_min=4500., tracks=None):
    """H_r of the WD cooling track at colour g_z for tangential velocity vtan (interpolated).

    Below ~4500 K the pure-H track turns blue again (collision-induced absorption), so g - z is not single-valued;
    only the part with Teff >= teff_min is used, and colours outside it give NaN."""
    teff, gz, Mr = (tracks or wd_tracks())[kind]
    k = teff >= teff_min
    o = np.argsort(gz[k])
    return np.interp(g_z, gz[k][o], H_from_M(Mr[k][o], vtan), left=np.nan, right=np.nan)


def wd_track_colours(path=WD_TABLE, teff_min=4500.):
    """Montreal 0.6 Msun DA (pure-H) track, Teff >= teff_min (where the colours are single-valued): dict teff, g_z,
    g_r (SDSS) and bp_rp (Gaia DR3 passbands, columns G3_BP, G3_RP)."""
    a = np.loadtxt(path, skiprows=2, max_rows=61)
    a = a[a[:, 0] >= teff_min]
    return dict(teff=a[:, 0], g_z=a[:, 25] - a[:, 28], g_r=a[:, 25] - a[:, 26], bp_rp=a[:, 38] - a[:, 39])


def bprp_relation(colour, bprp, kind='g_z', step=0.1, min_n=50, x_max=None):
    """Table (x, y) mapping a dereddened Legacy Surveys colour (kind 'g_z' or 'g_r') onto Gaia BP - RP, for a second
    colour axis. colour, bprp: dereddened colours of Gaia stars.

    Median BP - RP in bins of the colour with >= min_n stars. Bluer than the bluest such bin there are only white
    dwarfs, so the DA cooling track is used there, shifted to join the stars; redder than the reddest bin (or x_max),
    a linear extrapolation. Made strictly increasing so it can be inverted."""
    ok = np.isfinite(colour) & np.isfinite(bprp)
    edges = np.arange(np.floor(np.nanmin(colour[ok])), np.nanmax(colour[ok]) + step, step)
    idx = np.digitize(colour[ok], edges) - 1
    x, y = [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= min_n and (x_max is None or edges[i + 1] <= x_max):
            x.append(np.median(colour[ok][m])); y.append(np.median(bprp[ok][m]))
    x, y = np.array(x), np.array(y)
    tr = wd_track_colours()
    o = np.argsort(tr[kind])
    tx, ty = tr[kind][o], tr['bp_rp'][o]
    blue = tx < x[0] - step
    ty = ty + (y[0] - np.interp(x[0], tx, ty))
    slope = np.polyfit(x[-3:], y[-3:], 1)[0]
    xr = x[-1] + np.arange(1, 31) * 0.2
    x = np.concatenate([tx[blue], x, xr])
    y = np.concatenate([ty[blue], y, y[-1] + slope * (xr - x[-1 - len(xr)])])
    y = np.maximum.accumulate(y) + np.arange(len(y)) * 1e-6
    return x, y


def gz_bprp_table():
    """Table (g - z, BP - RP) through the pipeline's own colour relations: g - i from g - z with the cubic in
    py_files/colour_fallback.json (run/colour-fallback.ipynb), then the reference-colour relation
    g - i = -0.56 + 1.34 (BP - RP) of the DCR correction (py_files/brick_DCR.py), inverted. The cubic is fitted for
    -0.12 < g - z < 3.97 and extrapolated outside (it is monotonic for -2.1 < g - z < 5.4)."""
    with open(os.path.join(HERE, '..', 'py_files', 'colour_fallback.json')) as f:
        coeffs = json.load(f)['coeffs']
    gz = np.linspace(-2, 5.3, 1461)
    return gz, (np.polyval(coeffs, gz) + 0.56) / 1.34


def add_bprp_axis(ax, rel, ticks=None, label=r'$G_{\rm BP} - G_{\rm RP}$ (approx.)'):
    """Second x-axis on top of ax giving the Gaia BP - RP that corresponds to the bottom colour (rel from
    bprp_relation). The main axis's own top ticks are switched off."""
    fwd = lambda c: np.interp(c, rel[0], rel[1])
    inv = lambda b: np.interp(b, rel[1], rel[0])
    ax.tick_params(axis='x', which='both', top=False)
    sec = ax.secondary_xaxis('top', functions=(fwd, inv))
    if ticks is not None:
        sec.set_xticks(ticks)
    sec.set_xlabel(label)
    return sec


def H_from_M(M, vtan):
    """Reduced proper motion of an object of absolute magnitude M moving at vtan km/s:
    H = M + 5 log10(vtan / 4.74) = M + 5 log10(vtan) - 3.38."""
    return M + 5 * np.log10(vtan / K_VTAN)
