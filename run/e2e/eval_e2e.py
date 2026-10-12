"""Compare old (archive) vs new (cut swap + smooth lateral model) corrections and PMs on the e2e test bricks."""
import os, glob, numpy as np, fitsio

HERE = os.environ.get('LEGACYPM_E2E_DIR', '/pscratch/sd/n/nelfalou/e2e-test')
ARCHIVE = '/pscratch/sd/n/nelfalou/archive/legacypm-catalogue'
PIX = 262.


def rsig(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan


bricks = sorted(os.path.basename(p)[11:19] for p in glob.glob(f'{HERE}/pm_new/tractor-pm-*.fits'))
print('bricks:', bricks)
P = {k: [] for k in ('c', 'g', 'ivar', 'r_old_x', 'r_old_y', 'r_new_x', 'r_new_y', 'rm_old_x', 'rm_new_x',
                     'rm_old_y', 'rm_new_y', 'none_x', 'none_y', 'rmapp_old', 'rmapp_new', 'lmapp_old', 'lmapp_new')}
M = {k: [] for k in ('gaia', 'G', 'rmag', 'pmra_old', 'pmdec_old', 'pmra_new', 'pmdec_new', 'iv_old', 'iv_new',
                     'flag_new', 'gaia_pmra')}
for b in bricks:
    T = fitsio.read(f'{HERE}/in/tractor-forced-{b}.fits', columns=['ref_cat', 'flux_g', 'flux_i', 'flux_r',
                                                                 'gaia_phot_g_mean_mag', 'type', 'pmra'])
    cols = ['objid', 'x', 'y', 'full_fit_dra', 'full_fit_dra_ivar', 'dcr_full_fit_x', 'dcr_full_fit_y',
            'rm_full_fit_x', 'rm_full_fit_y', 'lm_full_fit_x', 'lm_full_fit_y']
    Fo = fitsio.read(f'{ARCHIVE}/{b[:3]}/forced-{b}.fits', columns=cols)
    Fn = fitsio.read(f'{HERE}/lm/forced-{b}.fits', columns=cols)
    assert len(Fo) == len(Fn)
    v = (Fo['x'] != 0) & (Fo['full_fit_dra_ivar'] > 0) & (Fo['full_fit_dra'] != 0) & (Fo['objid'] < len(T))
    gaia = np.char.strip(T['ref_cat'].astype(str)) == 'GE'
    with np.errstate(all='ignore'):
        col = -2.5 * np.log10(T['flux_g'] / T['flux_i'])
    o = Fo['objid'][v]
    s = gaia[o] & np.isfinite(col[o])
    sel = np.flatnonzero(v)[s]
    P['c'].append(col[Fo['objid'][sel]]); P['g'].append(T['gaia_phot_g_mean_mag'][Fo['objid'][sel]])
    P['ivar'].append(Fo['full_fit_dra_ivar'][sel])
    for ax in 'xy':
        P[f'none_{ax}'].append(Fo[f'dcr_full_fit_{ax}'][sel] - Fo[ax][sel])
        P[f'rm_old_{ax}'].append(Fo[f'rm_full_fit_{ax}'][sel] - Fo[ax][sel])
        P[f'rm_new_{ax}'].append(Fn[f'rm_full_fit_{ax}'][sel] - Fn[ax][sel])
        P[f'r_old_{ax}'].append(Fo[f'lm_full_fit_{ax}'][sel] - Fo[ax][sel])
        P[f'r_new_{ax}'].append(Fn[f'lm_full_fit_{ax}'][sel] - Fn[ax][sel])
    allv = np.flatnonzero(v)
    P['rmapp_old'].append(Fo['rm_full_fit_x'][allv] != Fo['dcr_full_fit_x'][allv])
    P['rmapp_new'].append(Fn['rm_full_fit_x'][allv] != Fn['dcr_full_fit_x'][allv])
    P['lmapp_old'].append(Fo['lm_full_fit_x'][allv] != Fo['rm_full_fit_x'][allv])
    P['lmapp_new'].append(Fn['lm_full_fit_x'][allv] != Fn['rm_full_fit_x'][allv])
    pcols = ['pmra_no_plx', 'pmdec_no_plx', 'pmra_no_plx_ivar', 'pm_flag']
    Ao = fitsio.read(f'{HERE}/pm_old/tractor-pm-{b}.fits', columns=pcols)
    An = fitsio.read(f'{HERE}/pm_new/tractor-pm-{b}.fits', columns=pcols)
    psf = np.char.strip(T['type'].astype(str)) == 'PSF'
    ok = psf & (An['pm_flag'] == 1) & (Ao['pm_flag'] == 1)
    M['gaia'].append(gaia[ok]); M['G'].append(T['gaia_phot_g_mean_mag'][ok])
    with np.errstate(all='ignore'):
        M['rmag'].append(22.5 - 2.5 * np.log10(T['flux_r'][ok]))
    M['gaia_pmra'].append(T['pmra'][ok])
    for k, A in (('old', Ao), ('new', An)):
        M[f'pmra_{k}'].append(A['pmra_no_plx'][ok]); M[f'pmdec_{k}'].append(A['pmdec_no_plx'][ok])
        M[f'iv_{k}'].append(A['pmra_no_plx_ivar'][ok])
    M['flag_new'].append(An['pm_flag'][ok])
P = {k: np.concatenate(v) for k, v in P.items()}
M = {k: np.concatenate(v) for k, v in M.items()}

print(f"\nFraction of valid measurements corrected:  CCD maps old {P['rmapp_old'].mean():.1%} -> new {P['rmapp_new'].mean():.1%};"
      f"  lateral old {P['lmapp_old'].mean():.1%} -> new {P['lmapp_new'].mean():.1%}")

print('\nPer-epoch scatter of Gaia stars around Gaia (mas, mean of x/y robust sigma)')
print(f"{'subset':26s} {'N':>8s} {'DCR only':>9s} {'old final':>10s} {'new final':>10s}")
subsets = {'all valid': np.ones(len(P['c']), bool),
           'sigma<10mas (old-corrected)': P['ivar'] > 1e4,
           'sigma>10mas (newly corr.)': P['ivar'] <= 1e4,
           'G<19 blue (c<0.92)': (P['g'] < 19) & (P['c'] < 0.92),
           'G<19 red  (c>2.33)': (P['g'] < 19) & (P['c'] > 2.33)}
for name, m in subsets.items():
    f = lambda k: PIX * np.mean([rsig(P[f'{k}_x'][m]), rsig(P[f'{k}_y'][m])])
    print(f"{name:26s} {m.sum():8d} {f('none'):9.2f} {f('r_old'):10.2f} {f('r_new'):10.2f}")

print('\nGaia stars: fitted PM (relative to Gaia, should be ~0) robust sigma, mas/yr; and pull width (PM * sqrt(ivar))')
print(f"{'G range':10s} {'N':>6s} {'old':>7s} {'new':>7s}   {'pull old':>8s} {'pull new':>8s}")
for lo, hi in ((14, 18), (18, 20), (20, 21.5)):
    m = M['gaia'] & (M['G'] > lo) & (M['G'] <= hi) & (np.abs(M['pmra_old']) < 1000) & (np.abs(M['pmra_new']) < 1000)
    so = np.mean([rsig(M['pmra_old'][m]), rsig(M['pmdec_old'][m])])
    sn = np.mean([rsig(M['pmra_new'][m]), rsig(M['pmdec_new'][m])])
    po = rsig(M['pmra_old'][m] * np.sqrt(M['iv_old'][m])); pn = rsig(M['pmra_new'][m] * np.sqrt(M['iv_new'][m]))
    print(f"{lo}-{hi:<7} {m.sum():6d} {so:7.2f} {sn:7.2f}   {po:8.2f} {pn:8.2f}")

print('\nNon-Gaia PSF stars: rms change in PM (new - old), mas/yr, by r mag; and formal sigma for scale')
for lo, hi in ((18, 20), (20, 22), (22, 23), (23, 24)):
    m = ~M['gaia'] & (M['rmag'] > lo) & (M['rmag'] <= hi) & (np.abs(M['pmra_old']) < 1000) & (np.abs(M['pmra_new']) < 1000)
    if m.sum() < 20:
        continue
    d = np.hypot(M['pmra_new'][m] - M['pmra_old'][m], M['pmdec_new'][m] - M['pmdec_old'][m]) / np.sqrt(2)
    print(f"  r {lo}-{hi}: N={m.sum():6d}  median |dPM| per axis = {np.median(d):5.2f}  "
          f"median formal sigma = {np.median(1 / np.sqrt(M['iv_new'][m])):6.2f}")
