"""Per-epoch positional error model, used by the corrections and the PM fit (single place to change it).

Since 2026-10-26 (Nada: refit sigma_sys on v2 + an extra error along the refraction direction), each epoch's error is
a 2x2 covariance in (RA*, Dec). With p the unit vector along the parallactic (refraction) direction and q across it:
    across (q):  sigma^2 = sigma_stat,q^2 + (sigma_sys,b + sigma_sys_k,b tan z)^2
    along  (p):  sigma^2 = sigma_stat,p^2 + (sigma_sys,b + sigma_sys_k,b tan z)^2 + (along_a,b + along_k,b tan z)^2
i.e. C = C_stat + sigma_iso^2 I + (a + k tan z)^2 p p^T (the isotropic part also grows with airmass: image motion from
turbulence; the along part is the extra scatter in the refraction direction), with p = (-sin pa, -cos pa) in (RA*, Dec) (CCD x = -Dec,
y = +RA; same convention as brick_DCR). sigma_stat from the forced photometry (full_fit_dra_ivar, full_fit_ddec_ivar,
the latter corrected for the legacypipe cos^2(Dec) bug, below). The along term describes random per-epoch scatter that
grows with airmass (turbulence / refraction / PSF elongation / SED-level DCR residuals); it lands mostly in Dec from CTIO.
Parameters per band: py_files/error_model.json, fitted by run/error-model-fit.ipynb on v2 per-epoch residuals (pulls of
unit width across and along, at every magnitude and tan z). Without that file: the v1-era isotropic values below
(sigma_sys g 16, r 12, i 12, z 10 mas from run/pull-vs-mag.ipynb on slice 000; 2026-10-23), no along term.
"""
import os
import json
import numpy as np

_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_model.json')
if os.path.exists(_MODEL_FILE):
    with open(_MODEL_FILE) as _f:
        ERROR_MODEL = json.load(_f)
else:
    ERROR_MODEL = {'bands': {b: {'sigma_sys': s, 'along_a': 0., 'along_k': 0.}
                             for b, s in {'g': 16., 'r': 12., 'i': 12., 'z': 10.}.items()}, 'default_band': 'g'}
SIGMA_SYS_MAS = {b: p['sigma_sys'] for b, p in ERROR_MODEL['bands'].items()}
# short ID of the parameters, recorded in headers (ERRMODEL) so reruns can tell which model made a catalogue
ERROR_MODEL_ID = __import__('hashlib').md5(json.dumps(ERROR_MODEL['bands'], sort_keys=True).encode()).hexdigest()[:8]
SIGMA_SYS_DEFAULT_MAS = max(SIGMA_SYS_MAS.values())   # any other band


def _per_band(band, key, default):
    band = np.char.strip(np.asarray(band).astype(str))
    out = np.full(band.shape, default, dtype=float)
    for b, p in ERROR_MODEL['bands'].items():
        out[band == b] = p.get(key, default)
    return out


def sigma_sys(band):
    """Systematic error (mas) per element of `band` (filter names, scalar or array)."""
    band = np.char.strip(np.asarray(band).astype(str))
    out = np.full(band.shape, SIGMA_SYS_DEFAULT_MAS)
    for b, s in SIGMA_SYS_MAS.items():
        out[band == b] = s
    return out


def total_ivar(ivar, band):
    """Total inverse variance (arcsec^-2) from the statistical one; 0 where the statistical ivar is not > 0."""
    ivar = np.asarray(ivar, dtype=float)
    sys2 = np.broadcast_to((sigma_sys(band) / 1000.)**2, ivar.shape)
    out = np.zeros_like(ivar)
    ok = ivar > 0
    out[ok] = 1. / (1. / ivar[ok] + sys2[ok])
    return out


def total_error(ivar, band):
    """Total 1-sigma error (arcsec); inf where the statistical ivar is not > 0."""
    t = total_ivar(ivar, band)
    with np.errstate(divide='ignore'):
        return np.where(t > 0, 1. / np.sqrt(np.where(t > 0, t, 1.)), np.inf)


def header_cards():
    """FITS header cards recording the model: (key, value, comment)."""
    cards = [('ERRMODEL', ERROR_MODEL_ID, 'error-model parameters ID (error_model.json)')]
    for b, p in ERROR_MODEL['bands'].items():
        B = b.upper()
        cards += [(f'SIGSYS_{B}', p['sigma_sys'], f'systematic per-epoch error, {b} band (mas)'),
                  (f'SIGSYK_{B}', p.get('sigma_sys_k', 0.), f'systematic error growth with tan z, {b} (mas)'),
                  (f'ALONGA_{B}', p['along_a'], f'extra error along refraction dir., {b}: a (mas)'),
                  (f'ALONGK_{B}', p['along_k'], f'extra error along refraction dir., {b}: k (mas/tanz)')]
    return cards


def epoch_geometry(ra, dec, mjd, exptime):
    """(pa, tanz) per epoch, exactly as brick_DCR computes them (mid-exposure hour angle at CTIO)."""
    try:
        from . import dcr_geometry
    except ImportError:
        import dcr_geometry
    ha = dcr_geometry.hour_angle(np.asarray(ra, float), np.asarray(mjd, float) + 0.5 * np.asarray(exptime, float) / 86400.)
    am, pa = dcr_geometry.parallactic(np.asarray(dec, float), ha)
    return np.asarray(pa, float), np.sqrt(np.clip(np.asarray(am, float)**2 - 1, 0, None))


def epoch_covariance(ivar_ra, ivar_dec, band, pa, tanz):
    """Per-epoch total error covariance in (RA*, Dec), arcsec^2: (var_ra, var_dec, cov). ivar_dec must already be
    corrected for the cos^2(Dec) input bug (corrected_ddec_ivar). Epochs with a statistical ivar <= 0 get inf
    variances and zero covariance."""
    ivar_ra, ivar_dec = np.asarray(ivar_ra, float), np.asarray(ivar_dec, float)
    ok = (ivar_ra > 0) & (ivar_dec > 0)
    tanz = np.asarray(tanz, float)
    s2 = ((_per_band(band, 'sigma_sys', SIGMA_SYS_DEFAULT_MAS) + _per_band(band, 'sigma_sys_k', 0.) * tanz) / 1000.)**2
    e2 = ((_per_band(band, 'along_a', 0.) + _per_band(band, 'along_k', 0.) * np.asarray(tanz, float)) / 1000.)**2
    pr, pd = -np.sin(pa), -np.cos(pa)
    with np.errstate(divide='ignore'):
        var_ra = np.where(ok, 1. / np.where(ok, ivar_ra, 1.) + s2 + e2 * pr**2, np.inf)
        var_dec = np.where(ok, 1. / np.where(ok, ivar_dec, 1.) + s2 + e2 * pd**2, np.inf)
    cov = np.where(ok, e2 * pr * pd, 0.)
    return var_ra, var_dec, cov


# The forced photometry divided full_fit_ddec_ivar by cos^2(Dec) (like the RA term, wrongly) until legacypipe commit
# 7aece157 (Dustin, 2026-09-16, "forced-motions: remove cos(dec) term from dec invvar"); positions were not affected.
# Decision, in order: (1) the input header card DDIVFIX (Dustin, 2026-10-23: T = fixed; added to files he writes or
# patches from now on; in the input's PRIMARY header, copied to the table header by brick_DCR); (2) otherwise the input file's date (header INPDATE, stamped by brick_DCR from the
# file Dustin wrote, or by prep_archive_rerun for archive bricks): written before DEC_IVAR_FIX_DATE -> affected, and
# its ddec ivar is multiplied by cos^2(Dec). Survey 2026-10-23 (1627 input bricks of 000-063): CFS files are dated
# 2024-12 .. 2025-01 and all show sigma_Dec / sigma_RA = cos(Dec); scratch files are dated 2026-09-17 .. 2026-10-10 and
# all show ~1 -- the dates and the data agree wherever the data are unambiguous. (Found 2026-10-23 from faint-star Dec
# pulls, run/pull-vs-mag.ipynb.) The data test (dec_ivar_diagnosis) is kept as a recorded check (DECRATIO) and as the
# fallback for files of unknown date; it cannot be the rule because real PSF elongation along Dec (high airmass in the
# north) raises the ratio by 5-8%, comparable to 1 - cos(Dec) at |Dec| < 35 deg.
DEC_IVAR_FIX_DATE = '2026-09-16'


def input_has_dec_ivar_bug(inpdate):
    """True if an input file written on `inpdate` ('YYYY-MM-DD') predates the fix."""
    return str(inpdate)[:10] < DEC_IVAR_FIX_DATE

def dec_ivar_diagnosis(dra_ivar, ddec_ivar, dec):
    """(ratio, cos_dec, affected) for one brick: ratio = median sigma_Dec / sigma_RA over epochs with both ivars > 0,
    cos_dec = cos(median Dec); affected when the ratio is closer to cos_dec than to 1."""
    dra_ivar, ddec_ivar = np.asarray(dra_ivar, dtype=float), np.asarray(ddec_ivar, dtype=float)
    ok = (dra_ivar > 0) & (ddec_ivar > 0)
    if not ok.any():
        return np.nan, np.nan, False
    ratio = float(np.median(np.sqrt(dra_ivar[ok] / ddec_ivar[ok])))
    cos_dec = float(np.cos(np.radians(np.median(np.asarray(dec)[ok]))))
    return ratio, cos_dec, abs(ratio - cos_dec) < abs(ratio - 1.)


def _as_bool(v):
    return v if isinstance(v, (bool, np.bool_)) else str(v).strip().upper() in ('T', 'TRUE', '1')


def corrected_ddec_ivar(dra_ivar, ddec_ivar, dec, inpdate=None, ddivfix=None):
    """(statistical Dec ivar with the cos^2(Dec) bug undone if the input is affected, header cards). Affected unless
    the input header says DDIVFIX = T (if present it decides); else if the input file date `inpdate` predates the fix;
    else by the data test."""
    ratio, cos_dec, by_data = dec_ivar_diagnosis(dra_ivar, ddec_ivar, dec)
    if ddivfix is not None:
        affected, basis = not _as_bool(ddivfix), 'DDIVFIX'
    elif inpdate:
        affected, basis = input_has_dec_ivar_bug(inpdate), 'INPDATE'
    else:
        affected, basis = by_data, 'DATA'
    ddec_ivar = np.asarray(ddec_ivar, dtype=float)
    if affected:
        ddec_ivar = ddec_ivar * np.cos(np.radians(np.asarray(dec, dtype=float)))**2
    cards = [('DECRATIO', round(ratio, 4) if np.isfinite(ratio) else -1., 'median sigma_Dec/sigma_RA of input ivars'),
             ('DECFIX', bool(affected), 'input Dec ivar x cos^2(Dec) (legacypipe bug < 7aece157)'),
             ('DECBASIS', basis, 'how DECFIX was decided')]
    return ddec_ivar, cards
