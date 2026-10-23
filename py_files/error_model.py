"""Per-epoch positional error model, used by the corrections and the PM fit (single place to change it).

The forced photometry's statistical error underestimates the real scatter of bright stars, which levels off at a
floor set by systematics (atmosphere, residual distortions). The total error of each epoch is
    sigma_total = sqrt(sigma_stat^2 + SIGMA_SYS_MAS[band]^2)
with sigma_stat from the forced photometry (full_fit_dra_ivar, full_fit_ddec_ivar; arcsec^-2).
SIGMA_SYS_MAS per band (Nada, 2026-10-23): fitted in run/pull-vs-mag.ipynb so that per-epoch pulls of clean objects
have unit robust width at every magnitude (slice 000, Dec errors corrected for the cos(Dec) problem); the same fit
gives a scale factor k = 1 on sigma_stat, i.e. the statistical errors are right. Replaces a single 20 mas (and before
that 14.8 mas).
"""
import numpy as np

SIGMA_SYS_MAS = {'g': 16., 'r': 12., 'i': 12., 'z': 10.}
SIGMA_SYS_DEFAULT_MAS = max(SIGMA_SYS_MAS.values())   # any other band


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
    return [(f'SIGSYS_{b.upper()}', s, f'systematic per-epoch error, {b} band (mas)') for b, s in SIGMA_SYS_MAS.items()]


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
