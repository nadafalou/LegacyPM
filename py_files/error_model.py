"""Per-epoch positional error model, used by the corrections and the PM fit (single place to change it).

The forced photometry's statistical error underestimates the real scatter of bright stars, which levels off at a
floor set by systematics (atmosphere, residual distortions; see the scatter-vs-magnitude figure). The total error of
each epoch is
    sigma_total = sqrt(sigma_stat^2 + SIGMA_SYS_MAS^2)
with sigma_stat from the forced photometry (full_fit_dra_ivar, full_fit_ddec_ivar; arcsec^-2).
SIGMA_SYS_MAS = 20 mas (Nada, 2026-10-23, eyeballed from the corrected bright-star floor of scatter_vs_mag;
replaces the earlier 14.8 mas).
"""
import numpy as np

SIGMA_SYS_MAS = 20.


def total_ivar(ivar):
    """Total inverse variance (arcsec^-2) from the statistical one; 0 where the statistical ivar is not > 0."""
    ivar = np.asarray(ivar, dtype=float)
    out = np.zeros_like(ivar)
    ok = ivar > 0
    out[ok] = 1. / (1. / ivar[ok] + (SIGMA_SYS_MAS / 1000.)**2)
    return out


def total_error(ivar):
    """Total 1-sigma error (arcsec); inf where the statistical ivar is not > 0."""
    t = total_ivar(ivar)
    with np.errstate(divide='ignore'):
        return np.where(t > 0, 1. / np.sqrt(np.where(t > 0, t, 1.)), np.inf)
