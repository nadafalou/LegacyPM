"""The g - i colour used by the colour-dependent corrections (DCR, lateral colour) and the NO_COLOUR flag.

Measured g - i when the Tractor fluxes in g and i are both > 0. Otherwise (mostly bricks without i-band coverage)
g - i predicted from g - z with the cubic in colour_fallback.json, fitted by run/colour-fallback.ipynb on stars with
S/N > 50 in griz: held-out robust scatter 0.026 mag, |bias| < 0.025 mag at all colours (Nada, 2026-10-23: use a
fallback where accurate; g - r, r - z and Gaia BP - RP were tested and are not: 0.14-0.17 mag with colour-dependent
biases up to 0.27 mag). Outside the g - z range of the fit there is no colour.
"""
import os
import json
import numpy as np

COLOUR_SOURCE_NONE, COLOUR_SOURCE_GI, COLOUR_SOURCE_GZ = 0, 1, 2
FALLBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colour_fallback.json')
with open(FALLBACK_FILE) as _f:
    FALLBACK = json.load(_f)


def _mag_diff(f1, f2):
    f1, f2 = np.asarray(f1, dtype=float), np.asarray(f2, dtype=float)
    ok = (f1 > 0) & (f2 > 0)
    out = np.full(f1.shape, np.nan)
    out[ok] = -2.5 * np.log10(f1[ok] / f2[ok])
    return out


def g_minus_i(flux_g, flux_i, flux_z, model=FALLBACK):
    """(g - i, source) per object: source COLOUR_SOURCE_GI (measured), COLOUR_SOURCE_GZ (from g - z) or
    COLOUR_SOURCE_NONE (g - i is NaN)."""
    gi = _mag_diff(flux_g, flux_i)
    gz = _mag_diff(flux_g, flux_z)
    src = np.where(np.isfinite(gi), COLOUR_SOURCE_GI, COLOUR_SOURCE_NONE).astype(np.int16)
    use = ~np.isfinite(gi) & np.isfinite(gz) & (gz >= model['gz_min']) & (gz <= model['gz_max'])
    gi[use] = np.polyval(model['coeffs'], gz[use])
    src[use] = COLOUR_SOURCE_GZ
    return gi, src
