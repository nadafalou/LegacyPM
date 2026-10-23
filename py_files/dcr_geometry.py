"""Observing geometry and the DCR model: hour angle, airmass and parallactic angle at CTIO (DECam), and the DCR shift
brick_DCR.py applies. Kept separate from brick_DCR.py, which loads a 14 GB CCD table at import, so analysis code can use
these cheaply."""
import os
import json
import numpy as np
from astropy.time import Time
from astropy import units as u


def hour_angle(ra, mjd, long=-70.81489):
    lst = Time(mjd, format="mjd").sidereal_time('mean', longitude=long).to(u.deg)    #local sidereal time
    return lst.value - ra    # hour angle


def parallactic(dec, ha, lat=-30.16606):
    '''Function will calculate airmass and
    parallactic angle (in radians) given
    input of source declination, hour angle,
    and observatory latitude (in degrees)
    '''
    dtor = np.pi / 180.
    d = dec * dtor
    h = np.where(ha > 180., (ha - 360.) * dtor, ha * dtor)  # Make negative HA instead of large
    l = lat * dtor
    
    cosz = np.sin(l) * np.sin(d) + np.cos(l) * np.cos(d) * np.cos(h)
    sinz = np.sqrt(1 - cosz * cosz)  # zenith angle always 0-180
    
    airmass = 1. / cosz
    
    # Now the parallactic angle
    zero_angle_mask = sinz <= 0
    cosp = np.zeros_like(sinz)
    
    # Avoid division by zero when sinz is zero
    cosp[~zero_angle_mask] = (np.sin(l) * np.cos(d[~zero_angle_mask]) - np.cos(l) * np.sin(d[~zero_angle_mask]) * np.cos(h[~zero_angle_mask])) / sinz[~zero_angle_mask]
    
    # Clip to handle numerical inaccuracies that might lead to cosp slightly greater than 1
    cosp = np.clip(cosp, -1, 1)
    
    p = np.arccos(cosp) * np.sign(h)
    
    # Set parallactic angle to 0 where sinz is zero
    p[zero_angle_mask] = 0.
    
    return airmass, p


# The DCR model applied by brick_DCR.py (since 2026-10-23; see brick_DCR.py)
DCR_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dcr_model_twoline.json')
with open(DCR_MODEL_FILE) as _f:
    DCR_MODEL = json.load(_f)


def dcr_shift_mas(band, c_rel, tanz, model=DCR_MODEL):
    """DCR shift along the parallactic direction (mas, to subtract):
    tan z * (a + b_blue min(c - c_break, 0) + b_red max(c - c_break, 0)), c = c_rel clipped to [c_min, c_max].
    Bands not in the model get no correction."""
    band = np.asarray(band).astype(str)
    c = np.clip(np.asarray(c_rel, dtype=float), model['c_min'], model['c_max'])
    out = np.zeros(len(c))
    for b, p in model['bands'].items():
        m = band == b
        d = c[m] - p['c_break']
        out[m] = tanz[m] * (p['a'] + p['b_blue'] * np.minimum(d, 0) + p['b_red'] * np.maximum(d, 0))
    return out
