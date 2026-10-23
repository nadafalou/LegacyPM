"""Observing geometry for the DCR correction: hour angle, airmass and parallactic angle at CTIO (DECam).
Kept separate from brick_DCR.py, which loads a 14 GB CCD table at import, so analysis code can use these cheaply."""
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
