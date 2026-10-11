import os
import json
import numpy as np
from astrometry.util.fits import *
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import sys
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BINNING = 16
BY = int(4096 // BINNING)
BX = int(2048 // BINNING)
pcolors = np.array([0.07428406, 0.92169225, 2.33498669, 3.49276543])
median_color = 1.642 # pivot

# Smooth lateral-colour model (replaces the binned per-CCD maps below, which were noise-dominated: see PLAN.md
# "Lateral-colour (LM) diagnosis"). Fit and written by run/lateral-model.ipynb.
LATERAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lateral_model_poly3.json')


def load_lateral_model(path=LATERAL_MODEL_PATH):
    with open(path) as f:
        model = json.load(f)
    model['coef'] = {filt: np.array(c) for filt, c in model['coef'].items()}
    return model


def lateral_shift(model, ccdnames, filt, x, y, colour):
    """Lateral-colour shift (pixels) to subtract from the positions x, y (rm_full_fit_x/y) of measurements in
    band `filt`. Returns (dx, dy); zero for CCDs or bands the model doesn't cover."""
    dx = np.zeros(len(x))
    dy = np.zeros(len(x))
    if filt not in model['coef']:
        return dx, dy
    crpix = np.array([model['crpixes'].get(c, (np.nan, np.nan)) for c in ccdnames], dtype=float).reshape(-1, 2)
    ok = np.isfinite(crpix[:, 0])
    u = (x[ok] - crpix[ok, 0]) / model['fp_scale']
    v = (y[ok] - crpix[ok, 1]) / model['fp_scale']
    P = np.stack([u**i * v**j for i, j in model['poly_terms']], 1) * (colour[ok] - model['c0'])[:, None]
    n = len(model['poly_terms'])
    coef = model['coef'][filt]
    dx[ok] = P @ coef[:n]
    dy[ok] = P @ coef[n:]
    return dx, dy


# --- Legacy binned maps (no longer used by the pipeline; kept for create-maps.ipynb) ---


def get_gaia_data(directory='/pscratch/sd/d/dstn/rm-corrected-3-gaia/'): ## refer   to RM_colours.ipynb
    data = fits_table(os.path.join(directory, 'all-forced.fits'))
    T = fits_table(os.path.join(directory, 'all-tractor.fits'))
    dc = T.color[data.t_index]
    return data, dc
    # G = merge_tables([
    #         fits_table(directory + '/gaia-1k.fits'),
    #         fits_table(directory + '/gaia-1k-to-4k.fits'),
    #         fits_table(directory + '/gaia-4k-to-end.fits'),
    #     ])
    # return G

    
def make_lateralmaps(data, filename, data_colour):
    lateralmaps = {}
    ccdnames = np.unique(data.ccdname)
    first = True

    for ccdname in ccdnames:
        for filt in ['g','r','i','z']:
            J = np.flatnonzero((data.ccdname == ccdname)
                                * np.isin(data.filter, filt)
                                * (data.full_fit_dra_ivar > 1e4)  # build maps from high-S/N measurements only (sigma < 10 mas)
                                * (data.dcr_full_fit_dra != 0.)
                                * (data.dqmask == 0)
                                * ((data_colour > pcolors[0]) * (data_colour < pcolors[1])
                                | (data_colour > pcolors[2]) * (data_colour < pcolors[3])))
            # number of pixels to bin together for tree-rings
            xbin = np.floor(data.x[J] / BINNING).astype(int)
            ybin = np.floor(data.y[J] / BINNING).astype(int)
            dxvals = {}
            dyvals = {}
            for xb,yb,dx,dy in zip(xbin, ybin, ((data.rm_full_fit_x[J] - data.x[J]) / (data_colour[J] - median_color)), ((data.rm_full_fit_y[J] - data.y[J])) / (data_colour[J] - median_color)):
                if not (xb,yb) in dxvals:
                    dxvals[(xb,yb)] = []
                    dyvals[(xb,yb)] = []
                dxvals[(xb,yb)].append(dx)
                dyvals[(xb,yb)].append(dy)
            dxmap = np.zeros((BY,BX), np.float32)
            dymap = np.zeros((BY,BX), np.float32)
            for yb in range(BY):
                for xb in range(BX):
                    if not (xb,yb) in dxvals:
                        continue
                    v = dxvals[(xb,yb)]
                    dxmap[yb,xb] = np.median(v)
                    v = dyvals[(xb,yb)]
                    dymap[yb,xb] = np.median(v)
            lateralmaps[(ccdname, filt)] = (dxmap, dymap)

            img = np.dstack((dxmap,dymap))
            extname = '%s-%s' % (ccdname, filt)
            fitsio.write(filename, img, clobber=first, extname=extname)
            first = False
        

def get_lateralmaps(filename):
    lateralmaps = fits.open(filename)
    lateralmaps_dict = {}

    for rmap in lateralmaps:
        ccdname, filt = rmap.name[:-2], rmap.name[-1:]
        lateralmaps_dict[(ccdname, filt)] = (rmap.data[:,:,0], rmap.data[:,:,1])
    return lateralmaps_dict


def make_splines(lateralmaps):
    # copied and edited from https://github.com/gbernstein/pixmappy/blob/2c912e2f37d03fc0f0892768dd359d3ec7743618/pixmappy/DESMaps.py#L99C20-L99C20
    # makes the tweak map spline thing
    tweaks = {}

    for rmap in lateralmaps.keys():
        # Locate grid points, in 1-indexed pixel system
        xvals = BINNING * np.arange(BX) + 0.5 * BINNING + 1
        yvals = BINNING * np.arange(BY) + 0.5 * BINNING + 1
        bbox = [1, BX * BINNING + 1, 1, BY * BINNING + 1]
        # Create linear spline for x and y components
        # Note that data array comes in with (y,x) indexing
        tweaks[rmap] = (RectBivariateSpline(xvals, yvals, lateralmaps[rmap][0].transpose(),
                                                    bbox=bbox, kx=1, ky=1),
                            RectBivariateSpline(xvals, yvals, lateralmaps[rmap][1].transpose(),
                                                    bbox=bbox, kx=1, ky=1))
    return tweaks


def create_brick_corrected_data(filename, corr_dir, old_dir, tractor_dir, model):
    logger.info(f"Starting correction for brick: {filename}")
    start_time = time.time()
    
    try:
        f = os.path.join(old_dir, filename)
        forced_table = fits.open(f)
        corr_table = forced_table.copy()
        logger.debug(f"Loaded FITS from {f}")
    except Exception as e:
        logger.error(f"Failed to load FITS for {filename}: {e}")
        raise

    ccdnames = np.unique(corr_table[1].data.ccdname)
    t = fits_table(tractor_dir + "/tractor-forced-" + filename[-13:])

    t.color = -2.5 * (np.log10(t.flux_g / t.flux_i))
    oidmap = dict([((b,o),i) for i,(b,o) in enumerate(zip(t.brickid, t.objid))])
    t_index = np.array([oidmap.get((b,o), -1) for b,o in zip(forced_table[1].data.brickid, forced_table[1].data.objid)])
    dc = t.color[t_index]

    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_x", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
    corr_table[1].data["lm_full_fit_x"] = (corr_table[1].data.rm_full_fit_x).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_y", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
    corr_table[1].data["lm_full_fit_y"] = (corr_table[1].data.rm_full_fit_y).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_dra", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
    corr_table[1].data["lm_full_fit_dra"] = (corr_table[1].data.rm_full_fit_dra).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_ddec", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
    corr_table[1].data["lm_full_fit_ddec"] = (corr_table[1].data.rm_full_fit_ddec).copy()

    logger.info(f"Applying lateral colour model to {filename}")
    missing = set(np.char.strip(np.asarray(ccdnames).astype(str))) - set(model['crpixes'])
    if missing:
        logger.warning(f"No focal-plane position for CCD(s) {sorted(missing)}; their measurements are left uncorrected")
    for filt in ['g','r','i','z']:
        J = np.flatnonzero(np.isin(corr_table[1].data.filter, filt)
                            # apply to every measurement with a valid fit: a systematic shift doesn't depend on S/N,
                            # and correcting only some epochs of a star would add spurious jumps to its PM fit
                            * (corr_table[1].data.full_fit_dra_ivar > 0)
                            * (corr_table[1].data.full_fit_dra != 0.)
                            * np.isfinite(dc))
        if len(J) == 0:
            continue
        xpos = corr_table[1].data.rm_full_fit_x[J]
        ypos = corr_table[1].data.rm_full_fit_y[J]
        ccd = np.char.strip(np.asarray(corr_table[1].data.ccdname[J]).astype(str))
        dx, dy = lateral_shift(model, ccd, filt, xpos, ypos, dc[J])
        xpos, ypos = xpos - dx, ypos - dy

        dDEC = -0.262 * (xpos - corr_table[1].data.rm_full_fit_x[J]) # bc Dustin said so; verified empirically on every CCD
        dRA = +0.262 * (ypos - corr_table[1].data.rm_full_fit_y[J])

        corr_table[1].data["lm_full_fit_x"][J] = xpos
        corr_table[1].data["lm_full_fit_y"][J] = ypos
        corr_table[1].data["lm_full_fit_dra"][J] = corr_table[1].data["rm_full_fit_dra"][J] + dRA
        corr_table[1].data["lm_full_fit_ddec"][J] = corr_table[1].data["rm_full_fit_ddec"][J] + dDEC

    output_path = f"{ corr_dir }{ filename }"
    tmp_path = output_path + ".tmp"
    try:
        corr_table.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, output_path)  # atomic: never leaves a truncated file at output_path
    except Exception as e:
        logger.error(f"Error writing lateral-map-corrected file for brick {filename}: {e}")
        raise
    elapsed = time.time() - start_time
    logger.info(f"Finished brick {filename} in {elapsed:.2f} seconds")


def _output_is_complete(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def create_corrected_data(corr_dir, old_dir, tractor_dir, model, cont=False):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        out = os.path.join(corr_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced' or (cont is True and _output_is_complete(out)):
            continue
        print(f)
        try:
            create_brick_corrected_data(filename, corr_dir, old_dir, tractor_dir, model)
        except Exception as e:
            logger.error(f"Skipping brick {filename} due to error: {e}")
            continue


if __name__ == "__main__":
    if len(sys.argv) != 5:
        logger.error("Invalid arguments. Expected usage: <filename> <corr_dir> <old_dir> <tractor_dir>")
        sys.exit(1)

    filename, corr_dir, old_dir, tractor_dir = sys.argv[1:]
    logger.info(f"Main execution started for {filename}")
    try:
        model = load_lateral_model()
        create_brick_corrected_data(filename, corr_dir, old_dir, tractor_dir, model)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise
    
# def main():  
#     corr_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/rm-corrected-forced-motions/'
#     old_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/dcr-corrected-forced-motions/'

#     fn = 'lateralmaps.fits'
#     G = lateralmaps.get_gaia_data(old_dir)
#     lateralmaps.make_lateralmaps(G, corr_dir + fn)
#     rmaps = lateralmaps.get_lateralmaps(corr_dir + fn)
#     tweaks = lateralmaps.make_splines(rmaps)

#     lateralmaps.create_corrected_data(corr_dir, old_dir, tweaks, cont=True)
