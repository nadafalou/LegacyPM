from astropy.io import fits
import numpy as np
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u
from astrometry.util.fits import os, fits_table
import os
import sys
import json
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DCR_AMPLITUDES = {'r': 2.61, 'g': 40.06} # from Bernstein et. al.
# DCR_AMPLITUDES = {'r': 5.13, 'g': 20.47, 'i': 1, 'z': 1} # DCR amplitude in mas/mag/tan(z), from calculated slopes
# DCR_AMPLITUDES = {'r': 7.91, 'g': 29.82, 'i': 1, 'z': 1} # linear model used until 2026-10-23 (Dustin's ntbk)
# Since 2026-10-23: tan z x two straight lines in colour (James's suggestion), per band, fitted by run/dcr-polyfit.ipynb
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

# # CCD_annotated = fits_table('/pscratch/sd/d/dstn/zpt-dr10-new/ccds-annotated.fits')
# CCD_annotated = fits_table('/pscratch/sd/d/dstn/forced-motions-3/ccds-annotated.fits')
# BPRP_MAP = dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)])
# # CCD_annotated = fits_table('/pscratch/sd/d/dstn/zpt-dr10-new/ccds-annotated.fits')
# # BPRP_MAP.update(dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)]))
# CCD_annotated = fits_table('/pscratch/sd/d/dstn/forced-motions-4/ccds-annotated.fits')
# BPRP_MAP.update(dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)]))

CCD_annotated = fits_table('/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/survey-ccds-decam-dr10-bprp-kd.fits')
BPRP_MAP = dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)])


try:
    from .colours import g_minus_i
except ImportError:
    from colours import g_minus_i

# hour_angle and parallactic live in dcr_geometry.py (importable without loading the CCD table above)
try:
    from .dcr_geometry import hour_angle, parallactic
except ImportError:
    from dcr_geometry import hour_angle, parallactic


class BrickDCR:

    def __init__(self, forced_filename, tractor_filename, forced_table=False) -> None:
        if not forced_table: 
            f = fits.open(forced_filename)
            forced = f[1].data
            f.close()
        else: 
            forced = forced_filename
        t = fits.open(tractor_filename)
        tractor = t[1].data
        t.close()
        
        self.psf_filt = forced.x != 0 
        forced_psf = forced[self.psf_filt]
        
        ha = hour_angle(forced_psf.ra, forced_psf.mjd + 0.5 * forced_psf.exptime / (24 * 60 * 60))
        AM, PA = parallactic(forced_psf.dec, ha) # vectorised now, sample brick ran in half the time

        dx = forced_psf.full_fit_x - forced_psf.x
        dy = forced_psf.full_fit_y - forced_psf.y

        dp1 = dx * np.cos(PA) - dy * np.sin(PA) # what we want
        dp2 = dx * np.sin(PA) + dy * np.cos(PA) # throw out - mostly zeroes

        all_COLOUR = g_minus_i(tractor.flux_g, tractor.flux_i, tractor.flux_z)[0]  # measured, else from g - z (colours.py)
        colormap = dict(zip(tractor.objid, all_COLOUR))
        
        COLOUR = []
        for idx in range(len(forced_psf)):
            g_i_colour = colormap.get(forced_psf.objid[idx], 0.)
            bprp_calib = -0.56 + 1.34 * BPRP_MAP[(forced_psf.expnum[idx], forced_psf.ccdname[idx])]
            COLOUR.append(g_i_colour - bprp_calib)
        COLOUR = np.array(COLOUR)
      
        self.filt = np.logical_and(COLOUR > -2, COLOUR < 3) # TODO check trend of colour
        self.forced = forced_psf[self.filt]
        self.airmass = AM[self.filt]
        self.parallactic_angle = PA[self.filt]
        self.dp1 = dp1[self.filt]
        self.dp2 = dp2[self.filt]
        self.colour = COLOUR[self.filt]
        self.expnum = forced_psf.expnum[self.filt]
        self.brickid = forced_psf.brickid[self.filt]
        

    def get_filtered_by_band(self, band, cut_colours=True):
        J = np.flatnonzero(np.isin(self.forced.filter, band))
        
        if cut_colours:
            filt = np.logical_and(self.colour[J] != 0, self.colour[J] > -10)
        else:
            filt = np.ones(len(self.colour[J]))
            
        return self.forced[J][filt], self.colour[J][filt], self.airmass[J][filt], self.dp1[J][filt], self.dp2[J][filt], self.expnum[J][filt], self.brickid[J][filt]
    

    def _map_char_to_value(self, dic, char):
        if char in dic:
            return dic[char]
        else:
            return 0


    def apply_correction(self):
        print(len(self.forced))
        tanz = np.sqrt(self.airmass**2 - 1)
        corr_dp1 = dcr_shift_mas(np.char.strip(np.array(self.forced.filter, dtype=str)), self.colour, tanz) / 262 # mas to pixels
        dcrx = np.cos(self.parallactic_angle) * corr_dp1
        dcry = -np.sin(self.parallactic_angle) * corr_dp1
        dDEC = -0.262 * dcrx
        dRA = +0.262 * dcry
        self.dcr_full_fit_x = self.forced.full_fit_x - dcrx
        self.dcr_full_fit_y = self.forced.full_fit_y - dcry
        self.dcr_full_fit_dra = self.forced.full_fit_dra - dRA
        self.dcr_full_fit_ddec = self.forced.full_fit_ddec - dDEC
        self.colour_airmass = self.colour * np.sqrt(self.airmass**2 - 1) / 262

    def no_dcr(self):
        self.dcr_full_fit_x = self.forced.full_fit_x
        self.dcr_full_fit_y = self.forced.full_fit_y 
        self.dcr_full_fit_dra = self.forced.full_fit_dra 
        self.dcr_full_fit_ddec = self.forced.full_fit_ddec 
        self.colour_airmass = np.zeros(len(self.dcr_full_fit_ddec))
        
    
def create_brick_corrected_data(filename, corr_dir, old_dir, gaia=False):
    logger.info(f"Processing brick: {filename}")
    start_time = time.time()
    
    f = os.path.join(old_dir, filename)
    try:
        forced_table = fits.open(f)
        corr_table = forced_table.copy()
        # date the input was written (decides the Dec-ivar bug correction, error_model.py); archive bricks bring theirs
        if 'INPDATE' not in corr_table[1].header:
            corr_table[1].header['INPDATE'] = (time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(f))),
                                               'date the forced-photometry input was written')
    except Exception as e:
        logger.error(f"Failed to open FITS file {f}: {e}")
        raise

    if gaia:
        logger.debug(f"{filename} - Using Gaia tractor file")
        brickDCR = BrickDCR(corr_table[1].data, old_dir + "/gaia-tractor-" + filename[-8:], forced_table=True)
    else:
        brickDCR = BrickDCR(corr_table[1].data, old_dir + "/tractor-forced-" + filename[-13:], forced_table=True)

    # No longer skipped for bricks without i-band data (until 2026-10-23 they got no DCR at all): objects without a
    # measured g - i get it from g - z (colours.py); objects with neither colour are left uncorrected (filt above).
    logger.info(f"{filename} - Applying DCR correction")
    brickDCR.apply_correction()

    temp_psf = np.flatnonzero(brickDCR.psf_filt)

    dcr_full_fit_x = (corr_table[1].data.full_fit_x).copy()
    dcr_full_fit_x[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_x
    dcr_full_fit_y = (corr_table[1].data.full_fit_y).copy()
    dcr_full_fit_y[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_y
    dcr_full_fit_dra = (corr_table[1].data.full_fit_dra).copy()
    dcr_full_fit_dra[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_dra
    dcr_full_fit_ddec = (corr_table[1].data.full_fit_ddec).copy()
    dcr_full_fit_ddec[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_ddec
    dp1 = np.zeros(len(corr_table[1].data))
    dp1[temp_psf[brickDCR.filt]] = brickDCR.dp1
    colour_airmass = np.zeros(len(corr_table[1].data))
    colour_airmass[temp_psf[brickDCR.filt]] = brickDCR.colour_airmass

    corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_x", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
    corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_y", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
    corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_dra", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
    corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_ddec", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
    corr_table[1].columns.add_col(fits.Column(name="dp1", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].columns.add_col(fits.Column(name="colour_airmass", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].data.dcr_full_fit_x = dcr_full_fit_x
    corr_table[1].data.dcr_full_fit_y = dcr_full_fit_y
    corr_table[1].data.dcr_full_fit_dra = dcr_full_fit_dra
    corr_table[1].data.dcr_full_fit_ddec = dcr_full_fit_ddec
    corr_table[1].data.dp1 = dp1
    corr_table[1].data.colour_airmass = colour_airmass

    output_path = f"{ corr_dir }{ filename }"
    tmp_path = output_path + ".tmp"
    try:
        corr_table.writeto(tmp_path, overwrite=True)
        os.replace(tmp_path, output_path)  # atomic: never leaves a truncated file at output_path
    except Exception as e:
        logger.error(f"Error writing corrected file for brick {filename}: {e}")
        raise
    elapsed = time.time() - start_time
    logger.info(f"Finished processing {filename} in {elapsed:.2f} seconds")


def _output_is_complete(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def create_corrected_data(corr_dir, old_dir, cont=True):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        out = os.path.join(corr_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced' or (cont is True and _output_is_complete(out)):
            continue
        print(filename)
        try:
            create_brick_corrected_data(filename, corr_dir, old_dir)
        except Exception as e:
            logger.error(f"Skipping brick {filename} due to error: {e}")
            continue


def create_corrected_gaia(corr_dir, old_dir, cont=True):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        out = os.path.join(corr_dir, filename)
        if not os.path.isfile(f) or filename[:11] != 'gaia-forced' or (cont is True and _output_is_complete(out)):
            continue
        print(filename)
        try:
            create_brick_corrected_data(filename, corr_dir, old_dir, gaia=True)
        except Exception as e:
            logger.error(f"Skipping brick {filename} due to error: {e}")
            continue

    
def create_gaia_tractor(new_dir, old_dir, cont=True):
    tables = []
    i = 0
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:7] != 'tractor'or (cont is True and filename in os.listdir(corr_dir)):
            continue
        if (i % 100) == 0:  
            print(i)
        # print(filename)

        hdu = fits.open(f)
        table = Table(hdu[1].data)
        tables.append(table[table['ref_cat'] == 'GE'])
        hdu.close()
        i += 1
    
    combined_table = vstack(tables, join_type='outer')
    hdu = fits.BinTableHDU(combined_table)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(f"{ corr_dir }{ filename }", overwrite=True)

    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Invalid number of arguments")
        sys.exit(1)
    filename, corr_dir, old_dir = sys.argv[1:]
    logger.info(f"Starting DCR correction for {filename}")
    total_start = time.time()

    create_brick_corrected_data(filename, corr_dir, old_dir)

    logger.info(f"Total runtime: {time.time() - total_start:.2f} seconds")
    
#     corr_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/dcr-corrected-forced-motions/'
#     old_dir = '/pscratch/sd/d/dstn/forced-motions-3/'
#     # tractor_filename = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/gaia-tractor.fits'
#     tractor_filename = '/pscratch/sd/d/dstn/forced-motions-3/gaia-cat-all.fits'
    
#     # print("correcting gaia stars...")
#     # create_corrected_gaia(corr_dir, old_dir, tractor_filename)
    
#     print("correcting forced data...")
#     create_corrected_data(corr_dir, old_dir)