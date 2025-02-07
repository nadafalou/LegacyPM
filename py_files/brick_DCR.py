from astropy.io import fits
import numpy as np
from astropy.time import Time
from astropy.table import Table, vstack
from astropy import units as u
from astrometry.util.fits import os, fits_table
import sys


# DCR_AMPLITUDES = {'r': 2.61, 'g': 40.06} # from Bernstein et. al.
# DCR_AMPLITUDES = {'r': 5.13, 'g': 20.47, 'i': 1, 'z': 1} # DCR amplitude in mas/mag/tan(z), from calculated slopes
DCR_AMPLITUDES = {'r': 7.91, 'g': 29.82, 'i': 1, 'z': 1} # DCR amplitude in mas/mag/tan(z), from calculated slopes but cut out near 0's (Dustin's ntbk)

# # CCD_annotated = fits_table('/pscratch/sd/d/dstn/zpt-dr10-new/ccds-annotated.fits')
# CCD_annotated = fits_table('/pscratch/sd/d/dstn/forced-motions-3/ccds-annotated.fits')
# BPRP_MAP = dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)])
# # CCD_annotated = fits_table('/pscratch/sd/d/dstn/zpt-dr10-new/ccds-annotated.fits')
# # BPRP_MAP.update(dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)]))
# CCD_annotated = fits_table('/pscratch/sd/d/dstn/forced-motions-4/ccds-annotated.fits')
# BPRP_MAP.update(dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)]))

CCD_annotated = fits_table('/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/survey-ccds-decam-dr10-bprp-kd.fits')
BPRP_MAP = dict([((expnum, ccdname), bprp) for expnum,ccdname,bprp in zip(CCD_annotated.expnum, CCD_annotated.ccdname, CCD_annotated.ccdbprp)])


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

        all_COLOUR = -2.5 * (np.log10(tractor.flux_g / tractor.flux_i)) # division bc inside log        
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
        vectorized_mapping = np.vectorize(self._map_char_to_value)

        print(len(self.forced))
        ampl = vectorized_mapping(DCR_AMPLITUDES, np.array(self.forced.filter, dtype=str))

        corr_dp1 = ampl * self.colour * np.sqrt(self.airmass**2 - 1) / 262 # mas to pixels
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
    f = os.path.join(old_dir, filename)
    forced_table = fits.open(f)
    corr_table = forced_table.copy()
    
    if gaia:
        brickDCR = BrickDCR(corr_table[1].data, old_dir + "/gaia-tractor-" + filename[-8:], forced_table=True)
    else:
        brickDCR = BrickDCR(corr_table[1].data, old_dir + "/tractor-forced-" + filename[-13:], forced_table=True)
        
    if len(self.forced.filter) == 0:
        brickDCR.no_dcr()
    else:
        brickDCR.apply_correction()
    # except ValueError:
    #     print("VALUE ERROR - DCR DID NOT WORK")
    #     pass

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
  
    corr_table.writeto(f"{ corr_dir }{ filename }", overwrite=True)
    
    
def create_corrected_data(corr_dir, old_dir, cont=True):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced' or (cont is True and filename in os.listdir(corr_dir)):
            continue
        print(filename)
        
        create_brick_corrected_data(filename, corr_dir, old_dir)


def create_corrected_gaia(corr_dir, old_dir, cont=True):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:11] != 'gaia-forced' or (cont is True and filename in os.listdir(corr_dir)):
            continue
        print(filename)
        
        create_brick_corrected_data(filename, corr_dir, old_dir, gaia=True)

#         forced_table = fits.open(f)
#         corr_table = forced_table.copy()

#         brickDCR = BrickDCR(corr_table[1].data, tractor_file, forced_table=True)
#         brickDCR.apply_correction()
        
#         # print("back to create_corrected_gaia")

#         # print("got ddec and dra")
#         # corr_table = brickDCR.forced.copy()
#         temp_psf = np.flatnonzero(brickDCR.psf_filt)
        
#         # TODO edit to make consistent. Either table['column_name'] or table.column_name
#         dcr_full_fit_x = corr_table[1].data.full_fit_x.copy()
#         dcr_full_fit_x[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_x
#         dcr_full_fit_y = corr_table[1].data.full_fit_y.copy()
#         dcr_full_fit_y[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_y
#         dcr_full_fit_dra = (corr_table[1].data.full_fit_dra).copy()
#         dcr_full_fit_dra[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_dra
#         dcr_full_fit_ddec = (corr_table[1].data.full_fit_ddec).copy()
#         dcr_full_fit_ddec[temp_psf[brickDCR.filt]] = brickDCR.dcr_full_fit_ddec

#         corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_x", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
#         corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_y", array=np.zeros(len(corr_table[1].data)), format='E', unit='pixel'))
#         corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_dra", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
#         corr_table[1].columns.add_col(fits.Column(name="dcr_full_fit_ddec", array=np.zeros(len(corr_table[1].data)), format='E', unit='arcsec'))
#         corr_table[1].data["dcr_full_fit_x"] = dcr_full_fit_x
#         corr_table[1].data["dcr_full_fit_y"] = dcr_full_fit_y
#         corr_table[1].data["dcr_full_fit_dra"] = dcr_full_fit_dra
#         corr_table[1].data["dcr_full_fit_ddec"] = dcr_full_fit_ddec
        
#         corr_table.writeto(f"{ corr_dir }{ filename }", overwrite=True)
#         forced_table.close()

    
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
        print("NO")
    filename, corr_dir, old_dir = sys.argv[1:]
    create_brick_corrected_data(filename, corr_dir, old_dir)
    
#     corr_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/dcr-corrected-forced-motions/'
#     old_dir = '/pscratch/sd/d/dstn/forced-motions-3/'
#     # tractor_filename = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/gaia-tractor.fits'
#     tractor_filename = '/pscratch/sd/d/dstn/forced-motions-3/gaia-cat-all.fits'
    
#     # print("correcting gaia stars...")
#     # create_corrected_gaia(corr_dir, old_dir, tractor_filename)
    
#     print("correcting forced data...")
#     create_corrected_data(corr_dir, old_dir)