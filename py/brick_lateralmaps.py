import numpy as np
from astrometry.util.fits import *
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline


BINNING = 16
BY = int(4096 // BINNING)
BX = int(2048 // BINNING)
pcolors = np.array([0.07428406, 0.92169225, 2.33498669, 3.49276543])
median_color = 1.642 # pivot


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
                                * (data.full_fit_dra_ivar > 1)
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


def create_brick_corrected_data(filename, corr_dir, old_dir, tractor_dir, tweaks):
    forced_table = fits.open(f)
    corr_table = forced_table.copy()
    ccdnames = np.unique(corr_table[1].data.ccdname)
    t = fits_table(tractor_dir + "/tractor-forced-" + filename[-13:])

    t.color = -2.5 * (np.log10(t.flux_g / t.flux_i))
    oidmap = dict([((b,o),i) for i,(b,o) in enumerate(zip(t.brickid, t.objid))])
    t_index = np.array([oidmap.get((b,o), -1) for b,o in zip(forced_table[1].data.brickid, forced_table[1].data.objid)])
    dc = t.color[t_index]

    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_x", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].data["lm_full_fit_x"] = (corr_table[1].data.dcr_full_fit_x).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_y", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].data["lm_full_fit_y"] = (corr_table[1].data.dcr_full_fit_y).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_dra", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].data["lm_full_fit_dra"] = (corr_table[1].data.dcr_full_fit_dra).copy()
    corr_table[1].columns.add_col(fits.Column(name="lm_full_fit_ddec", array=np.zeros(len(corr_table[1].data)), format='E'))
    corr_table[1].data["lm_full_fit_ddec"] = (corr_table[1].data.dcr_full_fit_ddec).copy()

    for ccdname in ccdnames:
        for filt in ['g','r','i','z']:
            J = np.flatnonzero((corr_table[1].data.ccdname == ccdname)
                                * np.isin(corr_table[1].data.filter, filt)
                                * (corr_table[1].data.full_fit_dra_ivar > 1e4)
                                * (corr_table[1].data.full_fit_dra != 0.)
                                * (corr_table[1].data.dqmask == 0))

            spline = tweaks[(ccdname, filt)]
            xpos = corr_table[1].data.rm_full_fit_x[J]
            ypos = corr_table[1].data.rm_full_fit_y[J]

            xpos, ypos = xpos - spline[0](xpos, ypos, grid=False) * (dc[J] - median_color), ypos - spline[1](xpos, ypos, grid=False) * (dc[J] - median_color)

            dDEC = -0.262 * (xpos - corr_table[1].data.rm_full_fit_x[J]) # bc Dustin said so
            dRA = +0.262 * (ypos - corr_table[1].data.rm_full_fit_y[J])

            corr_table[1].data["lm_full_fit_x"][J] = xpos
            corr_table[1].data["lm_full_fit_y"][J] = ypos
            corr_table[1].data["lm_full_fit_dra"][J] = corr_table[1].data["rm_full_fit_dra"][J] + dRA
            corr_table[1].data["lm_full_fit_ddec"][J] = corr_table[1].data["rm_full_fit_ddec"][J] + dDEC
    corr_table.writeto(f"{ corr_dir }{ filename }", overwrite=True)

    
def create_corrected_data(corr_dir, old_dir, tractor_dir, tweaks, cont=False):
    for filename in os.listdir(old_dir):
        f = os.path.join(old_dir, filename)
        if not os.path.isfile(f) or filename[:6] != 'forced' or (cont is True and filename in os.listdir(corr_dir)):
            continue
        print(f)
        
        create_brick_corrected_data(filename, )
        create_brick_corrected_data(filename, corr_dir, old_dir, tractor_dir, tweaks)


# def main():  
#     corr_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/rm-corrected-forced-motions/'
#     old_dir = '../../../cfs/cdirs/cosmo/work/users/nelfalou/ls-motions/dcr-corrected-forced-motions/'

#     fn = 'lateralmaps.fits'
#     G = lateralmaps.get_gaia_data(old_dir)
#     lateralmaps.make_lateralmaps(G, corr_dir + fn)
#     rmaps = lateralmaps.get_lateralmaps(corr_dir + fn)
#     tweaks = lateralmaps.make_splines(rmaps)

#     lateralmaps.create_corrected_data(corr_dir, old_dir, tweaks, cont=True)
