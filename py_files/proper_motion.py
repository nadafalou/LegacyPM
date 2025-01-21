from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from random import randint
from legacypipe.py.legacypipe.survey import radec_at_mjd
import math


class BrickCollection:
    """
    A class used to calculate all objects' proper motions in a brick and 
    visualise it, including a visual comparison to Gaia data
    
    
    ===== Attributes =====
    tractor_data: astropy FITS_rec
        fake tractor data (fake = regenerated without proper motions)
    forced_data: astropy FITS_rec
        fake forced photometry data
    real_tractor_data: astropy FITS_rec
        real Gaia tractor data
        
    objects: Python list 
        list of ObjectPM
    pmra: Python list
        list of pmra for objects in the brick
    pmdec: Python list
        list of pmdec for objects in the brick
    pmra_cov: Python list
        list of pmra_cov for objects in the brick
    pmdec_cov: Python list
        list of pmdec_cov for objects in the brick
    mags: Python list
        list of mag for objects in the brick
    
    
    ===== Methods =====
    plot_calculated_vs_gaia_pm()
        plots the calculated pmra and pmdec of all objects vs their respective 
        measured pm from gaia
    plot_cov_vs_mag()
        plots covariance of calculated pm of all objects vs their respective 
        magnitudes    
    """
    
    def __init__(self, brickid, ref_cat=None, num_rand=15, area=None, directory=None) -> None:
        if directory is None:
            self.tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
            self.forced_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/forced/forced-brickwise-{ brickid }.fits')[1].data
        else:
            self.tractor_data = fits.open(f"/pscratch/sd/d/dstn/forced-motions-3/tractor-forced-{ brickid }.fits")[1].data
            self.forced_data = fits.open(f"{ directory }/forced-brickwise-{ brickid }.fits")[1].data
            
        self.real_tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/data/legacysurvey/dr9/south/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        self.brickid = brickid
        
        if ref_cat: 
            data = self.tractor_data[np.logical_and(self.tractor_data['ref_cat'] == ref_cat, self.tractor_data['type'] == 'PSF')]
        else: 
            data = self.tractor_data[self.tractor_data['type'] == 'PSF']
        
        self.objects = []
        self.ff_objects = []
        self.corr_objects = []
        skipped = 0

        f = open(f"/global/homes/n/nelfalou/star_data/{ self.brickid }.csv", "w")
        f.write(f"objid,real_objid,ra,dec,pmra,pmdec,parallax,gaia pmra,gaia pmdec,gaia parallax\n")
        fff = open(f"/global/homes/n/nelfalou/ff_star_data/{ self.brickid }.csv", "w") # sometimes, naming vars takes too much energy 
        fff.write(f"objid,real_objid,ra,dec,pmra,pmdec,parallax,gaia pmra,gaia pmdec,gaia parallax\n")
        corrf = open(f"/global/homes/n/nelfalou/corr_star_data/{ self.brickid }.csv", "w") # sometimes, naming vars takes too much energy 
        corrf.write(f"objid,real_objid,ra,dec,pmra,pmdec,parallax,gaia pmra,gaia pmdec,gaia parallax\n")
        # TEMPORARY: only select brightest 250 stars bc there are too many stars
        snr = (data["flux_r"] * np.sqrt(data["flux_ivar_r"]))**2 + (data["flux_g"] * np.sqrt(data["flux_ivar_g"]))**2 + (data["flux_i"] * np.sqrt(data["flux_ivar_i"]))**2 + (data["flux_z"] * np.sqrt(data["flux_ivar_z"]))**2
        # TODO each source has multiple rows so the loop actually repeats sources a lot
        for object in data[np.argsort(snr)][:400]:
            try:
                obj_pm = ObjectPM(object['ra'], object['dec'], self.tractor_data, self.forced_data, self.real_tractor_data, num_rand=num_rand)
                ff_obj_pm = ffObjectPM(object['ra'], object['dec'], self.tractor_data, self.forced_data, self.real_tractor_data, num_rand=num_rand)
                corr_obj_pm = CorrObjectPM(object['ra'], object['dec'], self.tractor_data, self.forced_data, self.real_tractor_data, num_rand=num_rand)
                if obj_pm.real_tractor_data[obj_pm.real_tractor_data['objid'] == obj_pm.real_objid]['pmra'][0] != 0:
                    self.objects.append(obj_pm)
                    self.ff_objects.append(ff_obj_pm)
                    self.corr_objects.append(corr_obj_pm)

                    f.write(f"{ obj_pm.objid },{ obj_pm.real_objid },{ obj_pm.ra },{ obj_pm.dec },\
                            { obj_pm.pmra },{ obj_pm.pmdec },{ obj_pm.parallax },\
                            { obj_pm.real_tractor_data['pmra'][obj_pm.real_objid] },\
                            { obj_pm.real_tractor_data['pmdec'][obj_pm.real_objid] },\
                            { obj_pm.real_tractor_data['parallax'][obj_pm.real_objid] } \n")
                    
                    fff.write(f"{ ff_obj_pm.objid },{ ff_obj_pm.real_objid },{ ff_obj_pm.ra },{ ff_obj_pm.dec },\
                            { ff_obj_pm.pmra },{ ff_obj_pm.pmdec },{ ff_obj_pm.parallax },\
                            { ff_obj_pm.real_tractor_data['pmra'][ff_obj_pm.real_objid] },\
                            { ff_obj_pm.real_tractor_data['pmdec'][ff_obj_pm.real_objid] },\
                            { ff_obj_pm.real_tractor_data['parallax'][ff_obj_pm.real_objid] } \n")
                    corrf.write(f"{ corr_obj_pm.objid },{ corr_obj_pm.real_objid },{ corr_obj_pm.ra },{ corr_obj_pm.dec },\
                            { corr_obj_pm.pmra },{ corr_obj_pm.pmdec },{ corr_obj_pm.parallax },\
                            { corr_obj_pm.real_tractor_data['pmra'][corr_obj_pm.real_objid] },\
                            { corr_obj_pm.real_tractor_data['pmdec'][corr_obj_pm.real_objid] },\
                            { corr_obj_pm.real_tractor_data['parallax'][corr_obj_pm.real_objid] } \n")
                else:
                    f.write(f"{ obj_pm.objid },{ obj_pm.real_objid },{ obj_pm.ra },{ obj_pm.dec },\
                            { obj_pm.pmra },{ obj_pm.pmdec },{ obj_pm.parallax } \n")
                    fff.write(f"{ ff_obj_pm.objid },{ ff_obj_pm.real_objid },{ ff_obj_pm.ra },{ ff_obj_pm.dec },\
                            { ff_obj_pm.pmra },{ ff_obj_pm.pmdec },{ ff_obj_pm.parallax } \n")
                    corrf.write(f"{ corr_obj_pm.objid },{ corr_obj_pm.real_objid },{ corr_obj_pm.ra },{ corr_obj_pm.dec },\
                            { corr_obj_pm.pmra },{ corr_obj_pm.pmdec },{ corr_obj_pm.parallax } \n")
            except:
                skipped += 1

        # f.write(f"out of { len(data) } stars (type=PSF), { skipped } were skipped \n")
        f.close()
        # fff.write(f"out of { len(data) } stars (type=PSF), { skipped } were skipped \n")
        fff.close()
        # print(f"out of { len(data) } stars (type=PSF), { skipped } were skipped")
        corrf.close()
        
    def get_corr_dx_dy_mags(self, mags_range=(18, 19)):
        try:
            return self.dx, self.dy, self.corr_dx, self.corr_dy, self.mags
        except:
            self.dx = []
            self.dy = []
            self.corr_dx = []
            self.corr_dy = []
            self.mags = []
            for object in self.corr_objects:
                if object.mag >= mags_range[0] and object.mag <= mags_range[1]:
                    psf_filt = object.x != 0
                    self.corr_dx.append((object.corr_ffx - object.x)[psf_filt])
                    self.corr_dy.append((object.corr_ffy - object.y)[psf_filt])
                    self.dx.append((object.ffx - object.x)[psf_filt])
                    self.dy.append((object.ffy - object.y)[psf_filt])
                    # obj_idx = (self.tractor_data.objid == object.objid)
                    # self.mags.append(self.tractor_data.gaia_phot_g_mean_mag[obj_idx[0]])
                    self.mags.append(object.mag)
            self.dx = np.concatenate(self.dx)
            self.dy = np.concatenate(self.dy)
            self.corr_dx = np.concatenate(self.corr_dx)
            self.corr_dy = np.concatenate(self.corr_dy)
            # self.mags = np.array(self.mags)
            return self.dx, self.dy, self.corr_dx, self.corr_dy, np.array(self.mags)
        
    def get_error_and_mags(self):
        try:
            return self.dra_error, self.ddec_error, self.all_mags
        except:
            dra_error = []
            ddec_error = []
            all_mags = []
            for object in self.objects:
                print(object.dra_error)
                dra_error.append(object.dra_error)
                ddec_error.append(object.ddec_error)
                all_mags.append(object.mag)
            self.dra_error = np.concatenate(dra_error)
            self.ddec_error = np.concatenate(ddec_error)
            self.all_mags = np.array(all_mags)
            return self.x, self.y, self.all_mags


    def plot_calculated_vs_gaia_pm(self, figheight=15, figwidth=15):
        self.pmra, self.pmdec, self.pmra_cov, self.pmdec_cov = [], [], [], []
        self.pmra_plx, self.pmdec_plx, self.parallax, parallax2 = [], [], [], []
        gaia_pmra, gaia_pmdec, gaia_pmra_cov, gaia_pmdec_cov, gaia_parallax = [], [], [], [], []
        for object in self.objects:
            self.pmra.append(object.pmra)
            self.pmdec.append(object.pmdec)
            self.pmra_cov.append(object.pmra_cov)
            self.pmdec_cov.append(object.pmdec_cov)
            
            self.pmra_plx.append(object.pmra_plx)
            self.pmdec_plx.append(object.pmdec_plx)
            self.parallax.append(object.parallax)
            parallax2.append(object.parallax_2)

            gaia_pmra.append(object.real_tractor_data['pmra'][object.real_objid])
            gaia_pmdec.append(object.real_tractor_data['pmdec'][object.real_objid])
            gaia_pmra_cov.append(1 / np.sqrt(object.real_tractor_data['pmra_ivar'][object.real_objid]))
            gaia_pmdec_cov.append(1 / np.sqrt(object.real_tractor_data['pmdec_ivar'][object.real_objid]))
            gaia_parallax.append(object.real_tractor_data['parallax'][object.real_objid])

        # ra_p, ra_V = np.polyfit(gaia_pmra, self.pmra, 1, w=1 / np.array(self.pmra_cov), cov=True)
        # ra_slope, ra_slope_cov = ra_p[0], ra_V[0][0]
        minmax_pmra = [min(gaia_pmra), max(gaia_pmra)]
        
        # dec_p, dec_V = np.polyfit(gaia_pmdec, self.pmdec, 1, w=1 / np.array(self.pmdec_cov), cov=True)
        # dec_slope, dec_slope_cov = dec_p[0], dec_V[0][0]
        minmax_pmdec = [min(gaia_pmdec), max(gaia_pmdec)]
        
        minmax_parallax = [min(gaia_parallax), max(gaia_parallax)]
        
        fig, ax = plt.subplots(3, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        ax[0, 0].errorbar(gaia_pmra, self.pmra, fmt='o', yerr=self.pmra_cov, xerr=gaia_pmra_cov, ms=3)
        ax[0, 0].set_xlabel("gaia pmra")
        ax[0, 0].set_ylabel(f"calculated pmra, { sum(np.logical_or(self.pmra < minmax_pmra[0], self.pmra > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[0, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[0, 0].legend()
        ax[0, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[0, 0].set_title("Calculated vs Gaia pmra (err on x- & y-axis)")
        
        ax[0, 1].errorbar(gaia_pmdec, self.pmdec, fmt='o', yerr=self.pmdec_cov, xerr=gaia_pmdec_cov, ms=3)
        ax[0, 1].set_xlabel("gaia pmdec")
        ax[0, 1].set_ylabel(f"calculated pmdec, { sum(np.logical_or(self.pmdec < minmax_pmdec[0], self.pmdec > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[0, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[0, 1].legend()
        ax[0, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[0, 1].set_title("Calculated vs Gaia pmdec (err on x- & y-axis)")
        
        ax[1, 0].scatter(gaia_pmra, self.pmra_plx, s=3)
        ax[1, 0].set_xlabel("gaia pmra")
        ax[1, 0].set_ylabel(f"calculated pmra w/ plx, { sum(np.logical_or(self.pmra_plx < minmax_pmra[0], self.pmra_plx > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[1, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[1, 0].legend()
        ax[1, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[1, 0].set_title("Calculated w/ plx vs Gaia pmra (no error bars)")
        
        ax[1, 1].scatter(gaia_pmdec, self.pmdec_plx, s=3)
        ax[1, 1].set_xlabel("gaia pmdec")
        ax[1, 1].set_ylabel(f"calculated pmdec w/ plx, { sum(np.logical_or(self.pmdec_plx < minmax_pmdec[0], self.pmdec_plx > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[1, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[1, 1].legend()
        ax[1, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[1, 1].set_title("Calculated w/ plx vs Gaia pmdec (no error bars)")
        
        ax[2, 0].scatter(gaia_parallax, self.parallax, s=3)
        ax[2, 0].set_xlabel("gaia parallax")
        ax[2, 0].set_ylabel(f"calculated parallax, { sum(np.logical_or(self.parallax < minmax_parallax[0], self.parallax > minmax_parallax[1])) } outbounds")
        ax[2, 0].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 0].legend()
        ax[2, 0].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 0].set_title("Calculated vs Gaia parallax (no error bars)")
        
        ax[2, 1].scatter(gaia_parallax, parallax2, s=3)
        ax[2, 1].set_xlabel("gaia parallax")
        ax[2, 1].set_ylabel(f"calculated parallax, { sum(np.logical_or(parallax2 < minmax_parallax[0], parallax2 > minmax_pmra[1])) } outbounds")
        ax[2, 1].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 1].legend()
        ax[2, 1].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 1].set_title("Calculated vs Gaia parallax (trial 2, no error bars)")

        fig.savefig(f"/global/homes/n/nelfalou/compare_gaia/{ self.brickid }.png")

        # print("pmra slope: ", ra_slope, "with cov ", ra_slope_cov)
        # print("pmdec slope: ", dec_slope, "with cov ", dec_slope_cov)

    def ff_plot_calculated_vs_gaia_pm(self, figheight=15, figwidth=15):
        self.ff_pmra, self.ff_pmdec, self.ff_pmra_cov, self.ff_pmdec_cov = [], [], [], []
        self.ff_pmra_plx, self.ff_pmdec_plx, self.ff_parallax, parallax2 = [], [], [], []
        gaia_pmra, gaia_pmdec, gaia_pmra_cov, gaia_pmdec_cov, gaia_parallax = [], [], [], [], []
        for object in self.ff_objects:
            self.ff_pmra.append(object.pmra)
            self.ff_pmdec.append(object.pmdec)
            self.ff_pmra_cov.append(object.pmra_cov)
            self.ff_pmdec_cov.append(object.pmdec_cov)
            
            self.ff_pmra_plx.append(object.pmra_plx)
            self.ff_pmdec_plx.append(object.pmdec_plx)
            self.ff_parallax.append(object.parallax)
            parallax2.append(object.parallax_2)

            gaia_pmra.append(object.real_tractor_data['pmra'][object.real_objid])
            gaia_pmdec.append(object.real_tractor_data['pmdec'][object.real_objid])
            gaia_pmra_cov.append(1 / np.sqrt(object.real_tractor_data['pmra_ivar'][object.real_objid]))
            gaia_pmdec_cov.append(1 / np.sqrt(object.real_tractor_data['pmdec_ivar'][object.real_objid]))
            gaia_parallax.append(object.real_tractor_data['parallax'][object.real_objid])

        # ra_p, ra_V = np.polyfit(gaia_pmra, self.ff_pmra, 1, w=1 / np.array(self.ff_pmra_cov), cov=True)
        # ra_slope, ra_slope_cov = ra_p[0], ra_V[0][0]
        minmax_pmra = [min(gaia_pmra), max(gaia_pmra)]
        
        # dec_p, dec_V = np.polyfit(gaia_pmdec, self.ff_pmdec, 1, w=1 / np.array(self.ff_pmdec_cov), cov=True)
        # dec_slope, dec_slope_cov = dec_p[0], dec_V[0][0]
        minmax_pmdec = [min(gaia_pmdec), max(gaia_pmdec)]
        
        minmax_parallax = [min(gaia_parallax), max(gaia_parallax)]
        
        fig, ax = plt.subplots(3, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        ax[0, 0].errorbar(gaia_pmra, self.ff_pmra, fmt='o', yerr=self.ff_pmra_cov, xerr=gaia_pmra_cov, ms=3)
        ax[0, 0].set_xlabel("gaia pmra")
        ax[0, 0].set_ylabel(f"calculated pmra, { sum(np.logical_or(self.ff_pmra < minmax_pmra[0], self.ff_pmra > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[0, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[0, 0].legend()
        ax[0, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[0, 0].set_title("Calculated vs Gaia pmra (err on x- & y-axis)")
        
        ax[0, 1].errorbar(gaia_pmdec, self.ff_pmdec, fmt='o', yerr=self.ff_pmdec_cov, xerr=gaia_pmdec_cov, ms=3)
        ax[0, 1].set_xlabel("gaia pmdec")
        ax[0, 1].set_ylabel(f"calculated pmdec, { sum(np.logical_or(self.ff_pmdec < minmax_pmdec[0], self.ff_pmdec > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[0, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[0, 1].legend()
        ax[0, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[0, 1].set_title("Calculated vs Gaia pmdec (err on x- & y-axis)")
        
        ax[1, 0].scatter(gaia_pmra, self.ff_pmra_plx, s=3)
        ax[1, 0].set_xlabel("gaia pmra")
        ax[1, 0].set_ylabel(f"calculated pmra w/ plx, { sum(np.logical_or(self.ff_pmra_plx < minmax_pmra[0], self.ff_pmra_plx > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[1, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[1, 0].legend()
        ax[1, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[1, 0].set_title("Calculated w/ plx vs Gaia pmra (no error bars)")
        
        ax[1, 1].scatter(gaia_pmdec, self.ff_pmdec_plx, s=3)
        ax[1, 1].set_xlabel("gaia pmdec")
        ax[1, 1].set_ylabel(f"calculated pmdec w/ plx, { sum(np.logical_or(self.ff_pmdec_plx < minmax_pmdec[0], self.ff_pmdec_plx > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[1, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[1, 1].legend()
        ax[1, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[1, 1].set_title("Calculated w/ plx vs Gaia pmdec (no error bars)")
        
        ax[2, 0].scatter(gaia_parallax, self.ff_parallax, s=3)
        ax[2, 0].set_xlabel("gaia parallax")
        ax[2, 0].set_ylabel(f"calculated parallax, { sum(np.logical_or(self.ff_parallax < minmax_parallax[0], self.ff_parallax > minmax_parallax[1])) } outbounds")
        ax[2, 0].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 0].legend()
        ax[2, 0].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 0].set_title("Calculated vs Gaia parallax (no error bars)")
        
        ax[2, 1].scatter(gaia_parallax, parallax2, s=3)
        ax[2, 1].set_xlabel("gaia parallax")
        ax[2, 1].set_ylabel(f"calculated parallax, { sum(np.logical_or(parallax2 < minmax_parallax[0], parallax2 > minmax_pmra[1])) } outbounds")
        ax[2, 1].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 1].legend()
        ax[2, 1].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 1].set_title("Calculated vs Gaia parallax (trial 2, no error bars)")

        fig.savefig(f"/global/homes/n/nelfalou/ff_compare_gaia/{ self.brickid }.png")

        # print("pmra slope: ", ra_slope, "with cov ", ra_slope_cov)
        # print("pmdec slope: ", dec_slope, "with cov ", dec_slope_cov)
        
    def corr_plot_calculated_vs_gaia_pm(self, figheight=15, figwidth=15):
        self.corr_pmra, self.corr_pmdec, self.corr_pmra_cov, self.corr_pmdec_cov = [], [], [], []
        self.corr_pmra_plx, self.corr_pmdec_plx, self.corr_parallax, parallax2 = [], [], [], []
        gaia_pmra, gaia_pmdec, gaia_pmra_cov, gaia_pmdec_cov, gaia_parallax = [], [], [], [], []
        for object in self.corr_objects:
            self.corr_pmra.append(object.pmra)
            self.corr_pmdec.append(object.pmdec)
            self.corr_pmra_cov.append(object.pmra_cov)
            self.corr_pmdec_cov.append(object.pmdec_cov)
            
            self.corr_pmra_plx.append(object.pmra_plx)
            self.corr_pmdec_plx.append(object.pmdec_plx)
            self.corr_parallax.append(object.parallax)
            parallax2.append(object.parallax_2)

            gaia_pmra.append(object.real_tractor_data['pmra'][object.real_objid])
            gaia_pmdec.append(object.real_tractor_data['pmdec'][object.real_objid])
            gaia_pmra_cov.append(1 / np.sqrt(object.real_tractor_data['pmra_ivar'][object.real_objid]))
            gaia_pmdec_cov.append(1 / np.sqrt(object.real_tractor_data['pmdec_ivar'][object.real_objid]))
            gaia_parallax.append(object.real_tractor_data['parallax'][object.real_objid])

        # ra_p, ra_V = np.polyfit(gaia_pmra, self.corr_pmra, 1, w=1 / np.array(self.corr_pmra_cov), cov=True)
        # ra_slope, ra_slope_cov = ra_p[0], ra_V[0][0]
        minmax_pmra = [min(gaia_pmra), max(gaia_pmra)]
        
        # dec_p, dec_V = np.polyfit(gaia_pmdec, self.corr_pmdec, 1, w=1 / np.array(self.corr_pmdec_cov), cov=True)
        # dec_slope, dec_slope_cov = dec_p[0], dec_V[0][0]
        minmax_pmdec = [min(gaia_pmdec), max(gaia_pmdec)]
        
        minmax_parallax = [min(gaia_parallax), max(gaia_parallax)]
        
        fig, ax = plt.subplots(3, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        ax[0, 0].errorbar(gaia_pmra, self.corr_pmra, fmt='o', yerr=self.corr_pmra_cov, xerr=gaia_pmra_cov, ms=3)
        ax[0, 0].set_xlabel("gaia pmra")
        ax[0, 0].set_ylabel(f"calculated pmra, { sum(np.logical_or(self.corr_pmra < minmax_pmra[0], self.corr_pmra > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[0, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[0, 0].legend()
        ax[0, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[0, 0].set_title("Calculated vs Gaia pmra (err on x- & y-axis)")
        
        ax[0, 1].errorbar(gaia_pmdec, self.corr_pmdec, fmt='o', yerr=self.corr_pmdec_cov, xerr=gaia_pmdec_cov, ms=3)
        ax[0, 1].set_xlabel("gaia pmdec")
        ax[0, 1].set_ylabel(f"calculated pmdec, { sum(np.logical_or(self.corr_pmdec < minmax_pmdec[0], self.corr_pmdec > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[0, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[0, 1].legend()
        ax[0, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[0, 1].set_title("Calculated vs Gaia pmdec (err on x- & y-axis)")
        
        ax[1, 0].scatter(gaia_pmra, self.corr_pmra_plx, s=3)
        ax[1, 0].set_xlabel("gaia pmra")
        ax[1, 0].set_ylabel(f"calculated pmra w/ plx, { sum(np.logical_or(self.corr_pmra_plx < minmax_pmra[0], self.corr_pmra_plx > minmax_pmra[1])) } outbounds")
        # ax[0].plot(gaia_pmra, np.poly1d(ra_p)(gaia_pmra), label="trend")
        ax[1, 0].plot(minmax_pmra, minmax_pmra, label="slope=1", c='c')
        ax[1, 0].legend()
        ax[1, 0].set_ylim(minmax_pmra[0] - 10, minmax_pmra[1] + 10)
        ax[1, 0].set_title("Calculated w/ plx vs Gaia pmra (no error bars)")
        
        ax[1, 1].scatter(gaia_pmdec, self.corr_pmdec_plx, s=3)
        ax[1, 1].set_xlabel("gaia pmdec")
        ax[1, 1].set_ylabel(f"calculated pmdec w/ plx, { sum(np.logical_or(self.corr_pmdec_plx < minmax_pmdec[0], self.corr_pmdec_plx > minmax_pmdec[1])) } outbounds")
        # ax[1].plot(gaia_pmdec, np.poly1d(dec_p)(gaia_pmdec), label="trend")
        ax[1, 1].plot(minmax_pmdec, minmax_pmdec, label="slope=1", c='c')
        ax[1, 1].legend()
        ax[1, 1].set_ylim(minmax_pmdec[0] - 10, minmax_pmdec[1] + 10)
        ax[1, 1].set_title("Calculated w/ plx vs Gaia pmdec (no error bars)")
        
        ax[2, 0].scatter(gaia_parallax, self.corr_parallax, s=3)
        ax[2, 0].set_xlabel("gaia parallax")
        ax[2, 0].set_ylabel(f"calculated parallax, { sum(np.logical_or(self.corr_parallax < minmax_parallax[0], self.corr_parallax > minmax_parallax[1])) } outbounds")
        ax[2, 0].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 0].legend()
        ax[2, 0].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 0].set_title("Calculated vs Gaia parallax (no error bars)")
        
        ax[2, 1].scatter(gaia_parallax, parallax2, s=3)
        ax[2, 1].set_xlabel("gaia parallax")
        ax[2, 1].set_ylabel(f"calculated parallax, { sum(np.logical_or(parallax2 < minmax_parallax[0], parallax2 > minmax_pmra[1])) } outbounds")
        ax[2, 1].plot(minmax_parallax, minmax_parallax, label="slope=1", c='c')
        ax[2, 1].legend()
        ax[2, 1].set_ylim(minmax_parallax[0] - 10, minmax_parallax[1] + 10)
        ax[2, 1].set_title("Calculated vs Gaia parallax (trial 2, no error bars)")

        fig.savefig(f"/global/homes/n/nelfalou/corr_compare_gaia/{ self.brickid }.png")

        # print("pmra slope: ", ra_slope, "with cov ", ra_slope_cov)
        # print("pmdec slope: ", dec_slope, "with cov ", dec_slope_cov)

    def plot_cov_vs_mag(self):
        self.mags = []
        for object in self.objects:
            self.mags.append(object.tractor_data['gaia_phot_rp_mean_mag'][object.objid])
        fig, ax = plt.subplots()
        ax.scatter(self.pmra_cov, self.mags, s=3, label="pmra")
        ax.scatter(self.pmdec_cov, self.mags, s=3, label="pmdec")
        ax.set_xlabel("pm cov")
        ax.set_ylabel("mag")
        ax.legend()
        ax.set_xscale('log')
        ax.set_title("Covariance of pm compared to magnitude")


class ObjectPM:
    """
    A class used to calculate an object's proper motion and visualise it, 
    including a visual comparison to Gaia data
    
    
    ===== Attributes =====
    tractor_data: astropy FITS_rec
        fake tractor data (fake = regenerated without proper motions)
    forced_data: astropy FITS_rec
        fake forced photometry data
    real_tractor_data: astropy FITS_rec
        real Gaia tractor data
        
    ra: float
        right ascension of object
    dec: float
        declination of object
    objid: int
        object ID in tractor and forced datasets (not in gaia!)
    real_objid: int
        object ID in real (gaia) tractor dataset 
    forced_objid_filt: bool array
        filter array according to objid 
        
    mjd: int array
        array of modified Julian dates object was observed on
    dra: float array
        change in ra of object on corresponding mjd
    dra_ivar: float array
        ivar of dra 
    ddec: float array
        change in dec of object on corresponding mjd
    ddec_ivar: float array
        ivar of ddec
    dra_error: float array
        error of dra
    ddec_error: float array
        error of ddec
        
    filter: str array
        array of exposures filter (g, i, r or z)
    filter_names: str array
        array of possible filter names (this would only be ~4 elements, 
        extracted in case filters are added)
    psfdepth: float array
        psfdepth of exposures
        
    dra_trend_func: numpy poly1d
        linear fit of object dra
    pmra: float
        proper motion in ra calculated from linear fit
    ddec_trend_func: numpy poly1d
        linear trend of object ddec
    pmdec: float
        proper motion in dec calculated from linear fit
    
    
    ===== Methods =====
    _generate_random_filter(num_rand, length)
        generates a filter of length <length> with <num_rand> Trues at random 
        indices in the array
    get_pmra_pmdec()
        prints pmra and pmdec of object with their covariances and the Gaia-measured
    plot_dra_and_ddec(alpha=0.1, ylim=None)
        plots dra and ddec through time (mjd) with their linear trend lines
    plot_ddec_vs_dra(xlim=None, ylim=None)
        plots ddec vs dra of object
    plot_ddec_vs_dra_filtered(xlim=None, ylim=None, figheight=15, figwidth=15)
        plots ddec vs dra from different esposure filters, one plot per filter
    plot_psfdepth_vs_error(figheight=15, figwidth=15)
        plots pdfdepth vs error of pm from different esposure filters, one plot per filter
    plot_with_real_gaia(dra_lim=None, ddec_lim=None, figheight=15, figwidth=15)
        plots dra and ddec, and gaia dra and ddec through time (mjd), one plot for each of
        dra and ddec
    """
    
    def __init__(self, ra, dec, tractor_data, forced_data, real_tractor_data, error_hyp=True, num_rand=None, brickid=None): # purposely not passing in objid bc seems different in real vs fake data
        self.ra = ra
        self.dec = dec
        
        if isinstance(tractor_data, str):
            self.tractor_data = fits.open(tractor_data)[1].data
            self.forced_data = fits.open(forced_data)[1].data
            self.real_tractor_data = fits.open(real_tractor_data)[1].data
        else:
            self.tractor_data = tractor_data
            self.forced_data = forced_data
            self.real_tractor_data = real_tractor_data
        # else: 
        #     self.tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        #     self.forced_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/forced/forced-brickwise-{ brickid }.fits')[1].data
        #     self.real_tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/data/legacysurvey/dr9/south/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        
        self.objid = np.argmin(np.hypot(self.ra - self.tractor_data['ra'], self.dec - self.tractor_data['dec']))
        self.real_objid = np.argmin(np.hypot(self.ra - self.real_tractor_data['ra'], self.dec - self.real_tractor_data['dec']))
        self.forced_objid_filt = np.logical_and(np.logical_and((self.forced_data['objid'] == self.objid), self.forced_data['dra_ivar'] != 0), self.forced_data['ddec_ivar'] != 0)
        if num_rand: # TODO if num_rand > size
            self.rand_filt = self._generate_random_filter(num_rand, sum(self.forced_objid_filt))
        else:
            self.rand_filt = np.ones(sum(self.forced_objid_filt), dtype=bool)  

        self.mjd = self.forced_data['mjd'][self.forced_objid_filt][self.rand_filt]
        self.dra = self.forced_data['dra'][self.forced_objid_filt][self.rand_filt]
        self.ddec = self.forced_data['ddec'][self.forced_objid_filt][self.rand_filt]
        self.filter = self.forced_data['filter'][self.forced_objid_filt][self.rand_filt]
        self.psfdepth = (self.forced_data['psfdepth'][self.forced_objid_filt])[self.rand_filt]

        self.dra_ivar = self.forced_data['dra_ivar'][self.forced_objid_filt][self.rand_filt]
        self.ddec_ivar = self.forced_data['ddec_ivar'][self.forced_objid_filt][self.rand_filt]
        self.dra_error = 1 / np.sqrt(self.dra_ivar)
        self.ddec_error = 1 / np.sqrt(self.ddec_ivar)
        if error_hyp:
            self.dra_error = np.hypot(self.dra_error, 0.005)
            self.ddec_error = np.hypot(self.ddec_error, 0.005)
            self.dra_ivar = 1 / self.dra_error ** 2 
            self.ddec_ivar = 1 / self.ddec_error ** 2 
                
        self._calculate_pm()
        
        self.filter_names = None

    def _generate_random_filter(self, num_rand, length):
        rand_filt = np.zeros(length)
        for i in range(num_rand):
            rand_num = randint(0, length)
            while rand_filt[rand_num] == 1:
                rand_num = randint(0, length)
            rand_filt[rand_num] = 1
        return rand_filt.astype(bool)

    def _calculate_pm(self):
        # plain slope fit
        ra_p, ra_V = np.polyfit(self.mjd, self.dra, 1, w=np.sqrt(self.dra_ivar), cov=True)
        self.dra_trend_func = np.poly1d(ra_p)
        self.pmra, self.pmra_cov = ra_p[0] * 365 * 1000, ra_V[0][0] * 365 * 1000

        dec_p, dec_V = np.polyfit(self.mjd, self.ddec, 1, w=np.sqrt(self.ddec_ivar), cov=True)
        self.ddec_trend_func = np.poly1d(dec_p)
        self.pmdec, self.pmdec_cov = dec_p[0] * 365 * 1000, dec_V[0][0] * 365 * 1000
        
        # optimise pm and plx simultaneuously
        daysperyear = 365.2425
        ref_mjd = 57174 # this is 2015.5 = ref_year
        _, _, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid] # TODO ref_year from forced?
        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, 0, 0, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 3))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec
        A[:N, 1] = dt # / sigma_dra
        A[N:, 2] = dt # /sigma_ddec
        
        B = np.append(self.dra, self.ddec) # divide by sigma_dra/ddec

        X, resid, rank, s = np.linalg.lstsq(A, B, rcond=None)
        X = X * 1000
        self.parallax, self.pmra_plx, self.pmdec_plx = X[0], X[1], X[2]
        
        
        # optimise plx using lin fit pm        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, self.pmra, self.pmdec, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 1))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec

        X, resid, rank, s = np.linalg.lstsq(A, np.append(self.dra, self.ddec), rcond=None)
        X = X * 1000
        self.parallax_2 = X[0]
        
    def get_pmra_pmdec(self):        
        print(f"pmra = { self.pmra } with covariance { self.pmra_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmra'][0] } from Gaia")
        print(f"pmdec = { self.pmdec } with covariance { self.pmdec_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmdec'][0] } from Gaia")
        
        print(f"FIT WITH PARALLAX: \n pmra = { self.pmra_plx }, pmdec = { self.pmdec_plx }, parallax = { self.parallax }, Gaia parallax = { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['parallax'][0] }")

    def plot_dra_and_ddec(self, alpha=0.1, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.mjd, self.dra, s=5, label='dra', alpha=alpha, c='c')
        ax.scatter(self.mjd, self.ddec, s=5, label='ddec', alpha=alpha, c='m')
        ax.plot(self.mjd, self.dra_trend_func(self.mjd), c='c')
        ax.plot(self.mjd, self.ddec_trend_func(self.mjd), c='m')
        ax.set_xlabel('mjd')
        ax.set_ylabel('dra and ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
        ax.set_title("Change in ra and dec through time")
        
    def plot_ddec_vs_dra(self, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.dra, self.ddec, c=self.mjd, s=3)
        ax.set_xlabel('dra')
        ax.set_ylabel('ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        
    def plot_ddec_vs_dra_filtered(self, xlim=None, ylim=None, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra[filter_filter], self.ddec[filter_filter], c=self.mjd[filter_filter], s=3)
            ax[row, col].set_xlabel('dra')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])
            if xlim:
                ax[row, col].set_xlim(xlim[0], xlim[1])
            
            col += 1
        fig.show()
        
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.mjd[filter_filter], self.ddec[filter_filter], s=3)
            ax[row, col].set_xlabel('mjd')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])           
            col += 1
        fig.show()
        
    def plot_psfdepth_vs_error(self, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].scatter(self.ddec_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].set_xlabel('dec error')
            ax[row, col].set_ylabel('psfdepth')
            ax[row, col].set_title(f"{ f } filter")
            ax[row, col].set_xscale('log')
            
            col += 1
        
    def plot_with_real_gaia(self, dra_lim=None, ddec_lim=None, figheight=15, figwidth=15):
        obj_ra, obj_dec, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid]
        pmra, pmdec, parallax = self.real_tractor_data['pmra'][self.real_objid], self.real_tractor_data['pmdec'][self.real_objid], self.real_tractor_data['parallax'][self.real_objid]

        proj_ra = []
        proj_dec = []
        for date in self.mjd:
            ra, dec = radec_at_mjd(obj_ra, obj_dec, ref_year, pmra, pmdec, parallax, date)
            proj_dec.append(dec)
            proj_ra.append(ra)
        proj_dra = (np.array(proj_ra) - obj_ra) * 3600
        proj_ddec = (np.array(proj_dec) - obj_dec) * 3600
        
        fig, ax = plt.subplots(3)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0].scatter(self.dra, self.ddec, c=self.mjd, s=3, label="Measured")
        ax[0].scatter(proj_dra, proj_ddec, s=3, c='r', label="Projected (Gaia)")
        ax[0].set_xlabel('dra')
        ax[0].set_ylabel('ddec')
        if dra_lim: ax[0].set_xlim(dra_lim[0], dra_lim[1])
        if ddec_lim: ax[0].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[0].set_title("Projected (Gaia) and measured change in location")
        
        ax[1].errorbar(self.mjd, self.dra, fmt='o', alpha=0.1, yerr=self.dra_error, label="Measured")
        ax[1].scatter(self.mjd, proj_dra, s=3, c='r', label="Projected (Gaia)")
        if dra_lim: ax[1].set_ylim(dra_lim[0], dra_lim[1])
        ax[1].set_xlabel('mjd')
        ax[1].set_ylabel('dra')
        ax[1].set_title("Projected (Gaia) and measured change in ra, error bars on y-axis")
        
        ax[2].errorbar(self.mjd, self.ddec, fmt='o', alpha=0.1, yerr=self.ddec_error, label="Measured")
        ax[2].scatter(self.mjd, proj_ddec, s=3, c='r', label="Projeced (Gaia)")
        if ddec_lim: ax[2].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[2].set_xlabel('mjd')
        ax[2].set_ylabel('ddec')
        ax[2].set_title("Projected (Gaia) and measured change in dec, error bars on y-axis")

    def plot_dra_ddec_vs_mjd_coloured(self, dra_lim=None, ddec_lim=None, figheight=30, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        fig, ax = plt.subplots(4, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0, 0].scatter(self.mjd, self.dra, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 0].set_xlabel('mjd')
        ax[0, 0].set_ylabel('dra')
        if dra_lim: ax[0, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[0, 0].set_title("Colour = airmass")
        
        ax[0, 1].scatter(self.mjd, self.ddec, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 1].set_xlabel('mjd')
        ax[0, 1].set_ylabel('ddec')
        if ddec_lim: ax[0, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        filters = np.zeros(self.filter.shape)
        for i, f in enumerate(self.filter_names):
            filters[self.filter == f] =int(i)
        
        ax[1, 0].scatter(self.mjd, self.dra, c=filters, s=3, alpha=0.5)
        ax[1, 0].set_xlabel('mjd')
        ax[1, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[1, 0].set_title("Colour = filters")
        
        ax[1, 1].scatter(self.mjd, self.ddec, c=filters, s=3, alpha=0.5)
        ax[1, 1].set_xlabel('mjd')
        ax[1, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[2, 0].scatter(self.mjd, self.dra, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 0].set_xlabel('mjd')
        ax[2, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[2, 0].set_title("Colour = flux")
        
        ax[2, 1].scatter(self.mjd, self.ddec, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 1].set_xlabel('mjd')
        ax[2, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[3, 0].scatter(self.mjd, self.dra, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 0].set_xlabel('mjd')
        ax[3, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[3, 0].set_title("Colour = exptime")
        
        ax[3, 1].scatter(self.mjd, self.ddec, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 1].set_xlabel('mjd')
        ax[3, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
      

class ffObjectPM: # FULL FIT
    """
    A class used to calculate an object's proper motion and visualise it, 
    including a visual comparison to Gaia data
    
    
    ===== Attributes =====
    tractor_data: astropy FITS_rec
        fake tractor data (fake = regenerated without proper motions)
    forced_data: astropy FITS_rec
        fake forced photometry data
    real_tractor_data: astropy FITS_rec
        real Gaia tractor data
        
    ra: float
        right ascension of object
    dec: float
        declination of object
    objid: int
        object ID in tractor and forced datasets (not in gaia!)
    real_objid: int
        object ID in real (gaia) tractor dataset 
    forced_objid_filt: bool array
        filter array according to objid 
        
    mjd: int array
        array of modified Julian dates object was observed on
    dra: float array
        change in ra of object on corresponding mjd
    dra_ivar: float array
        ivar of dra 
    ddec: float array
        change in dec of object on corresponding mjd
    ddec_ivar: float array
        ivar of ddec
    dra_error: float array
        error of dra
    ddec_error: float array
        error of ddec
        
    filter: str array
        array of exposures filter (g, i, r or z)
    filter_names: str array
        array of possible filter names (this would only be ~4 elements, 
        extracted in case filters are added)
    psfdepth: float array
        psfdepth of exposures
        
    dra_trend_func: numpy poly1d
        linear fit of object dra
    pmra: float
        proper motion in ra calculated from linear fit
    ddec_trend_func: numpy poly1d
        linear trend of object ddec
    pmdec: float
        proper motion in dec calculated from linear fit
    
    
    ===== Methods =====
    _generate_random_filter(num_rand, length)
        generates a filter of length <length> with <num_rand> Trues at random 
        indices in the array
    get_pmra_pmdec()
        prints pmra and pmdec of object with their covariances and the Gaia-measured
    plot_dra_and_ddec(alpha=0.1, ylim=None)
        plots dra and ddec through time (mjd) with their linear trend lines
    plot_ddec_vs_dra(xlim=None, ylim=None)
        plots ddec vs dra of object
    plot_ddec_vs_dra_filtered(xlim=None, ylim=None, figheight=15, figwidth=15)
        plots ddec vs dra from different esposure filters, one plot per filter
    plot_psfdepth_vs_error(figheight=15, figwidth=15)
        plots pdfdepth vs error of pm from different esposure filters, one plot per filter
    plot_with_real_gaia(dra_lim=None, ddec_lim=None, figheight=15, figwidth=15)
        plots dra and ddec, and gaia dra and ddec through time (mjd), one plot for each of
        dra and ddec
    """
    
    def __init__(self, ra, dec, tractor_data, forced_data, real_tractor_data, error_hyp=True, num_rand=None, brickid=None): # purposely not passing in objid bc seems different in real vs fake data
        self.ra = ra
        self.dec = dec
        
        if isinstance(tractor_data, str):
            self.tractor_data = fits.open(tractor_data)[1].data
            self.forced_data = fits.open(forced_data)[1].data
            self.real_tractor_data = fits.open(real_tractor_data)[1].data
        else:
            self.tractor_data = tractor_data
            self.forced_data = forced_data
            self.real_tractor_data = real_tractor_data
        # else: 
        #     self.tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        #     self.forced_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/forced/forced-brickwise-{ brickid }.fits')[1].data
        #     self.real_tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/data/legacysurvey/dr9/south/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        
        self.objid = np.argmin(np.hypot(self.ra - self.tractor_data['ra'], self.dec - self.tractor_data['dec']))
        self.real_objid = np.argmin(np.hypot(self.ra - self.real_tractor_data['ra'], self.dec - self.real_tractor_data['dec']))
        self.forced_objid_filt = np.logical_and(np.logical_and((self.forced_data['objid'] == self.objid), self.forced_data['full_fit_dra_ivar'] != 0), self.forced_data['full_fit_ddec_ivar'] != 0)
        if num_rand: # TODO if num_rand > size
            self.rand_filt = self._generate_random_filter(num_rand, sum(self.forced_objid_filt))
        else:
            self.rand_filt = np.ones(sum(self.forced_objid_filt), dtype=bool)  

        self.mjd = self.forced_data['mjd'][self.forced_objid_filt][self.rand_filt]
        self.dra = self.forced_data['full_fit_dra'][self.forced_objid_filt][self.rand_filt]
        self.ddec = self.forced_data['full_fit_ddec'][self.forced_objid_filt][self.rand_filt]
        self.filter = self.forced_data['filter'][self.forced_objid_filt][self.rand_filt]
        self.psfdepth = (self.forced_data['psfdepth'][self.forced_objid_filt])[self.rand_filt]

        self.dra_ivar = self.forced_data['full_fit_dra_ivar'][self.forced_objid_filt][self.rand_filt]
        self.ddec_ivar = self.forced_data['full_fit_ddec_ivar'][self.forced_objid_filt][self.rand_filt]
        self.dra_error = 1 / np.sqrt(self.dra_ivar)
        self.ddec_error = 1 / np.sqrt(self.ddec_ivar)
        if error_hyp:
            self.dra_error = np.hypot(self.dra_error, 0.005)
            self.ddec_error = np.hypot(self.ddec_error, 0.005)
            self.dra_ivar = 1 / self.dra_error ** 2 
            self.ddec_ivar = 1 / self.ddec_error ** 2 
                
        self._calculate_pm()
        
        self.filter_names = None

    def _generate_random_filter(self, num_rand, length):
        rand_filt = np.zeros(length)
        for i in range(num_rand):
            rand_num = randint(0, length)
            while rand_filt[rand_num] == 1:
                rand_num = randint(0, length)
            rand_filt[rand_num] = 1
        return rand_filt.astype(bool)

    def _calculate_pm(self):
        # plain slope fit
        ra_p, ra_V = np.polyfit(self.mjd, self.dra, 1, w=np.sqrt(self.dra_ivar), cov=True)
        self.dra_trend_func = np.poly1d(ra_p)
        self.pmra, self.pmra_cov = ra_p[0] * 365 * 1000, ra_V[0][0] * 365 * 1000

        dec_p, dec_V = np.polyfit(self.mjd, self.ddec, 1, w=np.sqrt(self.ddec_ivar), cov=True)
        self.ddec_trend_func = np.poly1d(dec_p)
        self.pmdec, self.pmdec_cov = dec_p[0] * 365 * 1000, dec_V[0][0] * 365 * 1000
        
        # optimise pm and plx simultaneuously
        daysperyear = 365.2425
        ref_mjd = 57174 # this is 2015.5 = ref_year
        _, _, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid] # TODO ref_year from forced?
        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, 0, 0, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 3))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec
        A[:N, 1] = dt
        A[N:, 2] = dt

        X, resid, rank, s = np.linalg.lstsq(A, np.append(self.dra, self.ddec), rcond=None)
        X = X * 1000
        self.parallax, self.pmra_plx, self.pmdec_plx = X[0], X[1], X[2]
        
        
        # optimise plx using lin fit pm        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, self.pmra, self.pmdec, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 1))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec

        X, resid, rank, s = np.linalg.lstsq(A, np.append(self.dra, self.ddec), rcond=None)
        X = X * 1000
        self.parallax_2 = X[0]
        
    def get_pmra_pmdec(self):
        print("proper motion all in mas/yr")
        print(f"pmra = { self.pmra } with covariance { self.pmra_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmra'][0] } from Gaia")
        print(f"pmdec = { self.pmdec } with covariance { self.pmdec_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmdec'][0] } from Gaia")
        
        print(f"FIT WITH PARALLAX: \n pmra = { self.pmra_plx }, pmdec = { self.pmdec_plx }, parallax = { self.parallax }, Gaia parallax = { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['parallax'][0] }")

    def plot_dra_and_ddec(self, alpha=0.1, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.mjd, self.dra, s=5, label='dra', alpha=alpha, c='c')
        ax.scatter(self.mjd, self.ddec, s=5, label='ddec', alpha=alpha, c='m')
        ax.plot(self.mjd, self.dra_trend_func(self.mjd), c='c')
        ax.plot(self.mjd, self.ddec_trend_func(self.mjd), c='m')
        ax.set_xlabel('mjd')
        ax.set_ylabel('dra and ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
        ax.set_title("Change in ra and dec through time")
        
    def plot_ddec_vs_dra(self, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.dra, self.ddec, c=self.mjd, s=3)
        ax.set_xlabel('dra')
        ax.set_ylabel('ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        
    def plot_ddec_vs_dra_filtered(self, xlim=None, ylim=None, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra[filter_filter], self.ddec[filter_filter], c=self.mjd[filter_filter], s=3)
            ax[row, col].set_xlabel('dra')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])
            if xlim:
                ax[row, col].set_xlim(xlim[0], xlim[1])
            
            col += 1
        fig.show()
        
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.mjd[filter_filter], self.ddec[filter_filter], s=3)
            ax[row, col].set_xlabel('mjd')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])           
            col += 1
        fig.show()
        
    def plot_psfdepth_vs_error(self, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].scatter(self.ddec_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].set_xlabel('dec error')
            ax[row, col].set_ylabel('psfdepth')
            ax[row, col].set_title(f"{ f } filter")
            ax[row, col].set_xscale('log')
            
            col += 1
        
    def plot_with_real_gaia(self, dra_lim=None, ddec_lim=None, figheight=15, figwidth=15):
        obj_ra, obj_dec, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid]
        pmra, pmdec, parallax = self.real_tractor_data['pmra'][self.real_objid], self.real_tractor_data['pmdec'][self.real_objid], self.real_tractor_data['parallax'][self.real_objid]

        proj_ra = []
        proj_dec = []
        for date in self.mjd:
            ra, dec = radec_at_mjd(obj_ra, obj_dec, ref_year, pmra, pmdec, parallax, date)
            proj_dec.append(dec)
            proj_ra.append(ra)
        proj_dra = (np.array(proj_ra) - obj_ra) * 3600
        proj_ddec = (np.array(proj_dec) - obj_dec) * 3600
        
        fig, ax = plt.subplots(3)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0].scatter(self.dra, self.ddec, c=self.mjd, s=3, label="Measured")
        ax[0].scatter(proj_dra, proj_ddec, s=3, c='r', label="Projected (Gaia)")
        ax[0].set_xlabel('dra')
        ax[0].set_ylabel('ddec')
        if dra_lim: ax[0].set_xlim(dra_lim[0], dra_lim[1])
        if ddec_lim: ax[0].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[0].set_title("Projected (Gaia) and measured change in location")
        
        ax[1].errorbar(self.mjd, self.dra, fmt='o', alpha=0.1, yerr=self.dra_error, label="Measured")
        ax[1].scatter(self.mjd, proj_dra, s=3, c='r', label="Projected (Gaia)")
        if dra_lim: ax[1].set_ylim(dra_lim[0], dra_lim[1])
        ax[1].set_xlabel('mjd')
        ax[1].set_ylabel('dra')
        ax[1].set_title("Projected (Gaia) and measured change in ra, error bars on y-axis")
        
        ax[2].errorbar(self.mjd, self.ddec, fmt='o', alpha=0.1, yerr=self.ddec_error, label="Measured")
        ax[2].scatter(self.mjd, proj_ddec, s=3, c='r', label="Projeced (Gaia)")
        if ddec_lim: ax[2].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[2].set_xlabel('mjd')
        ax[2].set_ylabel('ddec')
        ax[2].set_title("Projected (Gaia) and measured change in dec, error bars on y-axis")

    def plot_dra_ddec_vs_mjd_coloured(self, dra_lim=None, ddec_lim=None, figheight=30, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        fig, ax = plt.subplots(4, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0, 0].scatter(self.mjd, self.dra, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 0].set_xlabel('mjd')
        ax[0, 0].set_ylabel('dra')
        if dra_lim: ax[0, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[0, 0].set_title("Colour = airmass")
        
        ax[0, 1].scatter(self.mjd, self.ddec, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 1].set_xlabel('mjd')
        ax[0, 1].set_ylabel('ddec')
        if ddec_lim: ax[0, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        filters = np.zeros(self.filter.shape)
        for i, f in enumerate(self.filter_names):
            filters[self.filter == f] =int(i)
        
        ax[1, 0].scatter(self.mjd, self.dra, c=filters, s=3, alpha=0.5)
        ax[1, 0].set_xlabel('mjd')
        ax[1, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[1, 0].set_title("Colour = filters")
        
        ax[1, 1].scatter(self.mjd, self.ddec, c=filters, s=3, alpha=0.5)
        ax[1, 1].set_xlabel('mjd')
        ax[1, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[2, 0].scatter(self.mjd, self.dra, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 0].set_xlabel('mjd')
        ax[2, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[2, 0].set_title("Colour = flux")
        
        ax[2, 1].scatter(self.mjd, self.ddec, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 1].set_xlabel('mjd')
        ax[2, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[3, 0].scatter(self.mjd, self.dra, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 0].set_xlabel('mjd')
        ax[3, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[3, 0].set_title("Colour = exptime")
        
        ax[3, 1].scatter(self.mjd, self.ddec, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 1].set_xlabel('mjd')
        ax[3, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])


class CorrObjectPM: # CORRECTED
    """
    A class used to calculate an object's proper motion and visualise it, 
    including a visual comparison to Gaia data
    
    
    ===== Attributes =====
    tractor_data: astropy FITS_rec
        fake tractor data (fake = regenerated without proper motions)
    forced_data: astropy FITS_rec
        fake forced photometry data
    real_tractor_data: astropy FITS_rec
        real Gaia tractor data
        
    ra: float
        right ascension of object
    dec: float
        declination of object
    objid: int
        object ID in tractor and forced datasets (not in gaia!)
    real_objid: int
        object ID in real (gaia) tractor dataset 
    forced_objid_filt: bool array
        filter array according to objid 
        
    mjd: int array
        array of modified Julian dates object was observed on
    dra: float array
        change in ra of object on corresponding mjd
    dra_ivar: float array
        ivar of dra 
    ddec: float array
        change in dec of object on corresponding mjd
    ddec_ivar: float array
        ivar of ddec
    dra_error: float array
        error of dra
    ddec_error: float array
        error of ddec
        
    filter: str array
        array of exposures filter (g, i, r or z)
    filter_names: str array
        array of possible filter names (this would only be ~4 elements, 
        extracted in case filters are added)
    psfdepth: float array
        psfdepth of exposures
        
    dra_trend_func: numpy poly1d
        linear fit of object dra
    pmra: float
        proper motion in ra calculated from linear fit
    ddec_trend_func: numpy poly1d
        linear trend of object ddec
    pmdec: float
        proper motion in dec calculated from linear fit
    
    
    ===== Methods =====
    _generate_random_filter(num_rand, length)
        generates a filter of length <length> with <num_rand> Trues at random 
        indices in the array
    get_pmra_pmdec()
        prints pmra and pmdec of object with their covariances and the Gaia-measured
    plot_dra_and_ddec(alpha=0.1, ylim=None)
        plots dra and ddec through time (mjd) with their linear trend lines
    plot_ddec_vs_dra(xlim=None, ylim=None)
        plots ddec vs dra of object
    plot_ddec_vs_dra_filtered(xlim=None, ylim=None, figheight=15, figwidth=15)
        plots ddec vs dra from different esposure filters, one plot per filter
    plot_psfdepth_vs_error(figheight=15, figwidth=15)
        plots pdfdepth vs error of pm from different esposure filters, one plot per filter
    plot_with_real_gaia(dra_lim=None, ddec_lim=None, figheight=15, figwidth=15)
        plots dra and ddec, and gaia dra and ddec through time (mjd), one plot for each of
        dra and ddec
    """
    
    def __init__(self, ra, dec, tractor_data, forced_data, real_tractor_data, error_hyp=True, num_rand=None, brickid=None): # purposely not passing in objid bc seems different in real vs fake data
        self.ra = ra
        self.dec = dec
        
        if isinstance(tractor_data, str):
            self.tractor_data = fits.open(tractor_data)[1].data
            self.forced_data = fits.open(forced_data)[1].data
            self.real_tractor_data = fits.open(real_tractor_data)[1].data
        else:
            self.tractor_data = tractor_data
            self.forced_data = forced_data
            self.real_tractor_data = real_tractor_data
        # else: 
        #     self.tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        #     self.forced_data = fits.open(f'../../../cfs/cdirs/cosmo/work/users/dstn/ls-motions/fake-gaia-reductions-2/forced/forced-brickwise-{ brickid }.fits')[1].data
        #     self.real_tractor_data = fits.open(f'../../../cfs/cdirs/cosmo/data/legacysurvey/dr9/south/tractor/{ brickid[:3] }/tractor-{ brickid }.fits')[1].data
        
        self.objid = np.argmin(np.hypot(self.ra - self.tractor_data['ra'], self.dec - self.tractor_data['dec']))
        self.real_objid = np.argmin(np.hypot(self.ra - self.real_tractor_data['ra'], self.dec - self.real_tractor_data['dec']))
        self.forced_objid_filt = np.logical_and(np.logical_and((self.forced_data['objid'] == self.objid), self.forced_data['full_fit_dra_ivar'] != 0), self.forced_data['full_fit_ddec_ivar'] != 0)
        if num_rand: # TODO if num_rand > size
            self.rand_filt = self._generate_random_filter(num_rand, sum(self.forced_objid_filt))
        else:
            self.rand_filt = np.ones(sum(self.forced_objid_filt), dtype=bool)  

        self.mjd = self.forced_data['mjd'][self.forced_objid_filt][self.rand_filt]
        self.dra = self.forced_data['dcr_full_fit_dra'][self.forced_objid_filt][self.rand_filt]
        # self.dra[self.dra == 0] = self.forced_data['full_fit_dra'][self.forced_objid_filt][self.rand_filt][self.dra == 0]
        self.ddec = self.forced_data['dcr_full_fit_ddec'][self.forced_objid_filt][self.rand_filt]
        # self.ddec[self.ddec == 0] = self.forced_data['full_fit_ddec'][self.forced_objid_filt][self.rand_filt][self.ddec == 0]
        
        self.corr_ffx = self.forced_data['dcr_full_fit_x'][self.forced_objid_filt][self.rand_filt]
        # self.corr_ffx[self.corr_ffx == 0] = self.forced_data['full_fit_x'][self.forced_objid_filt][self.rand_filt][self.corr_ffx == 0]
        self.corr_ffy = self.forced_data['dcr_full_fit_y'][self.forced_objid_filt][self.rand_filt]
        # self.corr_ffy[self.corr_ffy == 0] = self.forced_data['full_fit_y'][self.forced_objid_filt][self.rand_filt][self.corr_ffy == 0]
        self.ffx = self.forced_data['full_fit_x'][self.forced_objid_filt][self.rand_filt]
        self.ffy = self.forced_data['full_fit_y'][self.forced_objid_filt][self.rand_filt]
        self.x = self.forced_data['x'][self.forced_objid_filt][self.rand_filt]
        self.y = self.forced_data['y'][self.forced_objid_filt][self.rand_filt]
        
        self.filter = self.forced_data['filter'][self.forced_objid_filt][self.rand_filt]
        self.psfdepth = (self.forced_data['psfdepth'][self.forced_objid_filt])[self.rand_filt]

        self.dra_ivar = self.forced_data['full_fit_dra_ivar'][self.forced_objid_filt][self.rand_filt]
        self.ddec_ivar = self.forced_data['full_fit_ddec_ivar'][self.forced_objid_filt][self.rand_filt]
        self.dra_error = 1 / np.sqrt(self.dra_ivar)
        self.ddec_error = 1 / np.sqrt(self.ddec_ivar)
        if error_hyp:
            self.dra_error = np.hypot(self.dra_error, 0.005)
            self.ddec_error = np.hypot(self.ddec_error, 0.005)
            self.dra_ivar = 1 / self.dra_error ** 2 
            self.ddec_ivar = 1 / self.ddec_error ** 2 
                
        self._calculate_pm()
        
        self.filter_names = None
        self.mag = self.tractor_data[self.tractor_data['objid'] == self.objid]['gaia_phot_g_mean_mag'][0]

    def _generate_random_filter(self, num_rand, length):
        rand_filt = np.zeros(length)
        for i in range(num_rand):
            rand_num = randint(0, length)
            while rand_filt[rand_num] == 1:
                rand_num = randint(0, length)
            rand_filt[rand_num] = 1
        return rand_filt.astype(bool)

    def _calculate_pm(self):
        # plain slope fit
        ra_p, ra_V = np.polyfit(self.mjd, self.dra, 1, w=np.sqrt(self.dra_ivar), cov=True)
        self.dra_trend_func = np.poly1d(ra_p)
        self.pmra, self.pmra_cov = ra_p[0] * 365 * 1000, ra_V[0][0] * 365 * 1000

        dec_p, dec_V = np.polyfit(self.mjd, self.ddec, 1, w=np.sqrt(self.ddec_ivar), cov=True)
        self.ddec_trend_func = np.poly1d(dec_p)
        self.pmdec, self.pmdec_cov = dec_p[0] * 365 * 1000, dec_V[0][0] * 365 * 1000
        
        # optimise pm and plx simultaneuously
        daysperyear = 365.2425
        ref_mjd = 57174 # this is 2015.5 = ref_year
        _, _, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid] # TODO ref_year from forced?
        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, 0, 0, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 3))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec
        A[:N, 1] = dt
        A[N:, 2] = dt

        X, resid, rank, s = np.linalg.lstsq(A, np.append(self.dra, self.ddec), rcond=None)
        X = X * 1000
        self.parallax, self.pmra_plx, self.pmdec_plx = X[0], X[1], X[2]
        
        
        # optimise plx using lin fit pm        
        plx_ra, plx_dec = [], []
        for date in self.mjd:
            ra, dec = radec_at_mjd(self.ra, self.dec, ref_year, self.pmra, self.pmdec, 1000, date) # mas/ye and mas, returns deg
            plx_ra.append(ra)
            plx_dec.append(dec)
        plx_dra, plx_ddec = (np.array(plx_ra) - self.ra) * np.cos(np.deg2rad(self.dec)) * 3600, (np.array(plx_dec) - self.dec) * 3600 # change in ra, dec due to parallax = 1 arcsec

        dt = (self.mjd - ref_mjd) / daysperyear
        N = len(self.dra)

        A = np.zeros((2 * N, 1))
        A[:N, 0] = plx_ra
        A[N:, 0] = plx_dec

        X, resid, rank, s = np.linalg.lstsq(A, np.append(self.dra, self.ddec), rcond=None)
        X = X * 1000
        self.parallax_2 = X[0]
        
    def get_pmra_pmdec(self):
        print("proper motion all in mas/yr")
        print(f"pmra = { self.pmra } with covariance { self.pmra_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmra'][0] } from Gaia")
        print(f"pmdec = { self.pmdec } with covariance { self.pmdec_cov }, compared to { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['pmdec'][0] } from Gaia")
        
        print(f"FIT WITH PARALLAX: \n pmra = { self.pmra_plx }, pmdec = { self.pmdec_plx }, parallax = { self.parallax }, Gaia parallax = { self.real_tractor_data[self.real_tractor_data['objid'] == self.real_objid]['parallax'][0] }")

    def plot_dra_and_ddec(self, alpha=0.1, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.mjd, self.dra, s=5, label='dra', alpha=alpha, c='c')
        ax.scatter(self.mjd, self.ddec, s=5, label='ddec', alpha=alpha, c='m')
        ax.plot(self.mjd, self.dra_trend_func(self.mjd), c='c')
        ax.plot(self.mjd, self.ddec_trend_func(self.mjd), c='m')
        ax.set_xlabel('mjd')
        ax.set_ylabel('dra and ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        ax.legend()
        ax.set_title("Change in ra and dec through time")
        
    def plot_ddec_vs_dra(self, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.scatter(self.dra, self.ddec, c=self.mjd, s=3)
        ax.set_xlabel('dra')
        ax.set_ylabel('ddec')
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        
    def plot_ddec_vs_dra_filtered(self, xlim=None, ylim=None, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra[filter_filter], self.ddec[filter_filter], c=self.mjd[filter_filter], s=3)
            ax[row, col].set_xlabel('dra')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])
            if xlim:
                ax[row, col].set_xlim(xlim[0], xlim[1])
            
            col += 1
        fig.show()
        
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.mjd[filter_filter], self.ddec[filter_filter], s=3)
            ax[row, col].set_xlabel('mjd')
            ax[row, col].set_ylabel('ddec')
            ax[row, col].set_title(f"{ f } filter")
            if ylim:
                ax[row, col].set_ylim(ylim[0], ylim[1])           
            col += 1
        fig.show()
        
    def plot_psfdepth_vs_error(self, figheight=15, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        numrows = math.ceil(len(self.filter_names) / 2)
        fig, ax = plt.subplots(numrows, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        row = 0
        col = 0
        for f in self.filter_names:
            if col == 2:
                col = 0
                row += 1
            filter_filter = (self.filter == f)
            ax[row, col].scatter(self.dra_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].scatter(self.ddec_error[filter_filter], -2.5 * (np.log10(5. / np.sqrt(self.psfdepth[filter_filter])) - 9), s=5)
            ax[row, col].set_xlabel('dec error')
            ax[row, col].set_ylabel('psfdepth')
            ax[row, col].set_title(f"{ f } filter")
            ax[row, col].set_xscale('log')
            
            col += 1
        
    def plot_with_real_gaia(self, dra_lim=None, ddec_lim=None, figheight=15, figwidth=15):
        obj_ra, obj_dec, ref_year = self.real_tractor_data['ra'][self.real_objid], self.real_tractor_data['dec'][self.real_objid], self.real_tractor_data['ref_epoch'][self.real_objid]
        pmra, pmdec, parallax = self.real_tractor_data['pmra'][self.real_objid], self.real_tractor_data['pmdec'][self.real_objid], self.real_tractor_data['parallax'][self.real_objid]

        proj_ra = []
        proj_dec = []
        for date in self.mjd:
            ra, dec = radec_at_mjd(obj_ra, obj_dec, ref_year, pmra, pmdec, parallax, date)
            proj_dec.append(dec)
            proj_ra.append(ra)
        proj_dra = (np.array(proj_ra) - obj_ra) * 3600
        proj_ddec = (np.array(proj_dec) - obj_dec) * 3600
        
        fig, ax = plt.subplots(3)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0].scatter(self.dra, self.ddec, c=self.mjd, s=3, label="Measured")
        ax[0].scatter(proj_dra, proj_ddec, s=3, c='r', label="Projected (Gaia)")
        ax[0].set_xlabel('dra')
        ax[0].set_ylabel('ddec')
        if dra_lim: ax[0].set_xlim(dra_lim[0], dra_lim[1])
        if ddec_lim: ax[0].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[0].set_title("Projected (Gaia) and measured change in location")
        
        ax[1].errorbar(self.mjd, self.dra, fmt='o', alpha=0.1, yerr=self.dra_error, label="Measured")
        ax[1].scatter(self.mjd, proj_dra, s=3, c='r', label="Projected (Gaia)")
        if dra_lim: ax[1].set_ylim(dra_lim[0], dra_lim[1])
        ax[1].set_xlabel('mjd')
        ax[1].set_ylabel('dra')
        ax[1].set_title("Projected (Gaia) and measured change in ra, error bars on y-axis")
        
        ax[2].errorbar(self.mjd, self.ddec, fmt='o', alpha=0.1, yerr=self.ddec_error, label="Measured")
        ax[2].scatter(self.mjd, proj_ddec, s=3, c='r', label="Projeced (Gaia)")
        if ddec_lim: ax[2].set_ylim(ddec_lim[0], ddec_lim[1])
        ax[2].set_xlabel('mjd')
        ax[2].set_ylabel('ddec')
        ax[2].set_title("Projected (Gaia) and measured change in dec, error bars on y-axis")

    def plot_dra_ddec_vs_mjd_coloured(self, dra_lim=None, ddec_lim=None, figheight=30, figwidth=15):
        if not self.filter_names: self.filter_names = Counter(self.filter).keys()
        fig, ax = plt.subplots(4, 2)
        fig.set_figheight(figheight)
        fig.set_figwidth(figwidth)
        
        ax[0, 0].scatter(self.mjd, self.dra, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 0].set_xlabel('mjd')
        ax[0, 0].set_ylabel('dra')
        if dra_lim: ax[0, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[0, 0].set_title("Colour = airmass")
        
        ax[0, 1].scatter(self.mjd, self.ddec, c=self.forced_data['airmass'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[0, 1].set_xlabel('mjd')
        ax[0, 1].set_ylabel('ddec')
        if ddec_lim: ax[0, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        filters = np.zeros(self.filter.shape)
        for i, f in enumerate(self.filter_names):
            filters[self.filter == f] =int(i)
        
        ax[1, 0].scatter(self.mjd, self.dra, c=filters, s=3, alpha=0.5)
        ax[1, 0].set_xlabel('mjd')
        ax[1, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[1, 0].set_title("Colour = filters")
        
        ax[1, 1].scatter(self.mjd, self.ddec, c=filters, s=3, alpha=0.5)
        ax[1, 1].set_xlabel('mjd')
        ax[1, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[2, 0].scatter(self.mjd, self.dra, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 0].set_xlabel('mjd')
        ax[2, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[2, 0].set_title("Colour = flux")
        
        ax[2, 1].scatter(self.mjd, self.ddec, c=self.forced_data['flux'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[2, 1].set_xlabel('mjd')
        ax[2, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
        
        ax[3, 0].scatter(self.mjd, self.dra, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 0].set_xlabel('mjd')
        ax[3, 0].set_ylabel('dra')
        if dra_lim: ax[1, 0].set_ylim(dra_lim[0], dra_lim[1])
        ax[3, 0].set_title("Colour = exptime")
        
        ax[3, 1].scatter(self.mjd, self.ddec, c=self.forced_data['exptime'][self.forced_objid_filt][self.rand_filt], s=3, alpha=0.5)
        ax[3, 1].set_xlabel('mjd')
        ax[3, 1].set_ylabel('ddec')
        if ddec_lim: ax[1, 1].set_ylim(ddec_lim[0], ddec_lim[1])
      