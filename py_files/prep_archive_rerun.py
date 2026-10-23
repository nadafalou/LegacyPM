"""Prepare an archived brick for rerunning ALL steps (DCR -> RM -> LM -> PM) with the current code.

The archived forced-<brick>.fits holds the raw forced-photometry columns plus old dcr_/rm_/lm_ columns, and the archived
tractor-pm-<brick>.fits is the tractor-forced catalogue plus old PM columns. The original tractor-forced/forced
inputs for these bricks were purged, so we rebuild them from the archive by dropping the derived columns:
  <new_dir>/forced-<brick>.fits          (dcr_*/rm_*/lm_*, dp1, colour_airmass, *_ivar_total dropped; DCR is redone,
                                          since the DCR model changed on 2026-10-23)
  <new_dir>/tractor-forced-<brick>.fits  (PM columns dropped)
Usage: python prep_archive_rerun.py <filename> <new_dir> <archive_dir>
"""
import os
import sys
import time
import logging
from astropy.io import fits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PM_COLS = {'pmra_plx', 'pmra_plx_ivar', 'pmdec_plx', 'pmdec_plx_ivar', 'parallax_new', 'parallax_new_ivar',
           'new_ra_plx', 'new_dec_plx', 'new_ra_plx_ivar', 'new_dec_plx_ivar', 'pmra_no_plx', 'pmra_no_plx_ivar',
           'pmdec_no_plx', 'pmdec_no_plx_ivar', 'new_ra_no_plx', 'new_dec_no_plx', 'new_ra_no_plx_ivar',
           'new_dec_no_plx_ivar', 'pm_flag', 'ref_mjd',
           'pm_quality', 'pm_n_epochs', 'pm_baseline_days', 'pm_n_bad_epochs', 'pm_colour_source'}


def strip_columns(src, dst, drop, header=()):
    with fits.open(src) as h:
        keep = [c for c in h[1].columns if c.name not in drop]
        hdu = fits.BinTableHDU.from_columns(fits.ColDefs(keep), header=h[1].header)
        for key, val, comment in header:
            hdu.header[key] = (val, comment)
        tmp = dst + ".tmp"
        fits.HDUList([h[0].copy(), hdu]).writeto(tmp, overwrite=True)
        os.replace(tmp, dst)  # atomic


def prep_brick(filename, new_dir, archive_dir):
    brick = filename[-13:-5]
    forced = os.path.join(archive_dir, filename)
    with fits.open(forced) as h:
        names = h[1].columns.names
    if 'full_fit_x' not in names:
        raise ValueError(f"{forced} has no raw full_fit columns")
    strip_columns(forced, os.path.join(new_dir, filename),
                  {c for c in names if c.startswith(('dcr_', 'rm_', 'lm_')) or c.endswith('_ivar_total')
                   or c in ('dp1', 'colour_airmass')},
                  # archive copies (2025) of pre-fix inputs: date for the Dec-ivar bug correction (error_model.py)
                  header=[('INPDATE', time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(forced))),
                           'date the forced-photometry input was written')])
    strip_columns(os.path.join(archive_dir, f"tractor-pm-{brick}.fits"),
                  os.path.join(new_dir, f"tractor-forced-{brick}.fits"), PM_COLS)
    logger.info(f"Prepared {brick} from {archive_dir} into {new_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: <filename> <new_dir> <archive_dir>")
        sys.exit(1)
    prep_brick(*sys.argv[1:])
