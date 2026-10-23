"""Prepare an archived brick for rerunning the RM -> LM -> PM steps with the current code.

The archived forced-<brick>.fits already holds the DCR columns but also old rm_/lm_ columns, and the archived
tractor-pm-<brick>.fits is the tractor-forced catalogue plus old PM columns. The original tractor-forced/forced
inputs for these bricks were purged, so we rebuild them from the archive by dropping the derived columns:
  <new_dir>/forced-<brick>.fits          (DCR columns kept, rm_*/lm_* dropped)
  <new_dir>/tractor-forced-<brick>.fits  (PM columns dropped)
Usage: python prep_archive_rerun.py <filename> <new_dir> <archive_dir>
"""
import os
import sys
import logging
from astropy.io import fits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PM_COLS = {'pmra_plx', 'pmra_plx_ivar', 'pmdec_plx', 'pmdec_plx_ivar', 'parallax_new', 'parallax_new_ivar',
           'new_ra_plx', 'new_dec_plx', 'new_ra_plx_ivar', 'new_dec_plx_ivar', 'pmra_no_plx', 'pmra_no_plx_ivar',
           'pmdec_no_plx', 'pmdec_no_plx_ivar', 'new_ra_no_plx', 'new_dec_no_plx', 'new_ra_no_plx_ivar',
           'new_dec_no_plx_ivar', 'pm_flag', 'ref_mjd',
           'pm_quality', 'pm_n_epochs', 'pm_baseline_days'}


def strip_columns(src, dst, drop):
    with fits.open(src) as h:
        keep = [c for c in h[1].columns if c.name not in drop]
        hdu = fits.BinTableHDU.from_columns(fits.ColDefs(keep), header=h[1].header)
        tmp = dst + ".tmp"
        fits.HDUList([h[0].copy(), hdu]).writeto(tmp, overwrite=True)
        os.replace(tmp, dst)  # atomic


def prep_brick(filename, new_dir, archive_dir):
    brick = filename[-13:-5]
    forced = os.path.join(archive_dir, filename)
    with fits.open(forced) as h:
        names = h[1].columns.names
    if 'dcr_full_fit_x' not in names:
        raise ValueError(f"{forced} has no DCR columns -- run brick_DCR.py on the original input instead")
    strip_columns(forced, os.path.join(new_dir, filename), {c for c in names if c.startswith(('rm_', 'lm_')) or c.endswith('_ivar_total')})
    strip_columns(os.path.join(archive_dir, f"tractor-pm-{brick}.fits"),
                  os.path.join(new_dir, f"tractor-forced-{brick}.fits"), PM_COLS)
    logger.info(f"Prepared {brick} from {archive_dir} into {new_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: <filename> <new_dir> <archive_dir>")
        sys.exit(1)
    prep_brick(*sys.argv[1:])
