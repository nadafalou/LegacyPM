# Applications of the catalogue (pilots)

Kept apart from the pipeline and from `run/`. Only reads the v2 catalogues
(`/pscratch/sd/n/nelfalou/legacypm-catalogue-v2/`), never writes there.

## A. Reduced proper motion diagram + white-dwarf candidates

**Full write-up: `RPM_PILOT.md`** (motivation, method, results, vetting, caveats, how to reproduce). Numbers there are
from the catalogue after the error-model v3 redo (2026-10-27).

| file | what |
|---|---|
| `rpm-diagram.ipynb`, `rpm.py` | the RPM diagram on 200 bricks at RA 25-35, Dec -40 to -30; faint-end false-positive and PM-direction checks |
| `rpm-diagram-fieldB.ipynb` | same notebook run with `RPM_FIELD=B` on a held-out field (RA 5-15, Dec -20 to -10; outputs `*_fieldB*`); see RPM_PILOT.md §3.4 |
| `wd_vet_extract.py` | per-epoch data for the white-dwarf candidates + calibration/control samples (compute node, 20 s) |
| `wd-vetting.ipynb`, `wd_vet.py` | vetting: astrometric tests, stacked parallax, subdwarf velocities, colours, cutouts, grades |
| `data/bergeron_Table_Mass_0.6.txt` | Montreal WD cooling models, 0.6 Msun (downloaded 2026-10-27) |
| `plots/` | `rpm_*` (diagram, PM directions, r histogram), `wdvet_*` (vetting figures, cutout montage) |
| `output/` | `hpm_mu20_n200.fits` (HPM sample), `wd_vet_*.fits` (extraction), `wd_candidates_vetted.{fits,csv}` (grades) |

Headline (200 bricks, 12.5 deg²): 7,438 HPM objects at r < 23, 1,897 not in Gaia; 157 non-Gaia white-dwarf candidates,
of which 89 grade A after vetting. Catalogue issues found: faint-end PM errors underestimated (x1.2-2.2 at r > 23, also
with v3); Gaia-star parallax pull width 1.22 with v3 (0.96 before).

Files named `*_n20*` in `plots/` and `output/` are from the first 20-brick pilot (old error model).
