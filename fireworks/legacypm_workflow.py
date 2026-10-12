"""Add the per-brick pipeline (DCR -> ring maps -> lateral colour -> PM catalogue) to the FireWorks launchpad.
One workflow per RA degree slice; 4 fireworks per brick, chained. See RUNBOOK.md for the full procedure.

  python legacypm_workflow.py 060 061                   # new bricks from Dustin's forced photometry
  python legacypm_workflow.py 060 --from-archive        # rerun RM/LM/PM on the archived bricks with current code
  python legacypm_workflow.py 030 --input-root /global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick
  add --dry-run to list what would be added without touching the launchpad

Bricks whose tractor-pm-<brick>.fits already exists (non-empty) in the output dir are skipped.
"""
import os
import argparse
from fireworks import Firework, Workflow, LaunchPad, ScriptTask

# absolute path, so the jobs run the code in the clone you add the workflow from
PY_FILES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "py_files"))

DEFAULT_INPUT_ROOT = "/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick"
DEFAULT_ARCHIVE_ROOT = "/pscratch/sd/n/nelfalou/archive/legacypm-catalogue"
DEFAULT_OUT_ROOT = "/pscratch/sd/n/nelfalou/legacypm-catalogue"


def get_launchpad():
    # credentials pulled from env instead of hardcoded; `source ~/.legacypm_env.sh` sets:
    #   LEGACYPM_MONGO_HOST, LEGACYPM_MONGO_USER, LEGACYPM_MONGO_PASSWORD
    return LaunchPad(
        host=os.environ['LEGACYPM_MONGO_HOST'],
        username=os.environ['LEGACYPM_MONGO_USER'],
        name='legacypm',
        password=os.environ['LEGACYPM_MONGO_PASSWORD'],
    )


def create_firework(script_file, parameters, fw_name, parents):
    task = ScriptTask(script=f"python {os.path.join(PY_FILES, script_file)} " + parameters)
    return Firework(task, name=fw_name, parents=parents)


def nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def find_bricks(src_dir, new_dir, from_archive):
    bricks = []
    for fn in sorted(os.listdir(src_dir)):
        if not (fn.startswith("forced-") and fn.endswith(".fits")) or not nonempty(os.path.join(src_dir, fn)):
            continue
        brick = fn[-13:-5]
        needed = f"tractor-pm-{brick}.fits" if from_archive else f"tractor-forced-{brick}.fits"
        if not nonempty(os.path.join(src_dir, needed)):
            continue
        if nonempty(os.path.join(new_dir, f"tractor-pm-{brick}.fits")):
            continue  # already finished
        bricks.append(brick)
    return bricks


def build_workflow(degree, src_dir, new_dir, from_archive):
    bricks = find_bricks(src_dir, new_dir, from_archive)
    # in archive mode the prep step writes tractor-forced-<brick>.fits into new_dir
    tractor_dir = new_dir if from_archive else src_dir
    fireworks = []
    for brick in bricks:
        forced_fname = f"forced-{brick}.fits"
        if from_archive:
            first_fw = create_firework("prep_archive_rerun.py", f"{forced_fname} {new_dir} {src_dir}",
                                       f"Prep_archive_{brick}", None)
        else:
            first_fw = create_firework("brick_DCR.py", f"{forced_fname} {new_dir} {src_dir}",
                                       f"DCR_correction_{brick}", None)
        ringmaps_fw = create_firework("brick_ringmaps.py", f"{forced_fname} {new_dir} {new_dir}",
                                      f"Ringmaps_correction_{brick}", [first_fw])
        lateralmaps_fw = create_firework("brick_lateralmaps.py", f"{forced_fname} {new_dir} {new_dir} {tractor_dir}",
                                         f"Lateralmaps_correction_{brick}", [ringmaps_fw])
        catalog_fw = create_firework("create_brick_catalogue.py", f"{forced_fname} {new_dir} {new_dir} {tractor_dir}",
                                     f"Catalogue_creation_{brick}", [lateralmaps_fw])
        fireworks.extend([first_fw, ringmaps_fw, lateralmaps_fw, catalog_fw])
    name = f"degree {degree}{' (archive rerun)' if from_archive else ''}"
    return bricks, (Workflow(fireworks, name=name) if fireworks else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("degrees", nargs="+", help="RA degree slices, e.g. 060 061")
    ap.add_argument("--from-archive", action="store_true",
                    help=f"rerun RM/LM/PM on archived bricks (from {DEFAULT_ARCHIVE_ROOT}/<deg>/)")
    ap.add_argument("--input-root", default=None,
                    help=f"forced-photometry root (default {DEFAULT_INPUT_ROOT}; archive root with --from-archive)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help=f"output root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--dry-run", action="store_true", help="list bricks, don't add anything to the launchpad")
    args = ap.parse_args()

    input_root = args.input_root or (DEFAULT_ARCHIVE_ROOT if args.from_archive else DEFAULT_INPUT_ROOT)
    launchpad = None if args.dry_run else get_launchpad()
    os.makedirs(os.path.join(args.out_root, "fw_logs"), exist_ok=True)  # my_qadapter.yaml logdir
    for degree in args.degrees:
        src_dir = os.path.join(input_root, degree) + "/"
        new_dir = os.path.join(args.out_root, degree) + "/"
        os.makedirs(new_dir, exist_ok=True)
        bricks, workflow = build_workflow(degree, src_dir, new_dir, args.from_archive)
        print(f"degree {degree}: {len(bricks)} bricks to run ({src_dir} -> {new_dir})")
        if workflow is None:
            continue
        if args.dry_run:
            print("  first few:", bricks[:5])
            continue
        launchpad.add_wf(workflow)
        print(f"  added {len(workflow.fws)} fireworks to the launchpad")
    print("Submit them with: qlaunch -r rapidfire -m <max_jobs> --nlaunches infinite   (from the fireworks/ dir)")
