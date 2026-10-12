"""Run the per-brick pipeline for whole RA degree slices inside one Slurm job (no FireWorks / MongoDB needed).
Same steps and commands as fireworks/legacypm_workflow.py, run as subprocesses in a pool of workers.
Restartable: bricks whose tractor-pm-<brick>.fits already exists (non-empty) are skipped.

  python run_bricks.py 060 061 --workers 64                 # new bricks from Dustin's forced photometry
  python run_bricks.py 060 --from-archive                   # rerun RM/LM/PM on the archived bricks
  python run_bricks.py 060 --dry-run                        # just count
Per-brick logs: <out-root>/<deg>/logs/<brick>.log; failures listed in <out-root>/<deg>/failed.txt
"""
import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

PY_FILES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "py_files"))
DEFAULT_INPUT_ROOT = "/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick"
DEFAULT_ARCHIVE_ROOT = "/pscratch/sd/n/nelfalou/archive/legacypm-catalogue"
DEFAULT_OUT_ROOT = "/pscratch/sd/n/nelfalou/legacypm-catalogue"


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


def brick_commands(brick, src_dir, new_dir, from_archive):
    fn = f"forced-{brick}.fits"
    tractor_dir = new_dir if from_archive else src_dir
    first = (["prep_archive_rerun.py", fn, new_dir, src_dir] if from_archive
             else ["brick_DCR.py", fn, new_dir, src_dir])
    return [first,
            ["brick_ringmaps.py", fn, new_dir, new_dir],
            ["brick_lateralmaps.py", fn, new_dir, new_dir, tractor_dir],
            ["create_brick_catalogue.py", fn, new_dir, new_dir, tractor_dir]]


def run_brick(brick, src_dir, new_dir, from_archive):
    t0 = time.time()
    with open(os.path.join(new_dir, "logs", f"{brick}.log"), "w") as log:
        for cmd in brick_commands(brick, src_dir, new_dir, from_archive):
            log.write(f"### {' '.join(cmd)}\n")
            log.flush()
            r = subprocess.run([sys.executable, os.path.join(PY_FILES, cmd[0])] + cmd[1:],
                               stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return brick, f"{cmd[0]} exit {r.returncode}", time.time() - t0
    return brick, None, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("degrees", nargs="+", help="RA degree slices, e.g. 060 061")
    ap.add_argument("--from-archive", action="store_true")
    ap.add_argument("--input-root", default=None)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_root = args.input_root or (DEFAULT_ARCHIVE_ROOT if args.from_archive else DEFAULT_INPUT_ROOT)
    jobs = []
    for degree in args.degrees:
        src_dir = os.path.join(input_root, degree) + "/"
        new_dir = os.path.join(args.out_root, degree) + "/"
        os.makedirs(os.path.join(new_dir, "logs"), exist_ok=True)
        bricks = find_bricks(src_dir, new_dir, args.from_archive)
        print(f"degree {degree}: {len(bricks)} bricks to run ({src_dir} -> {new_dir})", flush=True)
        jobs += [(b, src_dir, new_dir) for b in bricks]
    if args.dry_run or not jobs:
        sys.exit(0)

    n_ok, failed = 0, []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_brick, b, s, n, args.from_archive) for b, s, n in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            brick, err, dt = fut.result()
            if err:
                failed.append((brick, err))
                print(f"[{i}/{len(jobs)}] FAILED {brick}: {err}", flush=True)
            else:
                n_ok += 1
                print(f"[{i}/{len(jobs)}] done {brick} ({dt:.0f} s)", flush=True)
    for degree in args.degrees:  # bricks are named after their RA degree, e.g. 0600m050 -> 060
        with open(os.path.join(args.out_root, degree, "failed.txt"), "w") as f:
            f.writelines(f"{b} {e}\n" for b, e in failed if b[:3] == degree)
    print(f"finished: {n_ok} ok, {len(failed)} failed", flush=True)
