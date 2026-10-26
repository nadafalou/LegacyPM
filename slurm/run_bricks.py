"""Run the per-brick pipeline for whole RA degree slices inside one Slurm job (no FireWorks / MongoDB needed).
Same steps as fireworks/legacypm_workflow.py. The big calibration tables (the CCD table brick_DCR loads at import:
~45 s and 14 GB) are loaded once in the parent process and shared with forked worker processes.
Restartable: bricks whose tractor-pm-<brick>.fits already exists (non-empty) are skipped.

Each brick's input is taken from the first source that has it (see SOURCES):
  scratch  Dustin's current DR10 forced photometry on pscratch (newest)
  cfs      Dustin's Dec 2024 - Jan 2025 DR10 run on CFS (RA 000-057)
  archive  our Feb-Mar 2025 output (RA 060-063); raw inputs purged, so inputs are rebuilt from it and all steps rerun

  python run_bricks.py 000-063 --dry-run          # count bricks per slice and source
  python run_bricks.py 000-015 --workers 128      # run
  python run_bricks.py 064 --input-root DIR       # use only DIR (testing); add --from-archive if DIR is archive-like
  python run_bricks.py 000-063 --redo-errors      # finished bricks: new error_model.json -> errors + PM refit only
Per-brick logs: <out-root>/<deg>/logs/<brick>.log; failures listed in <out-root>/<deg>/failed.txt
"""
import os
import sys
import gc
import time
import argparse
import logging
import warnings
import contextlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "py_files")))

# name -> (root, is_archive), in priority order
SOURCES = {
    "scratch": ("/pscratch/sd/d/dstn/forced-motions-dr10/forced-brick", False),
    "cfs": ("/global/cfs/cdirs/cosmo/work/users/dstn/forced-motions-dr10/forced-brick", False),
    "archive": ("/pscratch/sd/n/nelfalou/archive/legacypm-catalogue", True),
}
# v2 (2026-10-23): rerun with the two-line DCR, rebuilt CCD maps, refitted lateral model, per-band errors, Dec-ivar fix,
# objid fix and quality flags. v1 (the Oct 2026 run) stays in /pscratch/sd/n/nelfalou/legacypm-catalogue until checked.
DEFAULT_OUT_ROOT = "/pscratch/sd/n/nelfalou/legacypm-catalogue-v2"

# set in the parent before forking, shared (copy-on-write) by the workers
STATE = {}


def parse_degrees(items):
    out = []
    for item in items:
        lo, _, hi = item.partition("-")
        out += [f"{d:03d}" for d in range(int(lo), int(hi or lo) + 1)]
    return out


def listing(path):
    """{filename: size} for a directory, {} if it doesn't exist."""
    try:
        return {e.name: e.stat().st_size for e in os.scandir(path)}
    except FileNotFoundError:
        return {}


def find_bricks(degree, sources, new_dir):
    """[(brick, src_dir, is_archive)] for unfinished bricks, each from the first source that has complete inputs."""
    done = {f[11:19] for f, size in listing(new_dir).items() if f.startswith("tractor-pm-") and size > 0}
    chosen = {}
    for root, is_archive in sources:
        src_dir = os.path.join(root, degree) + "/"
        files = listing(src_dir)
        needed = "tractor-pm-{}.fits" if is_archive else "tractor-forced-{}.fits"
        for f, size in files.items():
            if not (f.startswith("forced-") and f.endswith(".fits")) or size == 0:
                continue
            brick = f[-13:-5]
            if brick in chosen or brick in done or files.get(needed.format(brick), 0) == 0:
                continue
            chosen[brick] = (src_dir, is_archive)
    return [(b,) + chosen[b] for b in sorted(chosen)], len(done)


def find_done_bricks(degree, sources, new_dir, model_id):
    """[(brick, tractor_dir)] for finished bricks whose catalogue was not made with error model `model_id`
    (for --redo-errors). tractor_dir: new_dir for archive bricks (prep wrote tractor-forced there), else the first
    input source that has tractor-forced-<brick>.fits."""
    out_files = listing(new_dir)
    done = sorted(f[11:19] for f, size in out_files.items() if f.startswith("tractor-pm-") and size > 0)
    src_files = [(os.path.join(root, degree) + "/", listing(os.path.join(root, degree) + "/"))
                 for root, is_archive in sources if not is_archive]
    todo, n_current = [], 0
    for brick in done:
        h = fitsio.read_header(os.path.join(new_dir, f"tractor-pm-{brick}.fits"), 1)
        if h.get("ERRMODEL") == model_id[0] and str(h.get("ZPTABLE", '')) == model_id[1]:
            n_current += 1
            continue
        name = f"tractor-forced-{brick}.fits"
        tdir = new_dir if out_files.get(name, 0) > 0 else next((d for d, files in src_files if files.get(name, 0) > 0), None)
        if tdir is not None:
            todo.append((brick, tdir))
    return todo, n_current


def run_redo_errors(brick, new_dir, tractor_dir):
    """Apply the current error model to a finished brick: rewrite the total-error columns of its corrected forced
    file (positions untouched), then refit its PMs."""
    t0 = time.time()
    fn = f"forced-{brick}.fits"
    S = STATE
    root = logging.getLogger()
    with open(os.path.join(new_dir, "logs", f"{brick}.log"), "a") as log, contextlib.redirect_stdout(log):
        handler = logging.StreamHandler(log)
        root.addHandler(handler)
        try:
            print(f"### {brick}: redo errors + PM (tractor from {tractor_dir})", flush=True)
            for name, step in (("errors", lambda: S["lm"].update_total_errors(os.path.join(new_dir, fn))),
                               ("PM", lambda: S["pm"].create_brick_catalogue(fn, new_dir, new_dir, tractor_dir))):
                try:
                    step()
                except Exception as e:
                    logging.exception(f"{name} step failed")
                    return brick, f"{name}: {type(e).__name__}: {e}", time.time() - t0
        finally:
            root.removeHandler(handler)
    return brick, None, time.time() - t0


def load_state(need_dcr):
    if need_dcr:
        import brick_DCR  # loads the CCD table
        STATE["dcr"] = brick_DCR
    import brick_ringmaps, brick_lateralmaps, create_brick_catalogue, prep_archive_rerun
    STATE.update(rm=brick_ringmaps, lm=brick_lateralmaps, pm=create_brick_catalogue, prep=prep_archive_rerun,
                 tweaks=brick_ringmaps.make_splines(brick_ringmaps.get_ringmaps(brick_ringmaps.RINGMAPS_PATH)),
                 lm_model=brick_lateralmaps.load_lateral_model())


def worker_init():
    warnings.simplefilter("ignore")
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def run_brick(brick, src_dir, new_dir, is_archive):
    t0 = time.time()
    fn = f"forced-{brick}.fits"
    tractor_dir = new_dir if is_archive else src_dir
    S = STATE
    # archive bricks: rebuild the inputs in new_dir (derived columns dropped), then redo every step from there
    steps = ([("prep", lambda: S["prep"].prep_brick(fn, new_dir, src_dir))] if is_archive else []) + [
             ("DCR", lambda: S["dcr"].create_brick_corrected_data(fn, new_dir, tractor_dir if is_archive else src_dir)),
             ("RM", lambda: S["rm"].create_brick_corrected_data(fn, new_dir, new_dir, S["tweaks"])),
             ("LM", lambda: S["lm"].create_brick_corrected_data(fn, new_dir, new_dir, tractor_dir, S["lm_model"])),
             ("PM", lambda: S["pm"].create_brick_catalogue(fn, new_dir, new_dir, tractor_dir))]
    root = logging.getLogger()
    with open(os.path.join(new_dir, "logs", f"{brick}.log"), "w") as log, contextlib.redirect_stdout(log):
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(handler)
        try:
            print(f"### {brick} from {src_dir} ({'archive' if is_archive else 'forced photometry'})", flush=True)
            for name, step in steps:
                try:
                    step()
                except Exception as e:
                    logging.exception(f"{name} step failed")
                    return brick, f"{name}: {type(e).__name__}: {e}", time.time() - t0
        finally:
            root.removeHandler(handler)
    return brick, None, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("degrees", nargs="+", help="RA degree slices or ranges, e.g. 060 061 or 000-063")
    ap.add_argument("--sources", default=",".join(SOURCES), help=f"comma-separated, in priority order (default all)")
    ap.add_argument("--input-root", default=None, help="use only this root (overrides --sources)")
    ap.add_argument("--from-archive", action="store_true", help="with --input-root: it holds archive-style bricks")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo-errors", action="store_true",
                    help="finished bricks only: rewrite total errors (error_model.json) and exposure-CCD zero points "
                         "(zeropoints.py table), then refit PMs (skips bricks already made with both current)")
    args = ap.parse_args()

    sources = ([(args.input_root, args.from_archive)] if args.input_root
               else [SOURCES[s] for s in args.sources.split(",")])
    degrees = parse_degrees(args.degrees)
    if args.redo_errors:
        import fitsio, error_model, zeropoints, brick_lateralmaps, create_brick_catalogue
        jobs = []
        for degree in degrees:
            new_dir = os.path.join(args.out_root, degree) + "/"
            mid = (error_model.ERROR_MODEL_ID, zeropoints.table_id())
            todo, n_current = find_done_bricks(degree, sources, new_dir, mid)
            print(f"degree {degree}: {n_current} already on error model {mid[0]} + zero points {mid[1] or 'none'}, {len(todo)} to redo", flush=True)
            jobs += [(b, new_dir, t) for b, t in todo]
        print(f"total to redo: {len(jobs)}", flush=True)
        if args.dry_run or not jobs:
            sys.exit(0)
        STATE.update(lm=brick_lateralmaps, pm=create_brick_catalogue)
        t0 = time.time(); n_ok, failed = 0, []
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"), initializer=worker_init) as ex:
            futs = [ex.submit(run_redo_errors, *job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                brick, err, dt = fut.result()
                if err:
                    failed.append((brick, err)); print(f"[{i}/{len(jobs)}] FAILED {brick}: {err}", flush=True)
                else:
                    n_ok += 1; print(f"[{i}/{len(jobs)}] done {brick} ({dt:.0f} s)", flush=True)
        for degree in degrees:
            if os.path.isdir(os.path.join(args.out_root, degree)):
                with open(os.path.join(args.out_root, degree, "failed_redo_errors.txt"), "w") as f:
                    f.writelines(f"{b} {e}\n" for b, e in failed if b[:3] == degree)
        print(f"finished: {n_ok} ok, {len(failed)} failed in {(time.time() - t0) / 3600:.2f} h", flush=True)
        sys.exit(0)
    jobs = []
    for degree in degrees:
        new_dir = os.path.join(args.out_root, degree) + "/"
        found, n_done = find_bricks(degree, sources, new_dir)
        by_src = {}
        for _, src_dir, _ in found:
            by_src[src_dir] = by_src.get(src_dir, 0) + 1
        print(f"degree {degree}: {n_done} done, {len(found)} to run "
              + ", ".join(f"{n} from {s}" for s, n in by_src.items()), flush=True)
        if found and not args.dry_run:
            os.makedirs(os.path.join(new_dir, "logs"), exist_ok=True)
        jobs += [(b, s, new_dir, a) for b, s, a in found]
    print(f"total to run: {len(jobs)}", flush=True)
    if args.dry_run or not jobs:
        sys.exit(0)

    t0 = time.time()
    load_state(need_dcr=bool(jobs))
    print(f"loaded calibration tables in {time.time() - t0:.0f} s", flush=True)
    gc.freeze()  # keep the shared tables' pages from being copied into every worker by the garbage collector

    n_ok, failed = 0, []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork"), initializer=worker_init) as ex:
        futs = [ex.submit(run_brick, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            brick, err, dt = fut.result()
            if err:
                failed.append((brick, err))
                print(f"[{i}/{len(jobs)}] FAILED {brick}: {err}", flush=True)
            else:
                n_ok += 1
                print(f"[{i}/{len(jobs)}] done {brick} ({dt:.0f} s)", flush=True)
    for degree in degrees:  # bricks are named after their RA degree, e.g. 0600m050 -> 060
        if os.path.isdir(os.path.join(args.out_root, degree)):
            with open(os.path.join(args.out_root, degree, "failed.txt"), "w") as f:
                f.writelines(f"{b} {e}\n" for b, e in failed if b[:3] == degree)
    print(f"finished: {n_ok} ok, {len(failed)} failed in {(time.time() - t0) / 3600:.2f} h", flush=True)
