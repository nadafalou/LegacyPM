import os, sys, glob, logging
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e import DIRS, ARCHIVE, create_brick_catalogue

def pm(b):
    logging.getLogger().setLevel(logging.WARNING)
    fn = f'forced-{b}.fits'
    create_brick_catalogue.create_brick_catalogue(fn, DIRS['pm_new'], DIRS['lm'], DIRS['in'])
    create_brick_catalogue.create_brick_catalogue(fn, DIRS['pm_old'], f'{ARCHIVE}/{b[:3]}/', DIRS['in'])
    return b

if __name__ == '__main__':
    bricks = sorted(f[7:15] for f in os.listdir(DIRS['lm']) if f.startswith('forced-'))
    with ProcessPoolExecutor(len(bricks)) as ex:
        for b in ex.map(pm, bricks):
            print('done', b, flush=True)
