"""One entry point; independent of the old project folders."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import REPRESENTATIONS, MODELS, MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['all','prepare','train','verify','summarize','compare','ensemble'])
    p.add_argument('--jobs',type=int,default=3); p.add_argument('--download',action='store_true')
    p.add_argument('--run-id',default='main')
    p.add_argument('--representations',nargs='+',choices=REPRESENTATIONS,default=REPRESENTATIONS)
    p.add_argument('--models',nargs='+',choices=MODELS+('lightgbm_goss_min_child',),default=MODELS)
    p.add_argument('--modes',nargs='+',choices=MODES+('multi_snr_augmented',),default=MODES)
    p.add_argument('--grid',nargs=3,type=float)
    args = p.parse_args()
    if args.jobs<1 or args.jobs>4: p.error('Use 1 through 4 worker processes')
    if args.grid and (len(args.models)!=1 or args.run_id=='main'): p.error('Custom grids need one model and a new --run-id')
    if args.action in ('all','prepare'):
        from prepare_data import prepare
        prepare(args.download)
    if args.action in ('all','train'):
        from evaluation import run_configuration
        tasks = [(r,m,s,args.run_id,args.grid) for r in args.representations for m in args.models for s in args.modes]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for future in as_completed([pool.submit(run_configuration,*t) for t in tasks]):
                print('CONFIG_COMPLETE',future.result(),flush=True)
    if args.action in ('all','verify'):
        from verify import verify
        verify()
    if args.action in ('all','summarize'):
        from summarize import summarize
        summarize()
    if args.action in ('all','compare'):
        from common import ROOT
        from compare_results import analyse, write_outputs
        summary, pairs = analyse(ROOT/'runs'/'main')
        out = ROOT/'results'/'comparison'
        write_outputs(out,summary,pairs)
        print('COMPARISON_COMPLETE',out,flush=True)
    if args.action == 'ensemble':
        from common import ROOT, save_json
        from evaluate_ensemble import evaluate
        target=ROOT/'results'/'ensemble_equal_weight.json'
        save_json(target,evaluate(ROOT/'runs'/'main'))
        print('ENSEMBLE_COMPLETE',target,flush=True)


if __name__ == '__main__': main()
