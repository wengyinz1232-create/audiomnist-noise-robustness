"""Resume the same full L2 experiment, scheduling independent folds in parallel.

Run only when rerun_lightgbm_l2.py is not actively training. No score reuse:
every missing outer fold performs its full three-by-three inner search.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from common import ROOT, SEED, CONDITIONS, sha, save_json
from evaluation import grouped_folds, train_fold
import rerun_lightgbm_l2 as experiment

STATE=None


def initialize():
    global STATE
    experiment.setup()
    with np.load(ROOT/'data'/'mfcc_delta.npz') as d:
        y,groups=d['y'],d['groups'];variants={c:d[c] for c in CONDITIONS}
    STATE=(variants,y,groups,list(grouped_folds(groups,np.arange(len(y)),10,SEED)))


def work(args):
    mode,number=args;variants,y,groups,folds=STATE
    tr,te=folds[number-1] if number else (np.arange(len(y)),np.array([],dtype=int))
    out=ROOT/'runs'/experiment.RUN/f'mfcc_delta__lightgbm_goss_l2__{mode}'
    result=train_fold(variants,y,groups,'lightgbm_goss_l2',mode,experiment.GRID,number,tr,te,out,'mfcc_delta')
    return mode,result


def main():
    plan=json.loads((experiment.OUT/'plan.json').read_text())
    for name,digest in plan['source'].items():
        p=ROOT/'data'/name if name.endswith('.npz') else ROOT/name
        assert sha(p)==digest
    records={};paths={};tasks=[]
    for m in ('clean_matched','augmented'):
        p=ROOT/'runs'/experiment.RUN/f'mfcc_delta__lightgbm_goss_l2__{m}'/'result.json'
        d=json.loads(p.read_text());assert d['config']['grid']==experiment.GRID
        records[m]=d;paths[m]=p;done={r['fold'] for r in d['folds']}
        tasks.extend((m,f) for f in range(1,11) if f not in done)
        if 'full_fit' not in d:tasks.append((m,0))
    tasks.sort(key=lambda x:(x[1]==0,x[1],x[0]))
    save_json(experiment.OUT/'parallel_completion_plan.json',{'script_sha256':sha(Path(__file__)),'workers':6,
              'pending_tasks':tasks,'scope':'Scheduling change only. Original source, grids, seeds, inner search and model settings remain unchanged. No inner scores reused.'})
    with concurrent.futures.ProcessPoolExecutor(max_workers=6,initializer=initialize) as pool:
        for future in concurrent.futures.as_completed([pool.submit(work,t) for t in tasks]):
            mode,row=future.result()
            if row['fold']:records[mode]['folds'].append(row);records[mode]['folds'].sort(key=lambda r:r['fold'])
            else:records[mode]['full_fit']=row
            save_json(paths[mode],records[mode])
            print('DONE',mode,'fold',row['fold'],'selected',row['selected'],'seconds',row['seconds'],flush=True)
    # Original orchestrator now skips completed fits and performs all its audits.
    experiment.main()


if __name__=='__main__':main()
