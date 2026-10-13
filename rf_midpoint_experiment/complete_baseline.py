"""Reuse completed paired baselines; finish only missing baseline folds."""
import argparse
import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import tune_rf as rf
import train_rf_midpoint as runner

def one(cache,prior,out,rep,mode,fold):
    prior,out=Path(prior),Path(out)
    name=f'{rep}__forest__{mode}'; source=prior/name/f'fold_{fold:02}'
    folder=out/name/f'fold_{fold:02}'; destination=folder/'result.json'
    if destination.exists(): return
    folder.mkdir(parents=True,exist_ok=True)
    if (source/'result.json').exists():
        original=json.loads((source/'result.json').read_text(encoding='utf-8'))
        model=original['models']['baseline']
        assert rf.sha(source/'baseline.pkl')==model['model_sha256']
        assert rf.sha(source/'baseline_predictions.npz')==model['predictions_sha256']
        shutil.copy2(source/'baseline.pkl',folder/'baseline.pkl')
        shutil.copy2(source/'baseline_predictions.npz',folder/'baseline_predictions.npz')
        r={k:original[k] for k in ['representation','mode','fold','train_people','test_people','inner_valid_people']}
        r['models']=dict(baseline=model)
        r['origin']=dict(reused_result=str(source/'result.json'),sha256=rf.sha(source/'result.json'))
        rf.save(destination,r)
        print('BASELINE_REUSED',name,fold,flush=True)
        return
    config=dict(parameter='max_depth',grid=[4,48,128],fixed=dict(n_estimators=60,min_samples_leaf=2,max_features='sqrt'))
    have=all((source/'candidates'/(rf.key(rf.params(d))+'.json')).exists() for d in config['grid'])
    # New raw output is separate because its result schema differs.
    r=runner.run(cache,str(out/'new_refits'),config,rep,mode,fold,str(prior) if have else None)
    raw=out/'new_refits'/name/f'fold_{fold:02}'
    shutil.copy2(raw/'model.pkl',folder/'baseline.pkl')
    shutil.copy2(raw/'predictions.npz',folder/'baseline_predictions.npz')
    model={k:r[k] for k in ['params','metrics','fit_seconds','warnings','model_sha256','predictions_sha256','reload_verified']}
    model['inner_mean']=float(max(r['inner_means']))
    row={k:r[k] for k in ['representation','mode','fold','train_people','test_people','inner_valid_people']}
    row['models']=dict(baseline=model); row['origin']=dict(refit_result=str(raw/'result.json'),sha256=rf.sha(raw/'result.json'))
    rf.save(destination,row)
    print('BASELINE_FINISHED',name,fold,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--prior',type=Path,default=rf.ROOT/'results')
    p.add_argument('--output',type=Path,default=rf.ROOT/'baseline_results'); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); plan=json.loads((a.prior/'protocol.json').read_text(encoding='utf-8'))
    for rep in ['logmel','mfcc_delta']: assert rf.sha(a.cache_dir/f'{rep}.npz')==plan['feature_sha256'][rep]
    rf.save(a.output/'protocol.json',dict(scope='Complete old [4,48,128] baseline, reusing verified completed folds where available.',
                                        original_protocol_sha256=rf.sha(a.prior/'protocol.json'),runner_sha256=rf.sha(__file__),feature_sha256=plan['feature_sha256']))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(one,str(a.cache_dir),str(a.prior),str(a.output),r,m,f)
            for r,m in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']) for f in range(1,11)]
        for f in as_completed(fs): f.result()
    assert len(list(a.output.glob('*__forest__*/fold_*/result.json')))==40
    print('BASELINE_ALL_40_COMPLETE',flush=True)

if __name__=='__main__': main()
