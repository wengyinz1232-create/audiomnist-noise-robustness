"""Run honest argmax CV under locked configuration grids; never force the midpoint.

--reuse-inner-screen explicitly reuses exploratory scores, with hash checks.
Omit it to calculate all three inner folds again under the now-locked grid.
"""
import argparse
import itertools
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import tune_rf as rf
from audit_midpoints import audit

def candidate(config,value):
    p=rf.params(16,2,'sqrt',60)
    p.update(config['fixed']); p[config['parameter']]=value
    return p

def local_config(config,rep,mode):
    if 'grids' in config:
        return dict(config,grid=config['grids'][f'{rep}__forest__{mode}'])
    return config

def validate_config(config):
    if 'grids' in config and 'grid' not in config:
        expected={f'{r}__forest__{m}' for r,m in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented'])}
        if set(config['grids'])!=expected:
            raise ValueError('All four configuration grids must be provided.')
        for rep,mode in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']):
            validate_config(local_config(config,rep,mode))
        return
    name=config['parameter']; grid=config['grid']
    if name not in ['max_depth','max_features'] or len(grid)!=3 or len(set(grid))!=3:
        raise ValueError('Exactly three distinct supported parameter values are required.')
    if name=='max_depth':
        if any(x is not None and (type(x) is not int or x<1) for x in grid):
            raise ValueError('Depth must be a positive integer or None.')
        numeric=[float('inf') if x is None else x for x in grid]
    else:
        if any(type(x) is not float or not 0<x<=1 for x in grid):
            raise ValueError('Feature fractions must be floats in (0,1].')
        numeric=grid
    if numeric!=sorted(numeric):
        raise ValueError('Use increasing values; do not reorder candidates to manufacture a middle.')

def run(cache_dir,output,config,rep,mode,fold,screen):
    config=local_config(config,rep,mode)
    cache_dir,output=Path(cache_dir),Path(output)
    folder=output/f'{rep}__forest__{mode}'/f'fold_{fold:02}'
    path=folder/'result.json'; started=time.perf_counter()
    if path.exists():
        row=json.loads(path.read_text(encoding='utf-8'))
        assert row['model_sha256']==rf.sha(folder/'model.pkl')
        assert row['predictions_sha256']==rf.sha(folder/'predictions.npz')
        return row
    with np.load(cache_dir/f'{rep}.npz') as z:
        y,groups=z['y'],z['groups']
        variants={c:z[c] for c in ['clean','white_10']}
    train,test=list(rf.ev.grouped_folds(groups,np.arange(len(y)),10,rf.SEED))[fold-1]
    inner=list(rf.ev.grouped_folds(groups,train,3,rf.SEED+fold))
    scores=[]; origins=[]
    for value in config['grid']:
        p=candidate(config,value)
        if screen:
            s=Path(screen)/f'{rep}__forest__{mode}'/f'fold_{fold:02}'
            if config['parameter']=='max_depth': s=s/'candidates'
            source=s/(rf.key(p)+'.json')
            row=json.loads(source.read_text(encoding='utf-8'))
            assert row['params']==p and len(row['inner'])==3
            scores.append([r['score'] for r in row['inner']])
            origins.append(dict(source=str(source),sha256=rf.sha(source)))
        else:
            vals=[]
            for j,(fit_idx,valid_idx) in enumerate(inner):
                x,yy=rf.ev.training_rows(variants,y,fit_idx,mode); stats=rf.ev.scaling(x)
                model,_,ws=rf.fit(p,rf.ev.transform(x,stats),yy,rf.SEED+100*fold+j)
                assert not ws,ws
                valid=[rf.ev.transform(variants[c][valid_idx],stats) for c in ['clean','white_10']]
                vals.append(rf.score(model,valid,y[valid_idx]))
            scores.append(vals)
    means=np.mean(scores,axis=1); index=int(np.argmax(means)); chosen=config['grid'][index]
    p=candidate(config,chosen)
    rf.save(folder/'selection.json',dict(grid=config['grid'],selected=chosen,params=p,inner_scores=scores,
                                       inner_means=means.tolist(),origins=origins))
    with np.load(cache_dir/f'{rep}.npz') as z:
        variants.update({c:z[c] for c in ['white_20','pink_10']})
    x,yy=rf.ev.training_rows(variants,y,train,mode); stats=rf.ev.scaling(x)
    model,seconds,ws=rf.fit(p,rf.ev.transform(x,stats),yy,rf.SEED+fold)
    bundle=dict(estimator=model,scaler=stats,classes=np.arange(10),representation=rep,mode=mode,
                params=p,train_people=np.unique(groups[train]).tolist())
    with (folder/'model.pkl').open('wb') as f: pickle.dump(bundle,f,protocol=5)
    with (folder/'model.pkl').open('rb') as f: reloaded=pickle.load(f)
    probabilities={}
    for c in rf.CONDITIONS:
        xt=rf.ev.transform(variants[c][test],stats)
        probabilities[c]=rf.ev.proba(model,xt,np.arange(10))
        np.testing.assert_allclose(probabilities[c],rf.ev.proba(reloaded['estimator'],xt,np.arange(10)),atol=1e-12,rtol=1e-12)
    np.savez_compressed(folder/'predictions.npz',indices=test,y=y[test],groups=groups[test],classes=np.arange(10),**probabilities)
    row=dict(representation=rep,mode=mode,fold=fold,grid=config['grid'],parameter=config['parameter'],selected=chosen,
             params=p,inner_scores=scores,inner_means=means.tolist(),origins=origins,
             train_people=np.unique(groups[train]).tolist(),test_people=np.unique(groups[test]).tolist(),
             inner_valid_people=[np.unique(groups[v]).tolist() for _,v in inner],
             metrics={c:rf.ev.measure(y[test],probabilities[c],np.arange(10)) for c in rf.CONDITIONS},
             fit_seconds=seconds,warnings=ws,seconds=time.perf_counter()-started,reload_verified=True,
             model_sha256=rf.sha(folder/'model.pkl'),predictions_sha256=rf.sha(folder/'predictions.npz'))
    rf.save(path,row)
    print('MIDPOINT_DONE',rep,mode,fold,'selected',chosen,flush=True)
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-inner-screen',type=Path); p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); config=json.loads(a.config.read_text(encoding='utf-8'))
    validate_config(config)
    for name,expected in config['source_sha256'].items():
        assert rf.sha(rf.ROOT/'source'/name)==expected
    for rep in ['logmel','mfcc_delta']:
        assert rf.sha(a.cache_dir/f'{rep}.npz')==config['feature_sha256'][rep]
    if a.reuse_inner_screen:
        origin=json.loads((a.reuse_inner_screen/'protocol.json').read_text(encoding='utf-8'))
        assert origin['feature_sha256']==config['feature_sha256']
        if config['parameter']=='max_features':
            import screen_rf_features
            assert origin['source_sha256']==rf.sha(screen_rf_features.__file__)
        else:
            assert origin['runner_sha256']==rf.sha(rf.__file__)
    protocol=dict(config=config,config_sha256=rf.sha(a.config),runner_sha256=rf.sha(__file__),
                  sources={x.name:rf.sha(x) for x in (rf.ROOT/'source').glob('*.py')},
                  reuse_inner_screen=str(a.reuse_inner_screen) if a.reuse_inner_screen else None,
                  note='Exploratory global grid design; score-maximizing selection, no forced middle or outer-score-based changes.')
    pp=a.output/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text(encoding='utf-8'))==protocol
    else: rf.save(pp,protocol)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run,str(a.cache_dir),str(a.output),config,r,m,f,
                        str(a.reuse_inner_screen) if a.reuse_inner_screen else None)
            for r,m in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']) for f in range(1,11)]
        for f in as_completed(fs): rows.append(f.result())
    results={}
    for rep,mode in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']):
        relevant=sorted([r for r in rows if r['representation']==rep and r['mode']==mode],key=lambda r:r['fold'])
        assert [r['fold'] for r in relevant]==list(range(1,11))
        check=audit([r['inner_means'] for r in relevant])
        results[f'{rep}__forest__{mode}']=check
    passed=all(r['middle_unique_mode'] and not r['middle_tie_folds'] for r in results.values())
    rf.save(a.output/'midpoint_check.json',dict(status='PASS' if passed else 'FAIL',configurations=results))
    print('MIDPOINT_CHECK', 'PASS' if passed else 'FAIL',flush=True)
    # A failed check is reported as-is; never change winners or fabricate a passing count.

if __name__=='__main__': main()
