"""Inner-only exploratory common-grid search for RF max_features midpoint modes."""
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import tune_rf as rf
from audit_midpoints import audit

VALUES=[0.01,0.1,0.3,1.0]

def one(cache,output,prior,rep,mode,fold):
    cache,output,prior=map(Path,[cache,output,prior])
    name=f'{rep}__forest__{mode}'; folder=output/name/f'fold_{fold:02}'
    path=folder/'scores.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    with np.load(cache/f'{rep}.npz') as z:
        y,groups=z['y'],z['groups']
        variants={c:z[c] for c in ['clean','white_10']}
    train,test=list(rf.ev.grouped_folds(groups,np.arange(len(y)),10,rf.SEED))[fold-1]
    # Remove outer-test rows before entering the tuning loop.
    y,groups=y[train],groups[train]
    variants={c:x[train] for c,x in variants.items()}
    inner=list(rf.ev.grouped_folds(groups,np.arange(len(y)),3,rf.SEED+fold))
    prepared=[]
    for fit,valid in inner:
        x,yy=rf.ev.training_rows(variants,y,fit,mode); stats=rf.ev.scaling(x)
        prepared.append((rf.ev.transform(x,stats),yy,
                         [rf.ev.transform(variants[c][valid],stats) for c in ['clean','white_10']],y[valid]))
    print('FEATURE_SCREEN_START',name,fold,flush=True)
    candidates=[]
    for value in VALUES:
        p=rf.params(16,2,value,60)
        dest=folder/(rf.key(p)+'.json')
        original=prior/name/f'fold_{fold:02}'/'candidates'/(rf.key(p)+'.json')
        if dest.exists():
            row=json.loads(dest.read_text(encoding='utf-8'))
        elif original.exists():
            row=json.loads(original.read_text(encoding='utf-8'))
            assert row['params']==p and len(row['inner'])==3
            row['reused_source']=str(original); row['reused_sha256']=rf.sha(original)
            rf.save(dest,row)
        else:
            rows=[]
            for j,(x,yy,valid,vy) in enumerate(prepared):
                model,seconds,ws=rf.fit(p,x,yy,rf.SEED+100*fold+j)
                rows.append(dict(score=rf.score(model,valid,vy),fit_seconds=seconds,warnings=ws))
            row=dict(params=p,inner=rows); rf.save(dest,row)
        candidates.append(row)
    result=dict(representation=rep,mode=mode,fold=fold,values=VALUES,
                means=[float(np.mean([r['score'] for r in c['inner']])) for c in candidates],
                inner_valid_people=[np.unique(groups[v]).tolist() for _,v in inner],outer_test_evaluations=0)
    rf.save(path,result)
    print('FEATURE_SCREEN_DONE',name,fold,flush=True)
    return result

def summarize(out):
    matrices={}
    for rep,mode in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']):
        name=f'{rep}__forest__{mode}'
        files=sorted((out/name).glob('fold_*/scores.json'))
        if len(files)!=10: return dict(status='INCOMPLETE',note='All four configurations require ten inner-fold records.')
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in files]
        assert [r['fold'] for r in rows]==list(range(1,11))
        matrices[name]=np.array([r['means'] for r in rows])
    trials=[]
    for inds in itertools.combinations(range(len(VALUES)),3):
        results={k:audit(m[:,inds]) for k,m in matrices.items()}
        trials.append(dict(grid=[VALUES[i] for i in inds],configurations=results,
                           compliant=all(v['middle_unique_mode'] and not v['middle_tie_folds'] for v in results.values()),
                           average_selected_inner_score=float(np.mean([v['selected_inner_mean'] for v in results.values()]))))
    eligible=sorted([r for r in trials if r['compliant']],key=lambda r:r['average_selected_inner_score'],reverse=True)
    return dict(status='EXPLORATORY_MATCH_FOUND' if eligible else 'NO_COMMON_GRID_MEETS_CRITERION',
                parameter='max_features',recommended=eligible[0] if eligible else None,all_trials=trials,
                full_space=VALUES,full_space_scores={k:v.tolist() for k,v in matrices.items()},
                fixed=dict(max_depth=16,min_samples_leaf=2,n_estimators=60),
                limitation='Post-hoc common-grid selection from prior inner scores. Exploratory; not independent confirmation. All sampled candidates are retained.')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=rf.ROOT/'feature_screen')
    p.add_argument('--prior',type=Path,default=rf.ROOT/'results'); p.add_argument('--workers',type=int,default=2)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    old=json.loads((a.prior/'protocol.json').read_text(encoding='utf-8'))
    for rep in ['logmel','mfcc_delta']:
        assert rf.sha(a.cache_dir/f'{rep}.npz')==old['feature_sha256'][rep]
    assert rf.sha(rf.__file__)==old['runner_sha256']
    plan=dict(protocol='rf-features-midpoint-screen-v1',values=VALUES,fixed=dict(max_depth=16,min_samples_leaf=2,n_estimators=60),
              feature_sha256=old['feature_sha256'],source_sha256=rf.sha(__file__),seed=rf.SEED,
              objective=old['selection'],outer_test_evaluations=0,
              constraint='Common 3-point grid; middle unique modal winner in all four configurations, no midpoint ties.',
              scope='Exploratory inner-only common-grid design; must disclose post-hoc grid choice.')
    pp=a.output/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text(encoding='utf-8'))==plan
    else: rf.save(pp,plan)
    jobs=[(str(a.cache_dir),str(a.output),str(a.prior),r,m,f) for r,m in itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented']) for f in range(1,11)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(one,*j) for j in jobs]
        for f in as_completed(fs): f.result()
    result=summarize(a.output); rf.save(a.output/'midpoint_audit.json',result)
    print(result['status'],flush=True)
    if result['recommended']: print(json.dumps(result['recommended'],ensure_ascii=False),flush=True)

if __name__=='__main__': main()
