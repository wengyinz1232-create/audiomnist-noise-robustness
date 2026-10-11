"""Bounded inner-only diagnostic; never evaluates outer test recordings."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import os
from threadpoolctl import ThreadpoolController
from common import ROOT, SEED, sha, save_json
from evaluation import grouped_folds, training_rows, scaling, transform, fit, proba, measure

OUT = ROOT/'results'/'diagnosis_20261011'


def main():
    # Windows snapshot API can mangle non-ASCII DLL paths; use its wide-path fallback.
    if os.name == 'nt':
        ThreadpoolController._snapshot_loaded_modules = lambda self, kernel: None
    OUT.mkdir(exist_ok=True)
    with np.load(ROOT/'data'/'mfcc_delta.npz') as data:
        y, groups = data['y'], data['groups']
        variants = {c:data[c] for c in ('clean','white_10')}
    signature = {p.name:sha(p) for p in (ROOT/'data'/'mfcc_delta.npz', ROOT/'evaluation.py', Path(__file__))}
    grids = {'logistic':[.01,.1,1.,10.,100.], 'lightgbm_goss':[2,3,7]}
    plan = {'signature':signature,'grids':grids,'objective':'mean(clean, white_10) inner macro-F1',
            'scope':'Exploratory inner-only diagnostics on previously explored splits; no outer scores.',
            'reuse':'Main-run inner scores only after feature hash, folds and settings checks; historical probes are not reused.'}
    save_json(OUT/'plan.json',plan)
    trees=[]
    for mode in ('clean_matched','augmented'):
        folder=ROOT/'runs'/'main'/f'mfcc_delta__lightgbm_goss__{mode}'
        doc=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        for row in doc['folds']:
            path=folder/'models'/f"fold_{row['fold']:02}.pkl"
            assert sha(path)==row['model_sha256']
            with path.open('rb') as stream: model=pickle.load(stream)['estimator']
            leaves=[t['num_leaves'] for t in model.booster_.dump_model()['tree_info']]
            trees.append({'mode':mode,'fold':row['fold'],'selected':row['selected'],
                          'trees':len(leaves),'min_leaves':min(leaves),'max_leaves':max(leaves),
                          'mean_leaves':float(np.mean(leaves))})
    save_json(OUT/'saved_tree_structure.json',trees)
    classes=np.unique(y)
    folds=list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    for name,grid in grids.items():
        for mode in ('clean_matched','augmented'):
            path=OUT/f'{name}__{mode}.json'
            doc=json.loads(path.read_text()) if path.exists() else {'plan':plan,'rows':[]}
            assert doc['plan']==plan
            old=json.loads((ROOT/'runs'/'main'/f'mfcc_delta__{name}__{mode}'/'result.json').read_text())
            assert old['config']['signature']['mfcc_delta.npz']==signature['mfcc_delta.npz']
            oldrows={r['fold']:r for r in old['folds']}
            for number,(train,_) in enumerate(folds,1):
                inner=list(grouped_folds(groups,train,3,SEED+number))
                reference=oldrows[number]
                assert reference['train_people']==np.unique(groups[train]).tolist()
                assert reference['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in inner]
                for j,(tr,va) in enumerate(inner):
                    pending=[v for v in grid if not any(r['fold']==number and r['inner']==j and r['value']==v for r in doc['rows'])]
                    if not pending: continue
                    x,yy=training_rows(variants,y,tr,mode); stats=scaling(x); x=transform(x,stats)
                    xv=[transform(variants[c][va],stats) for c in ('clean','white_10')]
                    for value in pending:
                        start=time.perf_counter()
                        if value in old['config']['grid']:
                            score=reference['inner_scores'][old['config']['grid'].index(value)][j]
                            notes=[]; source='main saved inner score'
                        else:
                            model,notes=fit(name,value,x,yy,SEED+100*number+j)
                            score=float(np.mean([measure(y[va],proba(model,z,classes),classes)['macro_f1'] for z in xv]))
                            source='new fit'
                        doc['rows'].append({'fold':number,'inner':j,'value':value,'score':score,'warnings':notes,'source':source,'seconds':time.perf_counter()-start})
                        save_json(path,doc)
                    print(name,mode,'fold',number,'inner',j+1,flush=True)
            means=[[np.mean([r['score'] for r in doc['rows'] if r['fold']==f and r['value']==v]) for v in grid] for f in range(1,11)]
            selected=[grid[int(np.argmax(s))] for s in means]
            doc['summary']={'grid':grid,'counts':[selected.count(v) for v in grid],'mean_scores':np.mean(means,axis=0).tolist(),'fold_means':means,'warnings':sum(len(r['warnings']) for r in doc['rows'])}
            save_json(path,doc)
            print('COMPLETE',name,mode,doc['summary']['counts'],flush=True)


if __name__=='__main__': main()
