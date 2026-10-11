"""Second bounded, inner-only diagnosis. Never reads outer-test scores."""
import itertools
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np
from threadpoolctl import ThreadpoolController, threadpool_limits
from common import ROOT, SEED, sha, save_json
from evaluation import grouped_folds, training_rows, scaling, transform, fit, model_for, proba, measure

OUT = ROOT/'results'/'diagnosis_round2_20261011'
PREV = ROOT/'results'/'diagnosis_20261011'
GRIDS = {'logistic':[.003,.01,.03,.1,.3,1.,3.,10.,30.], 'lightgbm_rounds':[10,40,160]}


def main():
    if os.name == 'nt':
        ThreadpoolController._snapshot_loaded_modules = lambda self, kernel: None
    OUT.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/'evaluation.py',ROOT/'features.py',ROOT/'common.py',ROOT/'data'/'mfcc_delta.npz']
    signature={p.name:sha(p) for p in paths}
    plan={'signature':signature,'grids':GRIDS,'lightgbm_fixed_num_leaves':7,
          'lightgbm_other_settings':'Unchanged model_for(lightgbm_goss,7,seed); only n_estimators varies.',
          'rationale':'Explore logarithmic neighborhoods of prior logistic preferences; vary boosting duration with a previously examined leaf cap, identical in both modes.',
          'scope':'Exploratory inner validation only; same ten outer-training partitions and three inner folds; no outer-test evaluation.',
          'objective':'mean(clean macro-F1, white_10 macro-F1)',
          'rounds_evaluation':'Train 160 rounds and predict prefixes 10/40/160. Verify prefix against separate 10-round fit in first inner fold of each mode.',
          'stop':'Complete these fixed grids, report failures as well as successes; no automatic extra grid search.'}
    planpath=OUT/'plan.json'
    if planpath.exists(): assert json.loads(planpath.read_text())==plan
    save_json(planpath,plan)
    with np.load(ROOT/'data'/'mfcc_delta.npz') as d:
        y,groups=d['y'],d['groups'];variants={c:d[c] for c in ('clean','white_10')}
    classes=np.unique(y);folds=list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    for name,grid in GRIDS.items():
        for mode in ('clean_matched','augmented'):
            path=OUT/f'{name}__{mode}.json'
            doc=json.loads(path.read_text()) if path.exists() else {'plan':plan,'rows':[],'prefix_checks':[]}
            assert doc['plan']==plan
            previous=json.loads((PREV/f'logistic__{mode}.json').read_text())
            assert previous['plan']['signature']['mfcc_delta.npz']==signature['mfcc_delta.npz']
            assert previous['plan']['signature']['evaluation.py']==signature['evaluation.py']
            reuse={(r['fold'],r['inner'],r['value']):r for r in previous['rows']} if name=='logistic' else {}
            reference=json.loads((ROOT/'runs'/'main'/f'mfcc_delta__logistic__{mode}'/'result.json').read_text())
            refs={r['fold']:r for r in reference['folds']}
            for number,(train,_) in enumerate(folds,1):
                inner=list(grouped_folds(groups,train,3,SEED+number))
                assert refs[number]['train_people']==np.unique(groups[train]).tolist()
                assert refs[number]['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in inner]
                for j,(tr,va) in enumerate(inner):
                    done={r['value'] for r in doc['rows'] if r['fold']==number and r['inner']==j}
                    pending=[v for v in grid if v not in done]
                    if not pending: continue
                    xx,yy=training_rows(variants,y,tr,mode);stats=scaling(xx);xx=transform(xx,stats)
                    xv=[transform(variants[c][va],stats) for c in ('clean','white_10')]
                    seed=SEED+100*number+j
                    start=time.perf_counter();prefix=None;notes=[]
                    if name=='lightgbm_rounds':
                        model=model_for('lightgbm_goss',7,seed).set_params(n_estimators=max(grid))
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            with threadpool_limits(limits=1,user_api='blas'): model.fit(xx,yy)
                        notes=[str(w.message) for w in caught]
                        assert np.array_equal(model.classes_,classes)
                        prefix={v:[model.booster_.predict(z,num_iteration=v) for z in xv] for v in grid}
                        if number==1 and j==0:
                            check=model_for('lightgbm_goss',7,seed).set_params(n_estimators=min(grid))
                            with threadpool_limits(limits=1,user_api='blas'): check.fit(xx,yy)
                            difference=max(float(np.max(np.abs(proba(check,z,classes)-p))) for z,p in zip(xv,prefix[min(grid)]))
                            assert difference<1e-12
                            doc['prefix_checks'].append({'fold':number,'inner':j,'rounds':min(grid),'max_probability_difference':difference})
                        doc.setdefault('fitted_models',[]).append({'fold':number,'inner':j,'rounds':max(grid),'seconds':time.perf_counter()-start})
                    for value in pending:
                        key=(number,j,value);condition_scores=None
                        if key in reuse:
                            score=reuse[key]['score'];notes=reuse[key]['warnings'];source='round1 saved inner score'
                        else:
                            if prefix is None:
                                model,notes=fit('logistic',value,xx,yy,seed)
                                probabilities=[proba(model,z,classes) for z in xv]
                            else: probabilities=prefix[value]
                            condition_scores=[measure(y[va],p,classes)['macro_f1'] for p in probabilities]
                            score=float(np.mean(condition_scores));source='new logistic fit' if prefix is None else 'new boosting prefix'
                        doc['rows'].append({'fold':number,'inner':j,'value':value,'score':score,'condition_scores':condition_scores,'source':source,'warnings':notes})
                    save_json(path,doc)
                    print(name,mode,'fold',number,'inner',j+1,flush=True)
            means=[[float(np.mean([r['score'] for r in doc['rows'] if r['fold']==f and r['value']==v])) for v in grid] for f in range(1,11)]
            selected=[grid[int(np.argmax(s))] for s in means]
            counts=[selected.count(v) for v in grid]
            doc['summary']={'grid':grid,'counts':counts,'fold_means':means,'mean_scores':np.mean(means,axis=0).tolist(),'warnings':sum(len(r['warnings']) for r in doc['rows'])}
            save_json(path,doc);print('COMPLETE',name,mode,counts,flush=True)
    summaries={}
    for name,grid in GRIDS.items():
        docs=[json.loads((OUT/f'{name}__{m}.json').read_text()) for m in ('clean_matched','augmented')]
        triples=[]
        for ids in itertools.combinations(range(len(grid)),3):
            counts=[]
            for d in docs:
                choices=[int(np.argmax([s[k] for k in ids])) for s in d['summary']['fold_means']]
                counts.append([choices.count(k) for k in range(3)])
            triples.append({'grid':[grid[k] for k in ids],'clean_counts':counts[0],'augmented_counts':counts[1],'both_middle_unique':all(c[1]>max(c[0],c[2]) for c in counts)})
        summaries[name]={'configurations':[d['summary'] for d in docs],'triples':triples}
    save_json(OUT/'summary.json',{'scope':'Post-hoc diagnostic triple comparisons; not independent validation or formal outer results.','models':summaries})


if __name__=='__main__': main()
