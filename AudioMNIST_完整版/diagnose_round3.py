"""Bounded boosting-duration diagnostic, six workers, no outer-test scoring."""
import concurrent.futures
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np
from threadpoolctl import ThreadpoolController, threadpool_limits
from common import ROOT, SEED, sha, save_json
from evaluation import grouped_folds, training_rows, scaling, transform, model_for, measure

OUT=ROOT/'results'/'diagnosis_round3_20261011'
GRID=[160,480,1440]
STATE=None


def initialize():
    global STATE
    if os.name=='nt':
        ThreadpoolController._snapshot_loaded_modules=lambda self,kernel:None
    with np.load(ROOT/'data'/'mfcc_delta.npz') as d:
        y,groups=d['y'],d['groups'];variants={c:d[c] for c in ('clean','white_10')}
    STATE=(y,groups,variants,list(grouped_folds(groups,np.arange(len(y)),10,SEED)))


def run_task(task):
    mode,number,j=task
    y,groups,variants,folds=STATE
    train,_=folds[number-1]
    inner=list(grouped_folds(groups,train,3,SEED+number));tr,va=inner[j]
    xx,yy=training_rows(variants,y,tr,mode);stats=scaling(xx);xx=transform(xx,stats)
    xv=[transform(variants[c][va],stats) for c in ('clean','white_10')]
    model=model_for('lightgbm_goss',7,SEED+100*number+j).set_params(n_estimators=max(GRID))
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=1,user_api='blas'):model.fit(xx,yy)
    classes=np.unique(y);assert np.array_equal(classes,model.classes_)
    rows=[]
    for value in GRID:
        scores=[measure(y[va],model.booster_.predict(z,num_iteration=value),classes)['macro_f1'] for z in xv]
        rows.append({'fold':number,'inner':j,'value':value,'condition_scores':scores,'score':float(np.mean(scores))})
    return {'mode':mode,'fold':number,'inner':j,'rows':rows,'warnings':[str(w.message) for w in caught],
            'seconds':time.perf_counter()-start,'actual_iterations':model.booster_.current_iteration(),
            'train_people':np.unique(groups[tr]).tolist(),'valid_people':np.unique(groups[va]).tolist()}


def main():
    OUT.mkdir(exist_ok=True)
    modes=('clean_matched','augmented')
    paths=[Path(__file__),ROOT/'evaluation.py',ROOT/'features.py',ROOT/'common.py',ROOT/'data'/'mfcc_delta.npz']
    previous={m:ROOT/'results'/'diagnosis_round2_20261011'/f'lightgbm_rounds__{m}.json' for m in modes}
    plan={'grid':GRID,'fixed_num_leaves':7,'signature':{p.name:sha(p) for p in paths},
          'reference_sha256':{m:sha(p) for m,p in previous.items()},'workers':6,
          'objective':'mean(clean macro-F1, white_10 macro-F1)',
          'scope':'Previously explored speaker splits, inner only; no outer-test predictions. No early stopping, fixed 1440 upper cap.',
          'method':'Fit 1440 rounds, evaluate prefixes 160/480/1440; compare every 160-round inner score with prior direct 160-round training.',
          'stop':'Exactly ten outer training partitions times three inner folds for both modes. No automatic expansion.'}
    if (OUT/'plan.json').exists():
        oldplan=json.loads((OUT/'plan.json').read_text())
        if oldplan!=plan:
            # Only scheduling/source revision may change; numerical settings and inputs must match.
            a=json.loads(json.dumps(oldplan));b=json.loads(json.dumps(plan))
            a.pop('workers');b.pop('workers')
            a['signature'].pop(Path(__file__).name);b['signature'].pop(Path(__file__).name)
            assert a==b
            save_json(OUT/'plan_before_worker_change.json',oldplan)
            for mode in modes:
                p=OUT/f'{mode}.json'
                if p.exists():
                    d=json.loads(p.read_text());assert d['plan']==oldplan
                    d['plan']=plan;save_json(p,d)
    save_json(OUT/'plan.json',plan)
    docs={};refs={}
    for mode in modes:
        p=OUT/f'{mode}.json'
        docs[mode]=json.loads(p.read_text()) if p.exists() else {'plan':plan,'tasks':[]}
        assert docs[mode]['plan']==plan
        old=json.loads(previous[mode].read_text())
        for key in ('evaluation.py','features.py','common.py','mfcc_delta.npz'):
            assert old['plan']['signature'][key]==plan['signature'][key]
        refs[mode]={(r['fold'],r['inner']):r for r in old['rows'] if r['value']==160}
    done={(m,t['fold'],t['inner']) for m in modes for t in docs[m]['tasks']}
    tasks=[(m,f,j) for f in range(1,11) for j in range(3) for m in modes if (m,f,j) not in done]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6,initializer=initialize) as pool:
        pending={pool.submit(run_task,t):t for t in tasks}
        for future in concurrent.futures.as_completed(pending):
            result=future.result();mode=result['mode']
            reference=refs[mode][result['fold'],result['inner']]
            delta=max(abs(a-b) for a,b in zip(result['rows'][0]['condition_scores'],reference['condition_scores']))
            assert delta<1e-12, ('Prefix mismatch',mode,result['fold'],result['inner'],delta)
            result['prefix_160_max_metric_difference']=delta
            docs[mode]['tasks'].append(result)
            save_json(OUT/f'{mode}.json',docs[mode])
            print('DONE',mode,'fold',result['fold'],'inner',result['inner']+1,'seconds',round(result['seconds'],1),flush=True)
    summary={}
    for mode,d in docs.items():
        assert len(d['tasks'])==30 and len({(t['fold'],t['inner']) for t in d['tasks']})==30
        rows=[r for t in d['tasks'] for r in t['rows']]
        means=[[float(np.mean([r['score'] for r in rows if r['fold']==f and r['value']==v])) for v in GRID] for f in range(1,11)]
        choices=[int(np.argmax(s)) for s in means];counts=[choices.count(i) for i in range(3)]
        summary[mode]={'grid':GRID,'counts':counts,'fold_means':means,'mean_scores':np.mean(means,axis=0).tolist(),
                       'middle_unique':counts[1]>max(counts[0],counts[2]),'warnings':sum(len(t['warnings']) for t in d['tasks']),
                       'prefix_checks':len(d['tasks']),'actual_iteration_range':[min(t['actual_iterations'] for t in d['tasks']),max(t['actual_iterations'] for t in d['tasks'])]}
    save_json(OUT/'summary.json',{'scope':plan['scope'],'configurations':summary,'new_fits':60,'outer_test_evaluations':0})
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
