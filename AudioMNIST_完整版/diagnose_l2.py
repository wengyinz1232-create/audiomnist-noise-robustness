"""Predeclared L2 diagnosis. Inner validation only; checkpoint each fit group."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import ThreadpoolController
from common import ROOT, SEED, sha, save_json
from evaluation import grouped_folds, training_rows, scaling, transform, fit, proba, measure, model_for

OUT=ROOT/'results'/'diagnosis_l2_20261011'
GRID=[0.,.1,1.,10.,100.]
STATE=None


def initialize():
    global STATE
    if os.name=='nt':ThreadpoolController._snapshot_loaded_modules=lambda self,kernel:None
    with np.load(ROOT/'data'/'mfcc_delta.npz') as d:
        y,groups=d['y'],d['groups'];variants={c:d[c] for c in ('clean','white_10')}
    STATE=(y,groups,variants,list(grouped_folds(groups,np.arange(len(y)),10,SEED)))


def task(args):
    mode,number,j=args;y,groups,variants,outer=STATE
    train,_=outer[number-1];tr,va=list(grouped_folds(groups,train,3,SEED+number))[j]
    xx,yy=training_rows(variants,y,tr,mode);stats=scaling(xx);xx=transform(xx,stats)
    xv=[transform(variants[c][va],stats) for c in ('clean','white_10')];classes=np.unique(y)
    rows=[];start=time.perf_counter()
    for value in GRID:
        model,w=fit('lightgbm_goss_l2',value,xx,yy,SEED+100*number+j)
        scores=[measure(y[va],proba(model,z,classes),classes)['macro_f1'] for z in xv]
        rows.append({'value':value,'score':float(np.mean(scores)),'condition_scores':scores,'warnings':w})
    return {'mode':mode,'fold':number,'inner':j,'rows':rows,'train_people':np.unique(groups[tr]).tolist(),
            'valid_people':np.unique(groups[va]).tolist(),'seconds':time.perf_counter()-start}


def main():
    OUT.mkdir(exist_ok=True)
    signature={p.name:sha(p) for p in [Path(__file__),ROOT/'evaluation.py',ROOT/'common.py',ROOT/'features.py',ROOT/'data'/'mfcc_delta.npz']}
    plan={'grid':GRID,'model':'lightgbm_goss_l2','fixed_parameters':model_for('lightgbm_goss_l2',1.,SEED).get_params(),
          'signature':signature,'workers':4,'objective':'mean(clean macro-F1, white_10 macro-F1)',
          'rationale':'Fix the previously examined 7-leaf/70-round configuration; vary leaf-weight L2 penalty from none through strong regularization.',
          'scope':'Ten outer-training partitions, three inner folds, both modes; no outer-test evaluation.',
          'decision_rule':'Inspect complete-grid selection counts. A follow-up three-value grid must bracket a supported optimum; do not discard stronger candidates merely to force a middle-mode pass.',
          'stop':'Complete this declared five-value diagnostic before deciding any follow-up.'}
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
    save_json(OUT/'plan.json',plan)
    path=OUT/'scores.json';doc=json.loads(path.read_text()) if path.exists() else {'plan':plan,'tasks':[]}
    assert doc['plan']==plan
    done={(t['mode'],t['fold'],t['inner']) for t in doc['tasks']}
    args=[(m,f,j) for f in range(1,11) for j in range(3) for m in ('clean_matched','augmented') if (m,f,j) not in done]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        for future in concurrent.futures.as_completed([pool.submit(task,a) for a in args]):
            result=future.result();doc['tasks'].append(result);save_json(path,doc)
            print('DONE',result['mode'],result['fold'],result['inner']+1,'seconds',round(result['seconds'],1),flush=True)
    summary={}
    for mode in ('clean_matched','augmented'):
        ts=[t for t in doc['tasks'] if t['mode']==mode];assert len(ts)==30
        assert len({(t['fold'],t['inner']) for t in ts})==30
        means=[[float(np.mean([t['rows'][k]['score'] for t in ts if t['fold']==f])) for k in range(len(GRID))] for f in range(1,11)]
        choices=[int(np.argmax(s)) for s in means]
        counts=[choices.count(k) for k in range(len(GRID))]
        summary[mode]={'grid':GRID,'counts':counts,'fold_means':means,'mean_scores':np.mean(means,axis=0).tolist(),
                       'warnings':sum(len(r['warnings']) for t in ts for r in t['rows'])}
    save_json(OUT/'summary.json',{'status':'complete_inner_diagnostic','configurations':summary,'new_fits':300,'outer_test_evaluations':0})
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
