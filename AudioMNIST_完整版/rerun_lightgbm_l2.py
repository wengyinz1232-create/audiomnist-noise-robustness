"""Full, matched-grid LightGBM L2 rerun following saved inner diagnosis."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import concurrent.futures
import json
import pickle
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from threadpoolctl import ThreadpoolController
from common import ROOT, SEED, CONDITIONS, sha, save_json
from evaluation import run_configuration, grouped_folds, training_rows, scaling, transform, proba, model_for
from verify_saved_run import verify

RUN='lightgbm_l2_selected_20261011'
GRID=[0.,.1,1.]
OUT=ROOT/'results'/RUN


def setup():
    if os.name=='nt':ThreadpoolController._snapshot_loaded_modules=lambda self,kernel:None


def train(mode):
    setup()
    return run_configuration('mfcc_delta','lightgbm_goss_l2',mode,RUN,GRID)


def audit_models(path):
    doc=json.loads(path.read_text());mode=doc['config']['mode']
    with np.load(ROOT/'data'/'mfcc_delta.npz') as d:
        y,groups=d['y'],d['groups'];variants={c:d[c] for c in CONDITIONS}
    splits=list(grouped_folds(groups,np.arange(len(y)),10,SEED));checks=[]
    for row in [*doc['folds'],doc['full_fit']]:
        number=row['fold'];tr,te=splits[number-1] if number else (np.arange(len(y)),np.array([],dtype=int))
        assert row['grid']==GRID and row['selected']==GRID[int(np.argmax(np.mean(row['inner_scores'],axis=1)))]
        assert row['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in grouped_folds(groups,tr,3,SEED+number)]
        path_model=path.parent/'models'/f'fold_{number:02}.pkl';assert sha(path_model)==row['model_sha256']
        with path_model.open('rb') as f:b=pickle.load(f)
        assert b['estimator'].get_params()==model_for('lightgbm_goss_l2',row['selected'],SEED+number).get_params()
        assert b['train_people']==row['train_people']==np.unique(groups[tr]).tolist()
        assert b['source_signature']==sha(ROOT/'features.py') and np.array_equal(b['classes'],np.arange(10))
        xx,yy=training_rows(variants,y,tr,mode);s=scaling(xx)
        assert row['fit_rows']==len(yy) and all(np.array_equal(a,c) for a,c in zip(s,b['scaler']))
        delta=0.
        if number:
            with np.load(path.parent/f'predictions_{number:02}.npz') as z:
                for condition in CONDITIONS:
                    p=proba(b['estimator'],transform(variants[condition][te],s),b['classes'])
                    delta=max(delta,float(np.max(np.abs(p-z[condition]))));assert np.allclose(p,z[condition],atol=1e-10,rtol=1e-10)
                    cm=np.zeros((10,10),dtype=int);np.add.at(cm,(z['y'],z['classes'][z[condition].argmax(1)]),1)
                    assert cm.tolist()==row['metrics'][condition]['confusion']
                    den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(10),where=den!=0).mean()
                    assert abs(f1-row['metrics'][condition]['macro_f1'])<1e-12
        else:
            assert row['metrics']=={} and row['test_n']==0
            p=proba(b['estimator'],transform(variants['clean'][:8],s),b['classes'])
            assert p.shape==(8,10) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
        checks.append({'fold':number,'model_hash_and_parameters':'PASS','scaler':'PASS','max_prediction_difference':delta})
    return checks


def main():
    setup();OUT.mkdir(exist_ok=True)
    evidence=ROOT/'results'/'diagnosis_l2_20261011'/'chosen_grid.json'
    plan={'run_id':RUN,'grid':GRID,'model':'lightgbm_goss_l2','num_leaves':7,'n_estimators':70,
          'source':{p.name:sha(p) for p in [Path(__file__),ROOT/'evaluation.py',ROOT/'features.py',ROOT/'common.py',ROOT/'data'/'mfcc_delta.npz']},
          'diagnostic_selection_sha256':sha(evidence),'scope':'Full refit on common grids and original speaker splits after inner-data exploration; not independent confirmation.'}
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
    save_json(OUT/'plan.json',plan)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        paths=[Path(x) for x in pool.map(train,('clean_matched','augmented'))]
    audits={};docs={};diagnosis=json.loads(evidence.read_text())
    for path in paths:
        d=json.loads(path.read_text());m=d['config']['mode'];docs[m]=d
        a=verify(path);a['model_checks']=audit_models(path)
        assert a['selected_counts']==diagnosis['configurations'][m]['counts']
        for key,digest in d['config']['signature'].items():
            assert sha(ROOT/'data'/key if key.endswith('.npz') else ROOT/key)==digest
        audits[m]=a;save_json(OUT/f'{m}_verification.json',a)
        print('VERIFIED',m,a['selected_counts'],flush=True)
    metrics=[];comparisons=[];pairs=[]
    for m,d in docs.items():
        old=json.loads((ROOT/'runs'/'main'/f'mfcc_delta__lightgbm_goss__{m}'/'result.json').read_text())
        a={r['fold']:r for r in d['folds']};b={r['fold']:r for r in old['folds']}
        for f in range(1,11):assert a[f]['test_people']==b[f]['test_people']
        for condition in CONDITIONS:
            new=[a[f]['metrics'][condition]['macro_f1'] for f in range(1,11)];before=[b[f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
            diff=[x-y for x,y in zip(new,before)]
            metrics.append({'mode':m,'condition':condition,'macro_f1_mean':mean(new),'fold_sd':stdev(new)})
            comparisons.append({'mode':m,'condition':condition,'old_macro_f1_mean':mean(before),'new_macro_f1_mean':mean(new),'paired_change':mean(diff),'paired_change_sd':stdev(diff)})
    clean={r['fold']:r for r in docs['clean_matched']['folds']};aug={r['fold']:r for r in docs['augmented']['folds']}
    for f in range(1,11):
        assert clean[f]['test_people']==aug[f]['test_people'] and clean[f]['inner_valid_people']==aug[f]['inner_valid_people']
    for c in CONDITIONS:
        diff=[aug[f]['metrics'][c]['macro_f1']-clean[f]['metrics'][c]['macro_f1'] for f in range(1,11)]
        pairs.append({'condition':c,'augmented_minus_clean_mean':mean(diff),'paired_fold_sd':stdev(diff),'positive_folds':sum(x>0 for x in diff)})
    result={'status':'complete_verified_rerun','grid':GRID,'selected_counts':{m:a['selected_counts'] for m,a in audits.items()},
            'both_middle_unique':all(a['middle_unique_mode'] for a in audits.values()),'full_fit_selected':{m:d['full_fit']['selected'] for m,d in docs.items()},
            'warnings':{m:sum(len(r['warnings']) for r in [*d['folds'],d['full_fit']]) for m,d in docs.items()},
            'fits':220,'outer_models':20,'full_data_models':2,'metrics':metrics,'new_minus_original_configuration':comparisons,
            'matched_grid_augmentation_effects':pairs,'scope':plan['scope'],
            'limitations':'Fixed 7-leaf L2-tuned replacement, not the original leaf-tuned configuration. Original runs remain unchanged. Fold SD is not an independent confidence interval.'}
    save_json(OUT/'summary.json',result);print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
