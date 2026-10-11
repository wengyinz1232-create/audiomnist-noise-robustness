"""Re-fit both MFCC logistic pipelines on documented, mode-specific grids.

Run from this directory with installed requirements:
    python rerun_logistic_selected_grids.py
Original runs/main results are never overwritten. Grids were explored on the
same data, so these results are not an independent confirmation experiment.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json
import pickle
import platform
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from threadpoolctl import ThreadpoolController
from common import ROOT, SEED, CONDITIONS, sha, save_json
from evaluation import run_configuration, grouped_folds, training_rows, scaling, transform, proba
from verify_saved_run import verify

RUN_ID='logistic_selected_grids_20261011'
GRIDS={'clean_matched':[.0001,.003,.01],'augmented':[.01,.03,.1]}
OUT=ROOT/'results'/RUN_ID


def check_models(path):
    doc=json.loads(path.read_text(encoding='utf-8'));mode=doc['config']['mode']
    with np.load(ROOT/'data'/'mfcc_delta.npz') as cache:
        y,groups=cache['y'],cache['groups'];variants={c:cache[c] for c in CONDITIONS}
    splits=list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    checks=[]
    for row in [*doc['folds'],doc['full_fit']]:
        number=row['fold'];train,test=splits[number-1] if number else (np.arange(len(y)),np.array([],dtype=int))
        assert row['grid']==doc['config']['grid']
        means=np.mean(row['inner_scores'],axis=1)
        assert np.allclose(means,row['inner_means'],rtol=0,atol=1e-15)
        assert row['selected']==row['grid'][int(np.argmax(means))]
        assert row['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in grouped_folds(groups,train,3,SEED+number)]
        assert row['train_n']==len(train) and row['test_n']==len(test)
        if not number: assert row['metrics']=={} and row['test_people']==[]
        modelpath=path.parent/'models'/f'fold_{number:02}.pkl'
        assert sha(modelpath)==row['model_sha256']
        with modelpath.open('rb') as stream:bundle=pickle.load(stream)
        assert bundle['mode']==mode and bundle['representation']=='mfcc_delta'
        assert bundle['source_signature']==sha(ROOT/'features.py')
        assert bundle['selected']==row['selected']==bundle['estimator'].get_params()['C']
        assert bundle['train_people']==np.unique(groups[train]).tolist()==row['train_people']
        assert np.array_equal(bundle['classes'],np.arange(10))
        xx,yy=training_rows(variants,y,train,mode);stats=scaling(xx)
        assert row['fit_rows']==len(yy)
        assert all(np.array_equal(a,b) for a,b in zip(stats,bundle['scaler']))
        delta=0.
        if number:
            with np.load(path.parent/f'predictions_{number:02}.npz') as stored:
                for condition in CONDITIONS:
                    actual=proba(bundle['estimator'],transform(variants[condition][test],stats),bundle['classes'])
                    assert (actual>=0).all() and np.allclose(actual.sum(1),1)
                    delta=max(delta,float(np.max(np.abs(actual-stored[condition]))))
                    assert np.allclose(actual,stored[condition],atol=1e-10,rtol=1e-10)
        checks.append({'fold':number,'model_hash_verified':True,'scaler_matches_training_rows':True,'max_prediction_difference':delta})
    return checks


def main():
    if os.name=='nt':ThreadpoolController._snapshot_loaded_modules=lambda self,kernel:None
    OUT.mkdir(exist_ok=True)
    plan={'run_id':RUN_ID,'representation':'mfcc_delta','model':'logistic','grids':GRIDS,
          'seed':SEED,'python':platform.python_version(),
          'signature':{p.name:sha(p) for p in [Path(__file__),ROOT/'evaluation.py',ROOT/'features.py',ROOT/'common.py',ROOT/'data'/'mfcc_delta.npz']},
          'selection_evidence_sha256':sha(ROOT/'results'/'diagnosis_round2_20261011'/'logistic_mode_specific_option.json'),
          'scope':'Full 10x3 nested re-fit plus separately tuned full-data models; exploratory grids from previously observed data.',
          'limitation':'Mode-specific search ranges: comparisons are between separately tuned pipelines, not an identical-grid augmentation-only intervention.'}
    if (OUT/'plan.json').exists():assert json.loads((OUT/'plan.json').read_text())==plan
    save_json(OUT/'plan.json',plan)
    audits={};comparison=[];newdocs={}
    for mode,grid in GRIDS.items():
        path=Path(run_configuration('mfcc_delta','logistic',mode,RUN_ID,grid))
        audit=verify(path);audit['model_checks']=check_models(path)
        audits[mode]=audit
        new=json.loads(path.read_text());newdocs[mode]=new
        oldpath=ROOT/'runs'/'main'/f'mfcc_delta__logistic__{mode}'/'result.json'
        old=json.loads(oldpath.read_text());oldrows={r['fold']:r for r in old['folds']};newrows={r['fold']:r for r in new['folds']}
        for f in range(1,11):
            assert oldrows[f]['test_people']==newrows[f]['test_people']
            assert oldrows[f]['inner_valid_people']==newrows[f]['inner_valid_people']
        for condition in CONDITIONS:
            before=[oldrows[f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
            after=[newrows[f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
            diff=[a-b for a,b in zip(after,before)]
            comparison.append({'mode':mode,'condition':condition,'old_macro_f1':mean(before),'new_macro_f1':mean(after),'new_fold_sd':stdev(after),'paired_difference_mean':mean(diff),'paired_difference_sd':stdev(diff),'positive_folds':sum(d>0 for d in diff),'negative_folds':sum(d<0 for d in diff)})
        save_json(OUT/f'{mode}_verification.json',audit)
        print('VERIFIED',mode,audit['selected_counts'],'middle_unique',audit['middle_unique_mode'],flush=True)
    save_json(OUT/'comparison.json',{'scope':plan['scope'],'limitations':[plan['limitation'],'Standard deviations describe overlapping CV folds; not independent confidence intervals.'],'rows':comparison})
    # Preserve LightGBM roles explicitly, without asserting that unlike configurations isolate augmentation.
    lightgbm={}
    for label,path in {
        'clean_alternative':ROOT/'runs'/'alt_min_child'/'mfcc_delta__lightgbm_goss_min_child__clean_matched'/'result.json',
        'augmented_original':ROOT/'runs'/'main'/'mfcc_delta__lightgbm_goss__augmented'/'result.json'}.items():
        audit=verify(path)
        lightgbm[label]={'path':str(path.relative_to(ROOT)),'saved_evidence_audit':audit,
                        'role':'Previously completed alternative pipeline; not a matched augmentation-only pair.'}
    summary={'run_id':RUN_ID,'completed_configurations':2,'outer_folds':20,'full_data_models':2,'fits_in_complete_run':220,
             'selected_counts':{m:a['selected_counts'] for m,a in audits.items()},
             'both_middle_unique':all(a['middle_unique_mode'] for a in audits.values()),
             'warnings':{m:sum(len(r['warnings']) for r in [*d['folds'],d['full_fit']]) for m,d in newdocs.items()},
             'full_fit_selected':{m:d['full_fit']['selected'] for m,d in newdocs.items()},
             'scope':plan['scope'],'limitations':plan['limitation'],'comparison':comparison,'lightgbm_existing_options':lightgbm}
    save_json(OUT/'summary.json',summary)
    print('COMPLETE',json.dumps({'counts':summary['selected_counts'],'both_middle_unique':summary['both_middle_unique'],'warnings':summary['warnings']}),flush=True)


if __name__=='__main__':main()
