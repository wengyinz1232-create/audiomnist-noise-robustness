"""Joint LightGBM search: one verified outer fold per cloud job, then aggregate."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import itertools
import json
import pickle
import platform
import statistics
from pathlib import Path
import numpy as np
import reproduce

PLAN_PATH = Path(__file__).with_name('lightgbm_search.json')
PLAN = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
PARAMETERS = tuple(PLAN['search'])
GRID = [dict(zip(PARAMETERS, values)) for values in itertools.product(*PLAN['search'].values())]
KEYS = [f'{rep}__lightgbm_goss_joint__{mode}' for rep in ('logmel','mfcc_delta')
        for mode in ('clean_matched','augmented')]


def environment():
    return {'python': platform.python_version(), 'packages': {
        p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','lightgbm','threadpoolctl')}}


def protocol(work, key):
    rep, _, mode = key.split('__')
    return {'key': key, 'representation': rep, 'mode': mode, 'grid': GRID, 'fixed': PLAN['fixed'],
            'outer_folds': 10, 'inner_folds': 3, 'seed': 20261009,
            'selection_objective': 'mean(clean macro-F1, white_10 macro-F1)',
            'source_sha256': {n: reproduce.sha(work/n) for n in reproduce.CORE},
            'runner_sha256': reproduce.sha(Path(__file__)), 'plan_sha256': reproduce.sha(PLAN_PATH),
            'feature_sha256': reproduce.sha(work/'data'/f'{rep}.npz'), 'environment': environment()}


def verify_fold(work, folder, config, row, variants, y, groups):
    """Recompute metrics and predictions from the saved model, not reported scores."""
    import evaluation as ev
    assert config['seed'] == ev.SEED
    number = row['fold']
    assert number in range(1,11)
    tr, te = list(ev.grouped_folds(groups,np.arange(len(y)),10,ev.SEED))[number-1]
    inner = list(ev.grouped_folds(groups,tr,3,ev.SEED+number))
    assert row['train_people'] == np.unique(groups[tr]).tolist()
    assert row['test_people'] == np.unique(groups[te]).tolist()
    assert not set(row['train_people']) & set(row['test_people'])
    assert row['inner_valid_people'] == [np.unique(groups[v]).tolist() for _,v in inner]
    scores = np.asarray(row['inner_scores'])
    assert scores.shape == (len(GRID),3) and np.isfinite(scores).all()
    assert row['grid'] == GRID
    np.testing.assert_allclose(row['inner_means'],scores.mean(1),rtol=0,atol=1e-12)
    assert row['selected'] == GRID[int(np.argmax(scores.mean(1)))]
    model_path = folder/'models'/f'fold_{number:02}.pkl'
    assert reproduce.sha(model_path) == row['model_sha256']
    with model_path.open('rb') as stream: bundle=pickle.load(stream)
    expected=ev.model_for('lightgbm_goss_joint',row['selected'],ev.SEED+number)
    assert bundle['estimator'].get_params() == expected.get_params()
    assert all(expected.get_params()[k] == v for k,v in PLAN['fixed'].items())
    assert bundle['selected'] == row['selected']
    assert bundle['train_people'] == row['train_people']
    assert bundle['representation'] == config['representation'] and bundle['mode'] == config['mode']
    assert bundle['source_signature'] == reproduce.sha(work/'features.py')
    xfit,yfit=ev.training_rows(variants,y,tr,config['mode'])
    stats=ev.scaling(xfit)
    assert all(np.array_equal(a,b) for a,b in zip(stats,bundle['scaler']))
    assert (row['train_n'],row['test_n'],row['fit_rows']) == (len(tr),len(te),len(yfit))
    prediction_path=folder/f'predictions_{number:02}.npz'
    with np.load(prediction_path) as saved:
        assert np.array_equal(saved['indices'],te) and np.array_equal(saved['y'],y[te])
        assert np.array_equal(saved['classes'],np.arange(10))
        assert np.array_equal(bundle['classes'],saved['classes'])
        for condition in ev.CONDITIONS:
            p=saved[condition]
            assert p.shape == (len(te),10) and np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-6)
            actual=ev.proba(bundle['estimator'],ev.transform(variants[condition][te],stats),bundle['classes'])
            np.testing.assert_allclose(actual,p,rtol=1e-5,atol=1e-6)
            assert ev.measure(y[te],p,saved['classes']) == row['metrics'][condition]
    return {'status':'PASS','model_sha256':reproduce.sha(model_path),
            'prediction_sha256':reproduce.sha(prediction_path),'conditions_verified':4,
            'selection_verified':True,'speaker_splits_verified':True,'model_predictions_verified':True}


def run_fold(work,key,number):
    import evaluation as ev
    reproduce.check_data(work)
    config=protocol(work,key)
    folder=work/'lightgbm_joint'/key/f'fold_{number:02}'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'result.json'
    with np.load(work/'data'/f'{config["representation"]}.npz') as cache:
        y,groups=cache['y'],cache['groups']
        variants={c:cache[c] for c in ev.CONDITIONS}
    if path.exists():
        doc=json.loads(path.read_text(encoding='utf-8'))
        assert doc['config'] == config,'Resume rejected: protocol/source/data/environment changed'
        row=doc['result']
    else:
        tr,te=list(ev.grouped_folds(groups,np.arange(len(y)),10,ev.SEED))[number-1]
        row=ev.train_fold(variants,y,groups,'lightgbm_goss_joint',config['mode'],GRID,number,tr,te,folder,config['representation'])
        reproduce.save(path,{'config':config,'result':row})
    audit=verify_fold(work,folder,config,row,variants,y,groups)
    audit['result_sha256']=reproduce.sha(path)
    reproduce.save(folder/'audit.json',audit)
    print('LIGHTGBM_FOLD_VERIFIED',key,number,row['selected'],row['seconds'],flush=True)


def selection_counts(rows):
    result={}
    for param,values in PLAN['search'].items():
        counts=[sum(r['selected'][param]==v for r in rows) for v in values]
        result[param]={'values':values,'counts':counts,'middle_unique_mode':counts[1]>max(counts[0],counts[2]),
                       'middle_strict_majority':counts[1]>len(rows)/2}
    return result


def summarize(work):
    import evaluation as ev
    reproduce.check_data(work)
    metrics=[]; parameters={}; results={}; hashes={}
    for key in KEYS:
        config=protocol(work,key); rows=[]
        with np.load(work/'data'/f'{config["representation"]}.npz') as cache:
            y,groups=cache['y'],cache['groups']
            variants={c:cache[c] for c in ev.CONDITIONS}
        coverage=np.zeros(len(y),dtype=int)
        for number in range(1,11):
            folder=work/'lightgbm_joint'/key/f'fold_{number:02}'
            path=folder/'result.json';doc=json.loads(path.read_text(encoding='utf-8'))
            audit=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
            assert doc['config']==config and doc['result']['fold']==number
            assert audit['status']=='PASS' and audit['result_sha256']==reproduce.sha(path)
            assert audit['prediction_sha256']==reproduce.sha(folder/f'predictions_{number:02}.npz')
            row=doc['result'];tr,te=list(ev.grouped_folds(groups,np.arange(len(y)),10,ev.SEED))[number-1]
            assert row['train_people']==np.unique(groups[tr]).tolist() and row['test_people']==np.unique(groups[te]).tolist()
            inner=list(ev.grouped_folds(groups,tr,3,ev.SEED+number))
            assert row['inner_valid_people']==[np.unique(groups[v]).tolist() for _,v in inner]
            scores=np.asarray(row['inner_scores'])
            assert scores.shape==(len(GRID),3) and np.isfinite(scores).all() and row['grid']==GRID
            assert row['selected']==GRID[int(np.argmax(scores.mean(1)))]
            with np.load(folder/f'predictions_{number:02}.npz') as saved:
                assert np.array_equal(saved['indices'],te) and np.array_equal(saved['y'],y[te])
                assert np.array_equal(saved['classes'],np.arange(10))
                for condition in ev.CONDITIONS:
                    p=saved[condition]
                    assert p.shape==(len(te),10) and np.isfinite(p).all() and (p>=0).all()
                    np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
                    assert ev.measure(y[te],p,saved['classes'])==row['metrics'][condition]
            coverage[te]+=1; rows.append(row);hashes[f'{key}/fold_{number:02}']=reproduce.sha(path)
        assert np.all(coverage==1)
        parameters[key]=selection_counts(rows);results[key]=rows
        for condition in ev.CONDITIONS:
            entry={'configuration':key,'condition':condition}
            for metric in reproduce.METRICS:
                values=[r['metrics'][condition][metric] for r in rows]
                entry[metric+'_mean']=statistics.mean(values);entry[metric+'_fold_sd']=statistics.stdev(values)
            metrics.append(entry)
    report=work/'reports/lightgbm_joint';report.mkdir(parents=True,exist_ok=True)
    midpoint_failures=[{'configuration':key,'parameter':param,'counts':audit['counts']}
        for key,params in parameters.items() for param,audit in params.items() if not audit['middle_strict_majority']]
    reproduce.save(report/'summary.json',{'status':'PASS','outer_models':40,'inner_fits':3240,'total_fits':3280,
        'midpoint_requirement_status':'PASS' if not midpoint_failures else 'NOT_MET',
        'midpoint_requirement_failures':midpoint_failures,
        'full_data_models':0,'plan':PLAN,'environment':environment(),'metrics':metrics,
        'parameter_selection':parameters,'fold_results':results,'result_sha256':hashes,
        'limits':['Previously explored dataset; not independent confirmation.',
                  'Midpoint success is observed, not forced; unique mode and strict majority are separate.',
                  'Fold standard deviations are not independent confidence intervals.']})
    with (report/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    lines=['# LightGBM joint search','', '40 outer models verified; 3,280 fits; no all-data fit.',
           f'Midpoint strict-majority target: {"PASS" if not midpoint_failures else "NOT_MET"}.','',
           '| Configuration | Parameter | Counts (low / middle / high) | Middle unique mode | Strict majority |',
           '|---|---|---|---|---|']
    for key,params in parameters.items():
        for param,audit in params.items():
            lines.append(f'| {key} | {param} | {audit["counts"]} | {audit["middle_unique_mode"]} | {audit["middle_strict_majority"]} |')
    lines+=['','## Macro-F1 (mean ± fold SD)','','| Configuration | Condition | Macro-F1 |','|---|---|---|']
    for row in metrics:
        lines.append(f'| {row["configuration"]} | {row["condition"]} | {row["macro_f1_mean"]:.5f} ± {row["macro_f1_fold_sd"]:.5f} |')
    (report/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('LIGHTGBM_JOINT_COMPLETE',report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['fold','summarize'])
    parser.add_argument('--work-dir',type=Path,default=Path('.lightgbm-joint'))
    parser.add_argument('--config',choices=KEYS)
    parser.add_argument('--fold',type=int,choices=range(1,11))
    args=parser.parse_args()
    if args.action=='fold' and (args.config is None or args.fold is None):
        parser.error('fold requires --config and --fold')
    work=reproduce.stage(args.work_dir)
    if args.action=='fold':run_fold(work,args.config,args.fold)
    else:summarize(work)


if __name__=='__main__':main()
