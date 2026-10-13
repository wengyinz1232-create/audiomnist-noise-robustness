"""Manual speaker-disjoint nested RF tuning; fixed, staged search and paired baseline."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import copy
import hashlib
import itertools
import json
import pickle
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import evaluation as ev
from common import SEED, CONDITIONS

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)

def params(depth, leaf=2, features='sqrt', trees=60):
    return dict(max_depth=depth, min_samples_leaf=leaf, max_features=features, n_estimators=trees)

BASELINE = [params(d) for d in [4, 48, 128]]
STRUCTURES = BASELINE + [params(d, leaf) for d in [8, 16, None] for leaf in [1, 2, 5]]

def key(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]

def fit(p, x, y, seed):
    t = time.perf_counter()
    m = RandomForestClassifier(**p, bootstrap=True, criterion='gini',
                               min_samples_split=2, n_jobs=2, random_state=seed)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        m.fit(x, y)
    return m, time.perf_counter() - t, [str(w.message) for w in ws]

def score(m, valid, y):
    return float(np.mean([ev.measure(y, ev.proba(m, x, np.arange(10)), np.arange(10))['macro_f1'] for x in valid]))

def pick(candidates, cache):
    # Candidate order is fixed. Exact ties use the first entry (incumbent in B/C).
    return candidates[int(np.argmax([np.mean([r['score'] for r in cache[key(p)]['inner']]) for p in candidates]))]

def select_stages(prepared, fold, folder):
    cache = {}
    def evaluate(p, prefix=None):
        pth = folder / 'candidates' / (key(p) + '.json')
        if pth.exists():
            row = json.loads(pth.read_text(encoding='utf-8'))
            assert row['params'] == p and len(row['inner']) == 3
            cache[key(p)] = row
            if prefix is None or (folder / 'candidates' / (key(prefix) + '.json')).exists():
                if prefix is not None:
                    cache[key(prefix)] = json.loads((folder / 'candidates' / (key(prefix) + '.json')).read_text(encoding='utf-8'))
                return
        rows, pref = [], []
        for j, (x, yy, valid, vy) in enumerate(prepared):
            model, seconds, ws = fit(p, x, yy, SEED + 100 * fold + j)
            depths = [int(t.get_depth()) for t in model.estimators_]
            leaves = [int(t.get_n_leaves()) for t in model.estimators_]
            rows.append(dict(score=score(model, valid, vy), fit_seconds=seconds, warnings=ws,
                             actual_depth_mean=float(np.mean(depths)), actual_depth_max=max(depths),
                             leaves_mean=float(np.mean(leaves))))
            if prefix is not None:
                short = copy.copy(model)
                short.estimators_ = model.estimators_[:prefix['n_estimators']]
                short.n_estimators = prefix['n_estimators']
                pref.append(dict(score=score(short, valid, vy), fit_seconds=0,
                                 shared_fit=key(p), warnings=ws))
        row = dict(params=p, inner=rows)
        save(pth, row)
        cache[key(p)] = row
        if prefix is not None:
            row = dict(params=prefix, inner=pref)
            save(folder / 'candidates' / (key(prefix) + '.json'), row)
            cache[key(prefix)] = row
    for p in STRUCTURES:
        evaluate(p)
    baseline = pick(BASELINE, cache)
    a = pick(STRUCTURES, cache)
    print('STAGE_A', folder.parent.name, fold, json.dumps(a), flush=True)
    feature_candidates = [a] + [dict(a, max_features=f) for f in [0.1, 0.3]]
    for p in feature_candidates[1:]:
        evaluate(p)
    b = pick(feature_candidates, cache)
    print('STAGE_B', folder.parent.name, fold, json.dumps(b), flush=True)
    c120, c240 = dict(b, n_estimators=120), dict(b, n_estimators=240)
    evaluate(c240, prefix=c120)
    selected = pick([b, c120, c240], cache)
    assert np.mean([r['score'] for r in cache[key(selected)]['inner']]) >= np.mean([r['score'] for r in cache[key(baseline)]['inner']]) - 1e-15
    return baseline, selected, dict(A=a, B=b, C=selected), cache

def run_fold(cache_dir, output, rep, mode, number):
    started = time.perf_counter()
    output, cache_dir = Path(output), Path(cache_dir)
    folder = output / f'{rep}__forest__{mode}' / f'fold_{number:02}'
    result = folder / 'result.json'
    if result.exists():
        row = json.loads(result.read_text(encoding='utf-8'))
        for k, model in row['models'].items():
            assert sha(folder / f'{k}.pkl') == model['model_sha256']
            assert sha(folder / f'{k}_predictions.npz') == model['predictions_sha256']
        return row
    with np.load(cache_dir / f'{rep}.npz') as z:
        y, groups = z['y'], z['groups']
        # Test-only conditions aren't loaded until both parameter choices are frozen.
        variants = {c: z[c] for c in ['clean', 'white_10']}
    splits = list(ev.grouped_folds(groups, np.arange(len(y)), 10, SEED))
    train, test = splits[number - 1]
    inner = list(ev.grouped_folds(groups, train, 3, SEED + number))
    prepared = []
    for fit_idx, valid_idx in inner:
        assert not set(groups[fit_idx]) & set(groups[valid_idx])
        assert not set(groups[fit_idx]) & set(groups[test])
        x, yy = ev.training_rows(variants, y, fit_idx, mode)
        stats = ev.scaling(x)
        prepared.append((ev.transform(x, stats), yy,
                         [ev.transform(variants[c][valid_idx], stats) for c in ['clean', 'white_10']], y[valid_idx]))
    print('START', rep, mode, number, flush=True)
    baseline, selected, stages, scores = select_stages(prepared, number, folder)
    frozen = dict(baseline=baseline, tuned=selected, stages=stages)
    save(folder / 'selection.json', frozen)
    del prepared
    selection_seconds = time.perf_counter() - started
    with np.load(cache_dir / f'{rep}.npz') as z:
        variants.update({c: z[c] for c in ['white_20', 'pink_10']})
    x, yy = ev.training_rows(variants, y, train, mode)
    stats = ev.scaling(x)
    x = ev.transform(x, stats)
    models = {}
    for label, p in [('baseline', baseline), ('tuned', selected)]:
        model, seconds, ws = fit(p, x, yy, SEED + number)
        bundle = dict(estimator=model, scaler=stats, classes=np.arange(10), representation=rep,
                      mode=mode, params=p, train_people=np.unique(groups[train]).tolist())
        path = folder / f'{label}.pkl'
        with path.open('wb') as f:
            pickle.dump(bundle, f, protocol=5)
        with path.open('rb') as f:
            restored = pickle.load(f)
        probabilities = {}
        for c in CONDITIONS:
            xt = ev.transform(variants[c][test], stats)
            probabilities[c] = ev.proba(model, xt, np.arange(10))
            reloaded = ev.proba(restored['estimator'], ev.transform(variants[c][test], restored['scaler']), np.arange(10))
            np.testing.assert_allclose(probabilities[c], reloaded, atol=1e-12, rtol=1e-12)
        prediction_path = folder / f'{label}_predictions.npz'
        np.savez_compressed(prediction_path, indices=test, y=y[test], groups=groups[test], classes=np.arange(10), **probabilities)
        models[label] = dict(params=p, metrics={c: ev.measure(y[test], probabilities[c], np.arange(10)) for c in CONDITIONS},
                             inner_mean=float(np.mean([r['score'] for r in scores[key(p)]['inner']])),
                             fit_seconds=seconds, warnings=ws, actual_depths=[int(t.get_depth()) for t in model.estimators_],
                             model_sha256=sha(path), predictions_sha256=sha(prediction_path), reload_verified=True)
    row = dict(representation=rep, mode=mode, fold=number, train_people=np.unique(groups[train]).tolist(),
               test_people=np.unique(groups[test]).tolist(),
               inner_valid_people=[np.unique(groups[v]).tolist() for _, v in inner],
               stages=stages, models=models, selection_seconds=selection_seconds,
               seconds=time.perf_counter()-started, actual_inner_fits=45, outer_fits=2)
    save(result, row)
    print('DONE', rep, mode, number, 'seconds', round(row['seconds'], 1), flush=True)
    return row

def protocol(cache_dir):
    inv = json.loads((cache_dir / 'inventory.json').read_text(encoding='utf-8'))
    for f, k in [('features.py','frontend_sha256'), ('prepare_data.py','prepare_sha256')]:
        assert sha(ROOT/'source'/f) == inv['signature'][k]
    for rep in ['logmel','mfcc_delta']:
        assert sha(cache_dir/f'{rep}.npz') == inv['cache_sha256'][rep]
        with np.load(cache_dir/f'{rep}.npz') as z:
            assert len(z['y']) == len(set(z['ids'])) == 12000
            assert len(np.unique(z['groups'])) == 60
            assert np.array_equal(np.bincount(z['y']),np.repeat(1200,10))
    return dict(protocol='rf-staged-hyperparameters-v1', seed=SEED, outer_folds=10, inner_folds=3,
                configurations=list(itertools.product(['logmel','mfcc_delta'],['clean_matched','augmented'])),
                baseline=BASELINE, stage_A=STRUCTURES, stage_B_features=['incumbent sqrt',0.1,0.3],
                stage_C_trees=[60,120,240], candidate_count_per_fold=16,
                fit_calls_per_fold=dict(inner=45,outer=2),
                selection='maximum mean of three inner-fold mean(clean Macro-F1, white_10 Macro-F1)',
                tie_rule='exact argmax, first candidate; incumbent first in stages B/C',
                fixed=dict(bootstrap=True,criterion='gini',min_samples_split=2,n_jobs=2),
                feature_sha256=inv['cache_sha256'], python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                sources={p.name:sha(p) for p in (ROOT/'source').glob('*.py')}, runner_sha256=sha(__file__),
                limits=['Fixed staged search, not an exhaustive joint search; interactions may be missed.',
                        'Same previously explored speakers; exploratory comparison, not independent confirmation.',
                        'No midpoint requirement; no test-score-driven grid changes.',
                        'No full-data deployment model; all saved models are outer-fold evaluation models.',
                        'Fold SD describes variability, not an independent confidence interval.'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT/'results')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--fold', type=int, choices=range(1,11))
    ap.add_argument('--representation', choices=['logmel','mfcc_delta'])
    ap.add_argument('--mode', choices=['clean_matched','augmented'])
    a=ap.parse_args()
    out=a.output.resolve(); cache=a.cache_dir.resolve()
    if out == cache or cache in out.parents or out == ROOT/'source' or ROOT/'source' in out.parents:
        raise ValueError('Output must be separate from input cache and source.')
    plan=protocol(cache)
    pp=out/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text(encoding='utf-8')) == plan, 'Protocol changed; choose a fresh output.'
    else:
        save(pp,plan)
    jobs=[(str(cache),str(out),r,m,f) for r,m in plan['configurations'] for f in range(1,11)
          if (not a.fold or f==a.fold) and (not a.representation or r==a.representation) and (not a.mode or m==a.mode)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_fold,*j):j for j in jobs}
        for future in as_completed(futures):
            row=future.result()
            print('PROGRESS',row['representation'],row['mode'],row['fold'],flush=True)
    print('ALL REQUESTED FOLDS COMPLETE',flush=True)

if __name__=='__main__':
    main()
