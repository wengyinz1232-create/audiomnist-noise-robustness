"""Auditable, capacity-controlled RF experiment; leaves legacy reproduction intact."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import reproduce

sys.path.insert(0, str(reproduce.SOURCE))
import evaluation as ev
from common import SEED, CONDITIONS

DEFAULT_GRID = [4, 8, 16]  # Training-only screen: active capacity limits, not a universal optimum.
GRID_ORIGIN = 'Three-inner-fold depth screen using outer-fold-1 training speakers; exploratory.'


def validate_grid(grid):
    if len(grid) < 3 or len(grid) % 2 != 1 or any(type(x) is not int or x < 1 for x in grid):
        raise ValueError('Use an odd number (at least three) of positive integer depths.')
    if grid != sorted(set(grid)):
        raise ValueError('Depths must be unique and increasing.')


def fitted(value, x, y, seed):
    start = time.perf_counter()
    model, warnings = ev.fit('forest', value, x, y, seed)
    depths = [int(t.get_depth()) for t in model.estimators_]
    leaves = [int(t.get_n_leaves()) for t in model.estimators_]
    return model, {'depth_limit': value, 'actual_depths': depths,
                   'leaf_counts': leaves, 'depth_limit_fraction': float(np.mean(np.array(depths) == value)),
                   'fit_seconds': time.perf_counter() - start, 'warnings': warnings}


def train_fold(variants, y, groups, rep, mode, grid, number, train, test, output):
    validate_grid(grid)
    start = time.perf_counter()
    classes = np.arange(10)
    inner = list(ev.grouped_folds(groups, train, 3, SEED + number))
    scores = np.zeros((len(grid), 3))
    diagnostics = [[] for _ in grid]
    # Prepare once per split. No outer test rows or scores enter selection.
    for j, (fit_idx, valid_idx) in enumerate(inner):
        x, yy = ev.training_rows(variants, y, fit_idx, mode)
        stats = ev.scaling(x)
        x = ev.transform(x, stats)
        valid = [ev.transform(variants[c][valid_idx], stats) for c in ('clean', 'white_10')]
        for i, depth in enumerate(grid):
            model, detail = fitted(depth, x, yy, SEED + 100 * number + j)
            scores[i, j] = np.mean([ev.measure(y[valid_idx], ev.proba(model, v, classes), classes)['macro_f1'] for v in valid])
            diagnostics[i].append(detail)
    means = scores.mean(axis=1)
    best = int(np.argmax(means))  # Exact ties prefer the smallest depth.
    chosen = grid[best]
    x, yy = ev.training_rows(variants, y, train, mode)
    stats = ev.scaling(x)
    model, final = fitted(chosen, ev.transform(x, stats), yy, SEED + number)
    bundle = dict(estimator=model, scaler=stats, classes=classes, representation=rep,
                  mode=mode, selected=chosen, train_people=np.unique(groups[train]).tolist())
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / 'model.pkl'
    with model_path.open('wb') as stream:
        pickle.dump(bundle, stream, protocol=5)
    with model_path.open('rb') as stream:
        restored = pickle.load(stream)  # Artifact just created by this process.
    probabilities = {}
    for c in CONDITIONS:
        xt = ev.transform(variants[c][test], stats)
        probabilities[c] = ev.proba(model, xt, classes)
        reloaded = ev.proba(restored['estimator'], ev.transform(variants[c][test], restored['scaler']), classes)
        np.testing.assert_allclose(probabilities[c], reloaded, atol=1e-12, rtol=1e-12)
    pred_path = output / 'predictions.npz'
    np.savez_compressed(pred_path, indices=test, y=y[test], groups=groups[test], classes=classes, **probabilities)
    row = dict(fold=number, grid=grid, selected=chosen, inner_scores=scores.tolist(),
               inner_means=means.tolist(), inner_tree_diagnostics=diagnostics, final_tree_diagnostics=final,
               selection_objective='mean(clean macro-F1, white_10 macro-F1)',
               tie_rule='exact maximum; first/smallest depth',
               near_best_depths=[grid[i] for i in np.flatnonzero(np.abs(means - means[best]) <= 1e-12)],
               inner_score_span=float(np.ptp(means)),
               train_people=np.unique(groups[train]).tolist(), test_people=np.unique(groups[test]).tolist(),
               inner_valid_people=[np.unique(groups[v]).tolist() for _, v in inner],
               metrics={c: ev.measure(y[test], probabilities[c], classes) for c in CONDITIONS},
               model_sha256=reproduce.sha(model_path), predictions_sha256=reproduce.sha(pred_path),
               seconds=time.perf_counter() - start)
    reproduce.save(output / 'result.json', row)
    return row


def summarize(rows, grid):
    counts = [sum(r['selected'] == d for r in rows) for d in grid]
    middle = len(grid) // 2
    complete = sorted(r['fold'] for r in rows) == list(range(1, 11))
    return dict(status='COMPLETE' if complete else 'PARTIAL', outer_folds=len(rows),
                grid=grid, selected_counts=counts,
                middle_unique_mode=(counts[middle] > max(counts[:middle] + counts[middle+1:])) if complete else None,
                near_tied_folds=[r['fold'] for r in rows if len(r['near_best_depths']) > 1],
                metrics={c: {m: {'mean': float(np.mean([r['metrics'][c][m] for r in rows])),
                                 'fold_sd': float(np.std([r['metrics'][c][m] for r in rows], ddof=1)) if len(rows)>1 else None}
                             for m in reproduce.METRICS} for c in CONDITIONS},
                limitations=['Grid is prospective, not an established optimum.',
                             'Do not choose a new grid using outer test scores.',
                             'Partial runs cannot establish midpoint compliance or replace report results.',
                             'Fold SD is descriptive, not an independent confidence interval.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True, help='Existing verified data/ directory with feature NPZ and inventory.json.')
    p.add_argument('--work-dir', type=Path, default=reproduce.REPO / '.rf-depth-v3')
    p.add_argument('--representation', choices=['logmel', 'mfcc_delta'], required=True)
    p.add_argument('--mode', choices=['clean_matched', 'augmented'], required=True)
    p.add_argument('--depths', type=int, nargs='+', default=DEFAULT_GRID)
    p.add_argument('--fold', type=int, choices=range(1, 11), help='Optional smoke/individual fold; omitted runs all ten.')
    a = p.parse_args()
    validate_grid(a.depths)
    cache = a.cache_dir.resolve()
    work = a.work_dir.resolve()
    # Never place derived artifacts in historical evidence, source or data directories.
    for protected in (reproduce.SOURCE, reproduce.REPO/'reproduced_results', cache):
        if work == protected or protected in work.parents:
            raise ValueError('Choose a separate work directory outside source, evidence and cache.')
    inventory = json.loads((cache / 'inventory.json').read_text(encoding='utf-8'))
    for filename, field in [('features.py','frontend_sha256'), ('prepare_data.py','prepare_sha256')]:
        assert reproduce.sha(reproduce.SOURCE/filename) == inventory['signature'][field]
    feature = cache / (a.representation + '.npz')
    assert reproduce.sha(feature) == inventory['cache_sha256'][a.representation], 'Feature hash mismatch'
    with np.load(feature) as data:
        y, groups, ids = data['y'], data['groups'], data['ids']
        variants = {c: data[c] for c in CONDITIONS}
    assert len(y) == len(set(ids)) == 12000 and len(np.unique(groups)) == 60
    assert np.array_equal(np.bincount(y), np.repeat(1200,10))
    width = {'logmel':256,'mfcc_delta':320}[a.representation]
    assert all(x.shape == (12000,width) and np.isfinite(x).all() for x in variants.values())
    output = work / f'{a.representation}__forest__{a.mode}'
    config = dict(protocol='rf-depth-v3', grid=a.depths, grid_origin=GRID_ORIGIN if a.depths==DEFAULT_GRID else 'User-specified grid', representation=a.representation, mode=a.mode,
                  seed=SEED, outer_folds=10, inner_folds=3, feature_sha256=reproduce.sha(feature),
                  sources={f:reproduce.sha(reproduce.SOURCE/f) for f in ('evaluation.py','common.py','features.py')},
                  runner_sha256=reproduce.sha(__file__), fixed_parameters=ev.model_for('forest',a.depths[0],SEED).get_params())
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration/source changed; choose a new work directory.')
    reproduce.save(config_path, config)
    rows=[]
    for number,(train,test) in enumerate(ev.grouped_folds(groups,np.arange(len(y)),10,SEED),1):
        folder=output/f'fold_{number:02}'
        path=folder/'result.json'
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
            assert row['model_sha256']==reproduce.sha(folder/'model.pkl')
            assert row['predictions_sha256']==reproduce.sha(folder/'predictions.npz')
            rows.append(row)
        elif a.fold is None or a.fold==number:
            print('START',number,flush=True)
            row=train_fold(variants,y,groups,a.representation,a.mode,a.depths,number,train,test,folder)
            rows.append(row)
            print('DONE',number,'selected',row['selected'],'seconds',round(row['seconds'],2),flush=True)
        if rows:
            reproduce.save(output/'summary.json',summarize(rows,a.depths))
    print(output/'summary.json')


if __name__=='__main__':
    main()
