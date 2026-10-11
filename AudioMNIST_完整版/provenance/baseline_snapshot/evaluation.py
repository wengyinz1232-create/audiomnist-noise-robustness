"""Manual speaker-disjoint nested CV, hyperparameter search and metrics.

只有分类器拟合调用 sklearn/LightGBM；没有调用 CV、GridSearch 或 metrics 工具。
Both training modes have the SAME grids and validation objective, so the
augmentation comparison does not also change the tuning objective.
"""
import pickle
import time
import warnings
from datetime import datetime, timezone
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from common import ROOT, SEED, CONDITIONS, save_json, sha

# Fixed before this fresh run, informed by the earlier exploratory inner scores.
GRIDS = {'logistic':[.0001,.1,100.], 'forest':[4,48,128], 'lightgbm_goss':[3,63,255]}


def grouped_folds(groups, indices, k, seed):
    people = np.unique(groups[indices])
    assert len(people) >= k
    np.random.default_rng(seed).shuffle(people)
    for block in np.array_split(people,k):
        mask = np.isin(groups[indices],block)
        train,test = indices[~mask],indices[mask]
        assert len(train) and len(test) and not set(groups[train]) & set(groups[test])
        yield train,test


def scaling(x):
    mean = x.mean(0,dtype=np.float64); std = x.std(0,dtype=np.float64)
    std[std<1e-8] = 1
    return mean,std


def transform(x, stats):
    return ((x-stats[0])/stats[1]).astype(np.float32)


def measure(y, probabilities, classes):
    prediction = classes[np.argmax(probabilities,axis=1)]
    mapping = {int(c):i for i,c in enumerate(classes)}
    cm = np.zeros((len(classes),len(classes)),dtype=np.int64)
    for actual,guess in zip(y,prediction): cm[mapping[int(actual)],mapping[int(guess)]] += 1
    tp = np.diag(cm).astype(float)
    precision = tp/np.maximum(cm.sum(0),1); recall = tp/np.maximum(cm.sum(1),1)
    f1 = 2*precision*recall/np.maximum(precision+recall,1e-12)
    columns = np.array([mapping[int(c)] for c in y])
    return {'n':len(y),'accuracy':float(tp.sum()/len(y)), 'macro_f1':float(f1.mean()),
            'macro_precision':float(precision.mean()),'balanced_accuracy':float(recall.mean()),
            'log_loss':float(-np.log(np.maximum(probabilities[np.arange(len(y)),columns],1e-15)).mean()),
            'per_class_precision':precision.tolist(),'per_class_recall':recall.tolist(),
            'per_class_f1':f1.tolist(),'confusion':cm.tolist()}


def model_for(name, value, seed):
    if name == 'logistic':
        return LogisticRegression(C=value,max_iter=2000,tol=1e-3,solver='lbfgs',random_state=seed)
    if name == 'forest':
        return RandomForestClassifier(n_estimators=60,max_depth=int(value),min_samples_leaf=2,n_jobs=2,random_state=seed)
    if name == 'lightgbm_goss':
        return LGBMClassifier(n_estimators=70,num_leaves=int(value),learning_rate=.08,
            data_sample_strategy='goss',top_rate=.2,other_rate=.1,min_child_samples=20,
            reg_lambda=1.,random_state=seed,n_jobs=2,verbosity=-1)
    raise ValueError(name)


def fit(name,value,x,y,seed):
    estimator = model_for(name,value,seed)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=1,user_api='blas'): estimator.fit(x,y)
    return estimator,[str(w.message) for w in records if issubclass(w.category,ConvergenceWarning)]


def proba(estimator,x,classes):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='X does not have valid feature names')
        with threadpool_limits(limits=1,user_api='blas'): p = estimator.predict_proba(x)
    aligned = np.zeros((len(x),len(classes)))
    for col,c in enumerate(estimator.classes_): aligned[:,np.flatnonzero(classes==c)[0]] = p[:,col]
    return aligned


def training_rows(variants,y,index,mode):
    if mode == 'clean_matched': return variants['clean'][index],y[index]
    assert mode == 'augmented'
    # 噪声副本只来自当前训练人员，不增加独立录音数。
    return np.concatenate([variants['clean'][index],variants['white_10'][index]]),np.tile(y[index],2)


def train_fold(variants,y,groups,name,mode,grid,number,train,test,output,rep):
    start = time.perf_counter(); classes = np.unique(y); notes = []
    inner = list(grouped_folds(groups,train,3,SEED+number))
    prepared = []
    for fit_idx,valid_idx in inner:
        xfit,yfit = training_rows(variants,y,fit_idx,mode); stats = scaling(xfit)
        prepared.append((transform(xfit,stats),yfit,
            [transform(variants[c][valid_idx],stats) for c in ('clean','white_10')],y[valid_idx]))
    scores = []
    for value in grid:
        values = []
        for j,(xfit,yfit,xvalid,yvalid) in enumerate(prepared):
            estimator,w = fit(name,value,xfit,yfit,SEED+100*number+j); notes.extend(w)
            values.append(float(np.mean([measure(yvalid,proba(estimator,x,classes),classes)['macro_f1'] for x in xvalid])))
        scores.append(values)
    means = np.mean(scores,axis=1); best = int(np.argmax(means))
    # Ties deliberately prefer first/smaller candidate. No outer scores are used.
    chosen = grid[best]
    xfit,yfit = training_rows(variants,y,train,mode); stats = scaling(xfit)
    estimator,w = fit(name,chosen,transform(xfit,stats),yfit,SEED+number); notes.extend(w)
    probabilities = {c:proba(estimator,transform(x[test],stats),classes) for c,x in variants.items()} if len(test) else {}
    metrics = {c:measure(y[test],p,classes) for c,p in probabilities.items()}
    bundle = {'estimator':estimator,'scaler':stats,'classes':classes,'representation':rep,
              'mode':mode,'selected':chosen,'train_people':np.unique(groups[train]).tolist(),
              'source_signature':sha(ROOT/'features.py'),'warning':'Only load pickle files you trust.'}
    model_path = output/'models'/f'fold_{number:02}.pkl'; model_path.parent.mkdir(parents=True,exist_ok=True)
    with open(model_path,'wb') as stream: pickle.dump(bundle,stream,protocol=5)
    # Reload and compare a few predictions; trained artifacts must actually work.
    with open(model_path,'rb') as stream: reloaded = pickle.load(stream)
    if len(test):
        check = proba(reloaded['estimator'],transform(variants['clean'][test[:8]],reloaded['scaler']),classes)
        # float32 BLAS can round differently for full vs small batches.
        assert np.allclose(check,probabilities['clean'][:8],atol=1e-6,rtol=1e-5)
        np.savez_compressed(output/f'predictions_{number:02}.npz',indices=test,y=y[test],classes=classes,**probabilities)
    return {'fold':number,'selected':chosen,'grid':grid,'inner_scores':scores,'inner_means':means.tolist(),
        'selection_objective':'mean(clean macro-F1, white_10 macro-F1)',
        'train_n':len(train),'test_n':len(test),'fit_rows':len(xfit),
        'train_people':np.unique(groups[train]).tolist(),'test_people':np.unique(groups[test]).tolist(),
        'inner_valid_people':[np.unique(groups[v]).tolist() for _,v in inner],
        'metrics':metrics,'warnings':notes,'model_sha256':sha(model_path),
        'seconds':round(time.perf_counter()-start,3),'finished_utc':datetime.now(timezone.utc).isoformat()}


def run_configuration(rep,name,mode,run_id='main',grid_override=None):
    output = ROOT/'runs'/run_id/f'{rep}__{name}__{mode}'; output.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'data'/f'{rep}.npz') as cache:
        y = cache['y']; groups = cache['groups']; variants = {c:cache[c] for c in CONDITIONS}
    grid = grid_override or GRIDS[name]
    signature = {p.name:sha(p) for p in [ROOT/'data'/f'{rep}.npz',ROOT/'evaluation.py',ROOT/'common.py',ROOT/'features.py']}
    config = {'representation':rep,'model':name,'mode':mode,'grid':grid,'outer_folds':10,'inner_folds':3,
              'seed':SEED,'samples':len(y),'features':variants['clean'].shape[1],'signature':signature}
    path = output/'result.json'
    record = __import__('json').loads(path.read_text(encoding='utf-8')) if path.exists() else {'config':config,'folds':[]}
    assert record['config'] == config, 'Resume rejected: changed configuration/source'
    save_json(path,record)
    completed = {r['fold'] for r in record['folds']}
    folds = list(grouped_folds(groups,np.arange(len(y)),10,SEED))
    for number,(train,test) in enumerate(folds,1):
        if number in completed: continue
        print('START',output.name,number,flush=True)
        row = train_fold(variants,y,groups,name,mode,grid,number,train,test,output,rep)
        record['folds'].append(row); save_json(path,record)
        print('DONE',output.name,number,'selected',row['selected'],'sec',row['seconds'],flush=True)
    if 'full_fit' not in record:
        # Re-tune using 3 speaker folds on all data, then fit a reusable artifact.
        # No test score is assigned to this full-data model.
        record['full_fit'] = train_fold(variants,y,groups,name,mode,grid,0,np.arange(len(y)),np.array([],dtype=int),output,rep)
        save_json(path,record)
    return str(path)
