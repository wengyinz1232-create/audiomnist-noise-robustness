"""Diagnose an existing joint search using ONLY saved inner validation scores.

Reads downloaded fold ZIPs (or extracted result.json/audit.json pairs). Does not
train models, change a grid, rerank by outer-test scores, or force midpoint wins.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import zipfile

PARAMETERS=('learning_rate','n_estimators','num_leaves')
EXPECTED={f'{r}__lightgbm_goss_joint__{m}' for r in ('logmel','mfcc_delta') for m in ('clean_matched','augmented')}


def read_evidence(root):
    records={}
    def accept(raw,audit_raw):
        doc=json.loads(raw);audit=json.loads(audit_raw)
        if doc.get('config',{}).get('key') not in EXPECTED:return
        if audit.get('status')!='PASS' or audit.get('result_sha256')!=hashlib.sha256(raw).hexdigest():
            raise ValueError('Missing or invalid result audit')
        identity=(doc['config']['key'],doc['result']['fold'])
        if identity in records and records[identity]!=doc:raise ValueError(f'Conflicting duplicate fold: {identity}')
        records[identity]=doc
    for path in sorted(root.rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith('/result.json'):
                    folder=name.rsplit('/',1)[0]
                    if folder+'/audit.json' in archive.namelist():
                        accept(archive.read(name),archive.read(folder+'/audit.json'))
    for path in sorted(root.rglob('result.json')):
        if path.with_name('audit.json').exists():accept(path.read_bytes(),path.with_name('audit.json').read_bytes())
    if not records:raise ValueError('No audited joint LightGBM fold evidence found')
    return list(records.values())


def analyze_group(documents):
    documents=sorted(documents,key=lambda d:d['result']['fold'])
    config=documents[0]['config'];grid=config['grid']
    values={p:sorted({g[p] for g in grid}) for p in PARAMETERS}
    if any(len(v)!=3 for v in values.values()):raise ValueError('Expected three values per parameter')
    if {tuple(g[p] for p in PARAMETERS) for g in grid}!=set(itertools.product(*values.values())) or len(grid)!=27:
        raise ValueError('Incomplete Cartesian grid')
    folds=[d['result']['fold'] for d in documents]
    if len(folds)!=len(set(folds)) or not set(folds)<=set(range(1,11)):raise ValueError('Invalid outer-fold identities')
    means=[];winners=[];gaps=[];ties=0
    for doc in documents:
        if doc['config']!=config:raise ValueError('Mixed source, data, environment or search protocols')
        row=doc['result'];scores=row['inner_scores']
        if row['grid']!=grid or len(scores)!=27 or any(len(s)!=3 for s in scores):raise ValueError('Invalid inner score matrix')
        if not all(math.isfinite(x) and 0<=x<=1 for s in scores for x in s):raise ValueError('Invalid macro-F1')
        avg=[sum(s)/3 for s in scores];best=max(range(27),key=avg.__getitem__)
        if row['selected']!=grid[best]:raise ValueError('Selection is not the deterministic inner argmax')
        if len(row['inner_means'])!=27 or any(abs(a-b)>1e-12 for a,b in zip(avg,row['inner_means'])):
            raise ValueError('Stored inner means differ from recomputation')
        ordered=sorted(avg,reverse=True);gap=ordered[0]-ordered[1]
        gaps.append(gap);ties+=gap<=1e-12
        means.append(avg);winners.append(grid[best])
    complete=folds==list(range(1,11));parameters={}
    for param,choices in values.items():
        counts=[sum(w[param]==v for w in winners) for v in choices]
        # Optimize the other two parameters separately in each inner search.
        # These profiles are descriptive, not new outer-test performance estimates.
        profiles=[[max(m[i] for i,g in enumerate(grid) if g[param]==v) for v in choices] for m in means]
        middle_loss=[max(profile)-profile[1] for profile in profiles]
        if not complete:action='WAIT_FOR_ALL_TEN_FOLDS'
        elif counts[1]>5:action='KEEP_RANGE'
        elif counts[2]>5:action='INVESTIGATE_HIGHER_RANGE'
        elif counts[0]>5:action='INVESTIGATE_LOWER_RANGE'
        else:action='INSPECT_INTERACTIONS_AND_SCORE_GAPS'
        parameters[param]={'values':choices,'winner_counts':counts,
            'middle_unique_mode':counts[1]>max(counts[0],counts[2]) if complete else None,
            'middle_strict_majority':counts[1]>5 if complete else None,
            'profile_mean_inner_f1':[statistics.mean(p[i] for p in profiles) for i in range(3)],
            'mean_inner_f1_loss_if_middle_fixed':statistics.mean(middle_loss),
            'max_inner_f1_loss_if_middle_fixed':max(middle_loss),'next_action':action}
    pairs=[]
    for rate,rounds in itertools.product(values['learning_rate'],values['n_estimators']):
        profile=[max(m[i] for i,g in enumerate(grid) if g['learning_rate']==rate and g['n_estimators']==rounds) for m in means]
        pairs.append({'learning_rate':rate,'n_estimators':rounds,'profile_mean_inner_f1':statistics.mean(profile),
                      'joint_winner_count':sum(w['learning_rate']==rate and w['n_estimators']==rounds for w in winners)})
    return {'folds':folds,'complete':complete,'parameters':parameters,'learning_rate_rounds_interaction':pairs,
            'winner_runner_up_gap_mean':statistics.mean(gaps),'winner_runner_up_gap_median':statistics.median(gaps),
            'tied_best_folds':ties,'sum_fold_training_seconds':sum(d['result']['seconds'] for d in documents)}


def analyze(documents,allow_partial=False):
    grouped={}
    for doc in documents:grouped.setdefault(doc['config']['key'],[]).append(doc)
    if not set(grouped)<=EXPECTED:raise ValueError('Unexpected configuration')
    result={key:analyze_group(docs) for key,docs in sorted(grouped.items())}
    complete=set(result)==EXPECTED and all(v['complete'] for v in result.values())
    if not allow_partial and not complete:raise ValueError('All forty folds are required; use --allow-partial for provisional diagnostics')
    return {'status':'COMPLETE' if complete else 'PROVISIONAL','configurations':result,
        'missing_configurations':sorted(EXPECTED-set(result)),
        'decision_basis':'Inner validation scores only; outer metrics and P10 are not read.',
        'limits':['Profiles maximize over other searched parameters; they are descriptive inner scores, not generalization estimates.',
                  'Cross-fold grid revision is exploratory and must be disclosed.',
                  'A midpoint count target does not justify overriding a higher validation score.',
                  'Consider learning rate and rounds together; no proposed range is applied automatically.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();report=analyze(read_evidence(args.evidence),args.allow_partial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report['status'],args.output)


if __name__=='__main__':main()
