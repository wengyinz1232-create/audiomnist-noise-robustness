"""Build a read-only inventory of saved runs; write only results/current/."""
import csv
import json
import math
from statistics import mean, stdev
from common import ROOT, CONDITIONS, sha, save_json

OUT=ROOT/'results'/'current'
NEW='logistic_selected_grids_20261011'
NEW_L2='lightgbm_l2_selected_20261011'


def write_csv(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def build():
    OUT.mkdir(exist_ok=True)
    historical=ROOT/'results'/'summary.json';original_hash=sha(historical)
    catalog=[];summary=[];records={}
    for path in sorted((ROOT/'runs').glob('*/*/result.json')):
        d=json.loads(path.read_text(encoding='utf-8'));c=d['config'];batch=path.parent.parent.name
        key=f'{batch}/{path.parent.name}';rows={r['fold']:r for r in d['folds']};grid=c['grid']
        assert len(rows)==len(d['folds'])==10 and set(rows)==set(range(1,11)) and 'full_fit' in d,key
        assert len(grid)==3 and grid==sorted(set(grid))
        for f,r in rows.items():
            assert len(r['inner_scores'])==3 and all(len(v)==3 for v in r['inner_scores'])
            scores=[mean(v) for v in r['inner_scores']]
            assert all(math.isfinite(v) for v in scores)
            assert r['selected']==grid[max(range(3),key=lambda i:scores[i])]
            assert not set(r['train_people'])&set(r['test_people'])
            assert set(r['metrics'])==set(CONDITIONS)
            assert (path.parent/f'predictions_{f:02}.npz').is_file()
            assert (path.parent/'models'/f'fold_{f:02}.pkl').is_file()
        assert (path.parent/'models'/'fold_00.pkl').is_file()
        counts=[sum(r['selected']==v for r in rows.values()) for v in grid]
        passes=counts[1]>max(counts[0],counts[2])
        is_old_logistic=batch=='main' and c['representation']=='mfcc_delta' and c['model']=='logistic'
        is_old_l2=batch=='main' and c['representation']=='mfcc_delta' and c['model']=='lightgbm_goss'
        current=(batch=='main' and not is_old_logistic and not is_old_l2) or batch in (NEW,NEW_L2)
        role=('historical_logistic_reference' if is_old_logistic else
              'historical_lightgbm_reference' if is_old_l2 else
              'updated_logistic_different_grids' if batch==NEW else
              'updated_lightgbm_shared_l2_grid' if batch==NEW_L2 else
              'original_main' if batch=='main' else
              'lightgbm_alternative' if batch=='alt_min_child' else
              'multi_snr_supplement' if batch=='multi_snr_20261010' else 'unclassified_supplement')
        limits=[]
        if not passes:limits.append('Does not pass the conservative unique-middle-mode check.')
        if batch==NEW:limits.append('Mode-specific grids; post-exploration rerun, not identical-grid augmentation comparison.')
        if batch==NEW_L2:limits.append('Shared L2 grid with fixed seven-leaf trees; replaces the original leaf-tuned configuration, whose results remain historical.')
        if batch=='alt_min_child':limits.append('Two-leaf alternative; do not pair with original augmented LightGBM as an augmentation-only comparison.')
        limits.append('Previously explored data and splits; no independent confirmation claim.')
        entry={'id':key,'source':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),
               'batch':batch,'representation':c['representation'],'model':c['model'],'mode':c['mode'],
               'role':role,'in_current_12_view':current,'grid':grid,'selected_counts':counts,
               'middle_unique_mode':passes,'outer_folds':10,'full_fit_selected':d['full_fit']['selected'],
               'warning_count':sum(len(r['warnings']) for r in [*rows.values(),d['full_fit']]),
               'superseded_by':f'{NEW}/{path.parent.name}' if is_old_logistic else
                    f'{NEW_L2}/{path.parent.name.replace("__lightgbm_goss__","__lightgbm_goss_l2__")}' if is_old_l2 else None,'limitations':limits}
        catalog.append(entry);records[key]=(d,rows,entry)
        for condition in CONDITIONS:
            result={k:entry[k] for k in ('id','batch','representation','model','mode','role','in_current_12_view','middle_unique_mode','grid','selected_counts')}
            result['condition']=condition
            for metric in ('macro_f1','accuracy','macro_precision','balanced_accuracy','log_loss'):
                values=[rows[f]['metrics'][condition][metric] for f in range(1,11)]
                assert all(math.isfinite(x) for x in values)
                result[metric+'_mean']=mean(values);result[metric+'_fold_sd']=stdev(values)
            summary.append(result)
    assert all(e['superseded_by'] in records for e in catalog if e['superseded_by'])
    comparisons=[]
    for key,(aug,ar,ae) in records.items():
        if ae['mode']!='augmented':continue
        control=key.removesuffix('__augmented')+'__clean_matched'
        if control not in records:continue
        clean,cr,ce=records[control]
        for f in range(1,11):
            for field in ('train_people','test_people','inner_valid_people'):
                assert ar[f][field]==cr[f][field]
            assert ar[f]['selection_objective']==cr[f]['selection_objective']
        for field in ('seed','representation','model','samples','features','signature'):
            assert aug['config'][field]==clean['config'][field]
        same_grid=ae['grid']==ce['grid'];both=ae['middle_unique_mode'] and ce['middle_unique_mode']
        for condition in CONDITIONS:
            diffs=[ar[f]['metrics'][condition]['macro_f1']-cr[f]['metrics'][condition]['macro_f1'] for f in range(1,11)]
            comparisons.append({'clean_id':control,'augmented_id':key,'condition':condition,
                                'same_grid':same_grid,'both_middle_unique':both,
                                'interpretation':'matched_grid_training_modes' if same_grid else 'separately_tuned_pipelines',
                                'paired_difference_mean':mean(diffs),'paired_difference_fold_sd':stdev(diffs),
                                'positive_folds':sum(v>0 for v in diffs),'negative_folds':sum(v<0 for v in diffs)})
    diagnostics=[]
    for folder in sorted((ROOT/'results').glob('diagnosis*')):
        if not folder.is_dir():continue
        source=folder/'summary.json'
        if not source.exists():source=folder/'diagnosis_summary.json'
        if not source.exists():continue
        d=json.loads(source.read_text(encoding='utf-8'))
        incomplete=d.get('ten_fold_diagnostic_complete') is False
        diagnostics.append({'source':str(source.relative_to(ROOT)).replace('\\','/'),'sha256':sha(source),
                            'kind':'incomplete_inner_diagnostic' if incomplete else 'inner_diagnostic',
                            'reported_status':d.get('status','see source'),'is_outer_test_result':False,
                            'include_in_performance_summary':False})
    for path in sorted((ROOT/'provenance').glob('grid_probe_*.json')):
        diagnostics.append({'source':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha(path),
                            'kind':'historical_inner_probe','is_outer_test_result':False,'include_in_performance_summary':False})
    ensemble=ROOT/'analysis'/'三模型等权集成.json'
    supplementary=[{'source':str(ensemble.relative_to(ROOT)).replace('\\','/'),'sha256':sha(ensemble),
                    'role':'post_hoc_ensemble_of_original_main_MFCC_augmented_models',
                    'includes_new_logistic':False,'warning':'Not recomputed using updated logistic; not a separately tuned fourth algorithm.'}] if ensemble.exists() else []
    current=[r for r in summary if r['in_current_12_view']]
    assert sum(e['in_current_12_view'] for e in catalog)==12
    status={'complete_configurations':len(catalog),'current_view_configurations':12,
            'current_view_middle_check_failures':[e['id'] for e in catalog if e['in_current_12_view'] and not e['middle_unique_mode']],
            'current_view_definition':'Twelve configuration slots: original Log-Mel and MFCC forest, updated mode-specific-grid MFCC logistic, and updated shared-L2-grid MFCC LightGBM. Original runs are retained as references.',
            'not_a_compliance_certificate':True,'historical_summary_sha256':original_hash,
            'audit_scope':'Record structure, inner selections, paired splits, artifact presence and recomputed fold summaries; this index does not refit models or re-audit every prediction.',
            'uncertainty':'Fold sample SD, not independent-sample confidence intervals.',
            'artifacts':{'catalog':'catalog.json','current_metrics':'summary_current_12.csv','all_metrics':'summary_all_runs.csv','paired_comparisons':'paired_comparisons.csv','diagnostics':'diagnostics.json'}}
    save_json(OUT/'catalog.json',catalog);save_json(OUT/'summary_all_runs.json',summary)
    save_json(OUT/'paired_comparisons.json',comparisons);save_json(OUT/'diagnostics.json',diagnostics)
    save_json(OUT/'supplementary.json',supplementary);save_json(OUT/'status.json',status)
    write_csv('summary_all_runs.csv',summary);write_csv('summary_current_12.csv',current);write_csv('paired_comparisons.csv',comparisons)
    assert sha(historical)==original_hash
    print('INDEX_COMPLETE',len(catalog),'runs;',len(summary),'metric rows;',len(comparisons),'paired rows;',len(diagnostics),'diagnostic sources')


if __name__=='__main__':build()
