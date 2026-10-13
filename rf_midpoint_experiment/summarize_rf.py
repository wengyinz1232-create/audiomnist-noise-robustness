"""Verify persisted predictions and selections, then build a complete paired summary."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import tune_rf as rf

METRICS=['macro_f1','accuracy','macro_precision','balanced_accuracy','log_loss']

def independent_metrics(y,p):
    assert p.shape==(len(y),10) and np.isfinite(p).all()
    assert np.min(p)>=0 and np.max(p)<=1+1e-12
    np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
    cm=np.bincount(10*y+np.argmax(p,axis=1),minlength=100).reshape(10,10)
    tp=np.diag(cm).astype(float)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(10),where=cm.sum(0)!=0)
    recall=tp/cm.sum(1)
    f=np.divide(2*precision*recall,precision+recall,out=np.zeros(10),where=(precision+recall)!=0)
    return dict(macro_f1=float(f.mean()),accuracy=float(tp.sum()/len(y)),
                macro_precision=float(precision.mean()),balanced_accuracy=float(recall.mean()),
                log_loss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,None)).mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=rf.ROOT/'results')
    ap.add_argument('--legacy-summary',type=Path,required=True)
    ap.add_argument('--deliverables',type=Path,required=True)
    a=ap.parse_args()
    plan=json.loads((a.results/'protocol.json').read_text(encoding='utf-8'))
    with a.legacy_summary.open(encoding='utf-8-sig') as f:
        old={(r['representation'],r['mode'],r['condition']):r for r in csv.DictReader(f) if r['model']=='forest'}
    configurations={}; differences=[]; warnings=[]; selections=[]; candidate_fits=0
    for rep,mode in plan['configurations']:
        config=f'{rep}__forest__{mode}'
        files=sorted((a.results/config).glob('fold_*/result.json'))
        assert len(files)==10, (config,'incomplete',len(files))
        rows=[]; all_indices=[]; all_people=[]
        for f in files:
            r=json.loads(f.read_text(encoding='utf-8')); folder=f.parent
            assert r['representation']==rep and r['mode']==mode
            tr,te=set(r['train_people']),set(r['test_people'])
            assert len(tr)==54 and len(te)==6 and not tr&te
            valid=[set(v) for v in r['inner_valid_people']]
            assert all(len(v)==18 for v in valid) and set.union(*valid)==tr
            assert all(not v&w for i,v in enumerate(valid) for w in valid[i+1:])
            scores={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (folder/'candidates').glob('*.json')}
            assert len(scores)==16
            assert all(len(s['inner'])==3 for s in scores.values())
            assert rf.pick(rf.BASELINE,scores)==r['models']['baseline']['params']
            first=rf.pick(rf.STRUCTURES,scores)
            second=rf.pick([first]+[dict(first,max_features=x) for x in [.1,.3]],scores)
            third=rf.pick([second]+[dict(second,n_estimators=n) for n in [120,240]],scores)
            assert dict(A=first,B=second,C=third)==r['stages']
            assert third==r['models']['tuned']['params']
            assert r['models']['tuned']['inner_mean']>=r['models']['baseline']['inner_mean']-1e-15
            for name,model in r['models'].items():
                assert rf.sha(folder/f'{name}.pkl')==model['model_sha256']
                assert rf.sha(folder/f'{name}_predictions.npz')==model['predictions_sha256']
                assert model['reload_verified']
                with np.load(folder/f'{name}_predictions.npz') as z:
                    assert len(z['y'])==1200 and set(z['groups'])==te
                    assert np.array_equal(z['classes'],np.arange(10))
                    if name=='baseline': all_indices.extend(z['indices'].tolist())
                    for c in rf.CONDITIONS:
                        checked=independent_metrics(z['y'],z[c])
                        for k,v in checked.items():
                            assert abs(v-model['metrics'][c][k])<1e-12,(config,r['fold'],name,c,k)
                warnings.extend(model['warnings'])
            for s in scores.values():
                for item in s['inner']:
                    warnings.extend(item['warnings'])
                    candidate_fits+=int('shared_fit' not in item)
            all_people.extend(te); rows.append(r)
            selections.append(dict(configuration=config,fold=r['fold'],**r['models']['tuned']['params'],
                                   baseline_params=r['models']['baseline']['params'],
                                   inner_gain=r['models']['tuned']['inner_mean']-r['models']['baseline']['inner_mean']))
        assert sorted(all_indices)==list(range(12000))
        assert len(all_people)==len(set(all_people))==60
        summary={}
        for c in rf.CONDITIONS:
            summary[c]={}
            for metric in METRICS:
                b=np.array([r['models']['baseline']['metrics'][c][metric] for r in rows])
                t=np.array([r['models']['tuned']['metrics'][c][metric] for r in rows])
                delta=t-b
                summary[c][metric]=dict(baseline_mean=float(b.mean()),baseline_sd=float(b.std(ddof=1)),
                                       tuned_mean=float(t.mean()),tuned_sd=float(t.std(ddof=1)),
                                       delta_mean=float(delta.mean()),delta_sd=float(delta.std(ddof=1)),
                                       improved_folds=int(np.sum(delta<0 if metric=='log_loss' else delta>0)),
                                       tied_folds=int(np.sum(delta==0)),fold_deltas=delta.tolist())
                old_delta=float(b.mean())-float(old[rep,mode,c][metric+'_mean'])
                differences.append(dict(configuration=config,condition=c,metric=metric,baseline_minus_historical=old_delta))
        counts={k:dict(Counter(str(r['models']['tuned']['params'][k]) for r in rows)) for k in ['max_depth','min_samples_leaf','max_features','n_estimators']}
        configurations[config]=dict(conditions=summary,selected_counts=counts,
                                  fit_seconds={name:float(np.mean([r['models'][name]['fit_seconds'] for r in rows])) for name in ['baseline','tuned']},
                                  selection_seconds_mean=float(np.mean([r['selection_seconds'] for r in rows])))
    legacy_max=max(abs(r['baseline_minus_historical']) for r in differences)
    result=dict(status='VERIFIED_COMPLETE',configurations=configurations,selections=selections,
                verification=dict(outer_folds=40,outer_models=80,inner_fits=candidate_fits,
                                  metric_values_rescored=40*2*4*5,all_prediction_and_model_hashes_verified=True,
                                  all_saved_model_reload_checks_passed=True,all_speaker_splits_disjoint=True,
                                  all_stage_choices_recomputed=True,training_warnings=sorted(set(warnings)),
                                  maximum_baseline_difference_from_historical=legacy_max),
                historical_comparison=differences,protocol=plan)
    a.deliverables.mkdir(parents=True,exist_ok=True)
    rf.save(a.deliverables/'results.json',result)
    rf.save(a.deliverables/'selected_parameters.json',selections)
    lines=['随机森林超参数调优：四组完整十折结果','',
           '范围：Log-Mel / MFCC+delta × 干净训练 / 干净+白噪声10dB增强训练。',
           '方法：外层十折、内层三折，均按说话人分组。每折54位训练/调参、6位测试。',
           '选择目标：内层三折中干净音频与白噪声10dB的平均Macro-F1。粉红噪声不参与拟合或选参。',
           '', '搜索流程（四组相同，每个外层折独立执行）：',
           'A. 60棵树、sqrt特征采样；比较深度8/16/无限制 × 最小叶子样本1/2/5，共9组；另加入旧网格深度4/48/128、叶子样本2，共12组。',
           'B. 固定A选中的深度和叶子样本数，比较sqrt、0.1、0.3的特征采样比例。',
           'C. 固定B选中的结构和特征采样，比较60/120/240棵树。120棵使用240棵森林的前缀；前缀一致性已单独验证。',
           '这是预先规定的分阶段搜索，不是全部参数组合的穷举。精确平局保留候选列表首项，B/C优先保留上一阶段设置。',
           '旧网格[4,48,128]在本次环境中重新选参、拟合和测试，作为配对基线。',
           '', '结果：Macro-F1，均值 ± 十折样本标准差；变化为调优减基线。']
    labels={'clean':'干净','white_20':'白噪声20dB','white_10':'白噪声10dB','pink_10':'粉红噪声10dB'}
    for config,summary in configurations.items():
        lines+=['',config]
        for c,label in labels.items():
            v=summary['conditions'][c]['macro_f1']
            lines.append(f"  {label}: {v['baseline_mean']:.5f} ± {v['baseline_sd']:.5f} → {v['tuned_mean']:.5f} ± {v['tuned_sd']:.5f}; 差值 {v['delta_mean']:+.5f}; 改善 {v['improved_folds']}/10折")
        v=summary['conditions']['pink_10']['log_loss']
        lines.append(f"  粉红噪声log loss（越低越好）: {v['baseline_mean']:.5f} → {v['tuned_mean']:.5f}")
        lines.append('  参数选择频率: '+json.dumps(summary['selected_counts'],ensure_ascii=False))
        lines.append('  平均最终拟合秒数: '+json.dumps(summary['fit_seconds']))
    lines+=['','验证：'+json.dumps(result['verification'],ensure_ascii=False),
            '', '解释限制：',
            '1. 所有外层折内都执行完整选参，但同一数据集此前已被探索，本次属于探索性配对比较，不能宣称独立确认或算法全局最优。',
            '2. 分阶段搜索可能遗漏参数之间的相互作用；边界选择正常报告，不要求中间值胜出。',
            '3. 内层F1改善不保证每种外层条件都改善；不得根据粉红噪声等外层成绩再次修改搜索范围。',
            '4. 标准差描述折间变化；训练集重叠，不能将十折当作独立样本进行简单显著性推断。',
            '5. 不同特征和训练模式分别选参；结果比较的是整个训练流程。',
            '6. 输出80个模型用于配对评估，未训练全数据部署模型。原报告和历史结果未被覆盖。',
            '', '参数定义参考：https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html']
    (a.deliverables/'调优结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(json.dumps(result['verification'],ensure_ascii=False))
    for k,v in configurations.items():
        print(k,json.dumps(v['conditions']['pink_10']['macro_f1']))

if __name__=='__main__':
    main()
