"""Independent prediction verification and transparent report for constrained grids."""
import argparse
import json
from pathlib import Path
import numpy as np
import tune_rf as rf
from audit_midpoints import audit
from summarize_rf import independent_metrics,METRICS
from train_rf_midpoint import candidate,validate_config,local_config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,required=True)
    p.add_argument('--prior',type=Path,default=rf.ROOT/'results')
    p.add_argument('--screen',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    protocol=json.loads((a.results/'protocol.json').read_text(encoding='utf-8'))
    config=protocol['config']; validate_config(config)
    screening=json.loads((a.screen/'midpoint_audit.json').read_text(encoding='utf-8'))
    configurations={}; checked=0; parameters=[]
    for rep,mode in [(r,m) for r in ['logmel','mfcc_delta'] for m in ['clean_matched','augmented']]:
        name=f'{rep}__forest__{mode}'; rows=[]; baselines=[]; indices=[]; people=[]
        cfg=local_config(config,rep,mode)
        for f in range(1,11):
            folder=a.results/name/f'fold_{f:02}'
            r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            old=json.loads((a.prior/name/f'fold_{f:02}'/'result.json').read_text(encoding='utf-8'))
            assert r['test_people']==old['test_people'] and r['inner_valid_people']==old['inner_valid_people']
            assert not set(r['train_people'])&set(r['test_people'])
            assert r['params']==candidate(cfg,r['selected'])
            assert r['grid']==cfg['grid'] and r['parameter']==cfg['parameter']
            means=np.mean(r['inner_scores'],axis=1)
            np.testing.assert_allclose(means,r['inner_means'],atol=1e-15,rtol=0)
            original=np.asarray(screening['full_space_scores'][name])[f-1]
            expected=original[[screening['full_space'].index(value) for value in cfg['grid']]]
            np.testing.assert_allclose(means,expected,atol=1e-15,rtol=0)
            assert r['selected']==r['grid'][int(np.argmax(means))]
            assert rf.sha(folder/'model.pkl')==r['model_sha256']
            assert rf.sha(folder/'predictions.npz')==r['predictions_sha256']
            assert r['reload_verified'] and not r['warnings']
            baseline_folder=a.prior/name/f'fold_{f:02}'
            baseline=old['models']['baseline']
            assert rf.sha(baseline_folder/'baseline.pkl')==baseline['model_sha256']
            assert rf.sha(baseline_folder/'baseline_predictions.npz')==baseline['predictions_sha256']
            assert baseline['reload_verified']
            for origin in r['origins']: assert rf.sha(origin['source'])==origin['sha256']
            with np.load(folder/'predictions.npz') as z:
                assert len(z['y'])==1200 and set(z['groups'])==set(r['test_people'])
                indices.extend(z['indices'].tolist())
                for c in rf.CONDITIONS:
                    computed=independent_metrics(z['y'],z[c])
                    for m in METRICS:
                        assert abs(computed[m]-r['metrics'][c][m])<1e-12
                        checked+=1
                with np.load(baseline_folder/'baseline_predictions.npz') as before:
                    for field in ['indices','y','groups','classes']:
                        np.testing.assert_array_equal(before[field],z[field])
                    for c in rf.CONDITIONS:
                        computed=independent_metrics(before['y'],before[c])
                        for m in METRICS:
                            assert abs(computed[m]-baseline['metrics'][c][m])<1e-12
                            checked+=1
            people.extend(r['test_people']); rows.append(r); baselines.append(old['models']['baseline'])
            parameters.append(dict(configuration=name,fold=f,selected=r['selected'],params=r['params']))
        assert sorted(indices)==list(range(12000)) and len(people)==len(set(people))==60
        check=audit([r['inner_means'] for r in rows])
        conditions={}
        for c in rf.CONDITIONS:
            conditions[c]={}
            for m in METRICS:
                before=np.array([b['metrics'][c][m] for b in baselines])
                after=np.array([r['metrics'][c][m] for r in rows]); delta=after-before
                conditions[c][m]=dict(baseline_mean=float(before.mean()),baseline_sd=float(before.std(ddof=1)),
                                      midpoint_mean=float(after.mean()),midpoint_sd=float(after.std(ddof=1)),
                                      delta_mean=float(delta.mean()),delta_sd=float(delta.std(ddof=1)),
                                      improved_folds=int(np.sum(delta<0 if m=='log_loss' else delta>0)),fold_deltas=delta.tolist())
        broad=np.asarray(screening['full_space_scores'][name])
        restricted=np.asarray([r['inner_means'] for r in rows])
        loss=broad.max(1)-restricted.max(1)
        assert loss.min()>-1e-12
        configurations[name]=dict(midpoint_audit=check,conditions=conditions,
                                  mean_inner_score_loss_vs_full_screen=float(loss.mean()),
                                  folds_with_lower_inner_score_than_full_screen=int(np.sum(loss>1e-12)),
                                  final_fit_seconds_mean=float(np.mean([r['fit_seconds'] for r in rows])))
    passed=all(v['midpoint_audit']['middle_unique_mode'] and not v['midpoint_audit']['middle_tie_folds'] for v in configurations.values())
    result=dict(status='VERIFIED_PASS' if passed else 'VERIFIED_FAIL',config=config,configurations=configurations,
                verification=dict(outer_folds=40,new_outer_models=40,paired_baseline_models=40,metrics_independently_rescored=checked,
                                  all_argmax_choices_verified=True,all_model_and_prediction_hashes_verified=True,
                                  all_inner_score_source_hashes_verified=True,all_reload_checks_passed=True),
                exploration_note='Three-point grid was chosen after inspecting inner-score counts across the same previously explored outer partitions. It is not independent validation; all four screened values and all tried triples are retained.')
    a.output.mkdir(parents=True,exist_ok=True)
    rf.save(a.output/'中间值方案完整结果.json',result)
    rf.save(a.output/'中间值方案逐折参数.json',parameters)
    rf.save(a.output/'midpoint_config.json',config)
    lines=['随机森林中间值方案：四组完整十折验证','',
           '状态：'+result['status'],
           '目标：四组配置分别满足中间值是十折唯一最常选中的值。每折继续按内层验证Macro-F1最大值选择，绝不强制中间值。',
           '当前调参参数：'+config['parameter'], '四组各自的三点网格：'+json.dumps(config.get('grids',config.get('grid')),ensure_ascii=False),
           '固定设置：'+json.dumps(config['fixed'],ensure_ascii=False),
           '验证：外层十折、内层三折，说话人分组；内层目标为干净音频和白噪声10dB平均Macro-F1。',
           '', '中间值获选次数：']
    for name,v in configurations.items():
        c=v['midpoint_audit']; lines.append(f"{name}: {c['counts']}; 中间值严格胜出{c['strict_middle_wins']}折; 与端点同分的中间获选折={c['middle_tie_folds']}")
    lines+=['','外层Macro-F1：旧网格基线 → 中间值方案；括号内为差值。']
    for name,v in configurations.items():
        lines+=['',name]
        for condition,m in v['conditions'].items():
            q=m['macro_f1']; lines.append(f"  {condition}: {q['baseline_mean']:.5f} ± {q['baseline_sd']:.5f} → {q['midpoint_mean']:.5f} ± {q['midpoint_sd']:.5f} ({q['delta_mean']:+.5f}); 改善{q['improved_folds']}/10折")
        q=v['conditions']['pink_10']['log_loss']; lines.append(f"  pink_10 log loss（越低越好）: {q['baseline_mean']:.5f} → {q['midpoint_mean']:.5f}")
        lines.append(f"  与全部四个采样候选相比，收窄为三点网格的平均内层F1损失：{v['mean_inner_score_loss_vs_full_screen']:.6f}；涉及{v['folds_with_lower_inner_score_than_full_screen']}/10折。")
    lines+=['','完整核验：'+json.dumps(result['verification'],ensure_ascii=False),
            '', '必须在报告中保留的方法说明：',
            '本方案先对四组使用相同的四点候选范围，再根据内层分数为各组分别筛选满足中间值唯一众数条件的三点网格。各组最终范围不同，不能写成共同三点网格。搜索范围的选择本身已使用这些划分的信息。',
            '因此这些结果是探索性比较，不能写成预先固定网格上的独立确认；需新说话人或新数据才能独立检验。',
            '全部四点候选与全部三点组合的结果都予以保留。中间值方案不一定是更广搜索中的最高性能方案，不能把“中间值最常选”写成“全局最优”。',
            '中间值唯一众数不要求超过半数，但禁止与端点并列众数。中间值靠同分顺序获选不计为自然峰值证据。',
            '每个外层折都独立按内层最大分数选参数，再重新训练并评估外层测试说话人。粉红噪声不参与内层选参。',
            '改变固定结构和搜索空间后，性能不保证在每种测试条件下改善；应同时报告全部四种测试条件。',
            '本机各阶段并发负载不同，记录的拟合耗时仅作运行记录，不构成受控的算法速度比较。',
            '十折训练集重叠，样本标准差仅描述折间变化；没有作独立样本显著性声明。',
            '', '复现入口：train_rf_midpoint.py。省略--reuse-inner-screen即可从头计算已锁定网格的全部内层分数；检查失败会如实输出FAIL。']
    (a.output/'中间值方案结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print(result['status'])
    for name,v in configurations.items(): print(name,v['midpoint_audit']['counts'],v['conditions']['pink_10']['macro_f1'])

if __name__=='__main__': main()
