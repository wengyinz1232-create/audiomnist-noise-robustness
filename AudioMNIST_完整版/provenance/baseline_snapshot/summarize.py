"""Generate a teammate handoff from real run records, not a course report."""
import json
import platform
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import numpy as np
from common import ROOT, CONDITIONS, sha, save_json


def summarize():
    paths=sorted((ROOT/'runs').glob('*/*/result.json')); rows=[]; audits=[]; gains=[]
    for path in paths:
        result=json.loads(path.read_text(encoding='utf-8'));config=result['config'];folds=result['folds']
        if len(folds)!=10: continue
        counts=[sum(f['selected']==v for f in folds) for v in config['grid']]
        common={k:config[k] for k in ('representation','model','mode','grid')}
        common.update({'run_id':path.parent.parent.name,'config_dir':str(path.parent.relative_to(ROOT))})
        audit=common|{'selection_counts':counts,'middle_unique_mode':counts[1]>max(counts[0],counts[2]),
                      'middle_majority':counts[1]>5,'full_fit_selected':result.get('full_fit',{}).get('selected'),
                      'middle_upper_tie_folds':sum(abs(f['inner_means'][1]-f['inner_means'][2])<1e-12 for f in folds),
                      'mean_inner_score_range':float(np.mean([max(f['inner_means'])-min(f['inner_means']) for f in folds])),
                      'warning_count':sum(len(f['warnings']) for f in folds)+len(result.get('full_fit',{}).get('warnings',[])),
                      'seconds':sum(f['seconds'] for f in folds)+result.get('full_fit',{}).get('seconds',0)}
        audits.append(audit)
        for condition in CONDITIONS:
            row=common|{'condition':condition,'outer_folds':len(folds)}
            for metric in ('macro_f1','accuracy','macro_precision','balanced_accuracy','log_loss'):
                values=[f['metrics'][condition][metric] for f in folds]
                row[metric+'_mean']=float(np.mean(values));row[metric+'_sd']=float(np.std(values,ddof=1))
            rows.append(row)
        if config['mode']=='augmented':
            control=path.parent.parent/(path.parent.name.replace('__augmented','__clean_matched'))/'result.json'
            if control.exists():
                other=json.loads(control.read_text(encoding='utf-8'))
                if len(other['folds'])==10 and other['config']['grid']==config['grid']:
                    paired={f['fold']:f for f in other['folds']}
                    for condition in CONDITIONS:
                        delta=[f['metrics'][condition]['macro_f1']-paired[f['fold']]['metrics'][condition]['macro_f1'] for f in folds]
                        gains.append(common|{'condition':condition,'mean_gain':float(np.mean(delta)),
                            'gain_sd':float(np.std(delta,ddof=1)),'positive_folds':sum(v>0 for v in delta)})
    save_json(ROOT/'results'/'summary.json',rows);save_json(ROOT/'results'/'grid_audit.json',audits)
    save_json(ROOT/'results'/'paired_augmentation.json',gains)
    text=['# AudioMNIST 本次重新训练结果','',
          '这是数据/代码交接记录，不是课程四页 report。数值来自本目录新生成的结果；没有复制旧训练结果。',
          '所有数值为未见说话人的 10 外折均值；± 为折间样本标准差，不是置信区间。',
          '两种策略均按 clean/white10 验证 macro-F1 平均值选参；粉红噪声未参与训练或选参。',
          '外折样本 10,800 训练 / 1,200 测试；增强训练用 21,600 行，但独立训练录音仍为 10,800。','',
          '## Macro-F1（均值 ± 标准差）','',
          '|批次|特征|模型|训练策略|干净|白噪声20dB|白噪声10dB|粉红噪声10dB|',
          '|---|---|---|---|---:|---:|---:|---:|']
    for audit in audits:
        selected=[next(r for r in rows if r['config_dir']==audit['config_dir'] and r['condition']==c) for c in CONDITIONS]
        text.append('|'+ '|'.join([audit['run_id'],audit['representation'],audit['model'],audit['mode']]+[f"{r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f}" for r in selected])+'|')
    text+=['','## 超参数审查','',
           '中间值唯一众数与超过半数分开报告；未满足项不得标记为已完成 2.f。','',
           '|批次/特征/模型/策略|候选值|选择次数|中间唯一众数|中间超过半数|全量重训选参|收敛警告|',
           '|---|---|---|---|---|---:|---:|']
    for a in audits:
        text.append(f"|{a['run_id']}/{a['representation']}/{a['model']}/{a['mode']}|{a['grid']}|{a['selection_counts']}|{a['middle_unique_mode']}|{a['middle_majority']}|{a['full_fit_selected']}|{a['warning_count']}|")
    text+=['','参数敏感性补充：中间值获选不一定表示它显著优于大值；达到树深度上限平台时，同分按预定规则选较小值。','']
    for a in audits:
        text.append(f"- {a['run_id']}/{a['representation']}/{a['model']}/{a['mode']}：内折均分最大与最小差的外折平均={a['mean_inner_score_range']:.4f}；中间/最大值均分完全相同的折数={a['middle_upper_tie_folds']}/10。")
    text+=['','## 增强的按折配对收益','', '|批次/特征/模型|测试条件|F1 增量均值 ± SD|正向折数|','|---|---|---:|---:|']
    for r in gains:
        text.append(f"|{r['run_id']}/{r['representation']}/{r['model']}|{r['condition']}|{r['mean_gain']:+.3f} ± {r['gain_sd']:.3f}|{r['positive_folds']}/10|")
    text+=['','## 训练产物与边界','',
        '- 每套配置保留 10 个外折模型及 1 个全量模型（models/fold_00.pkl）。全量模型另做三折选参，使用全部 12,000 条录音训练，没有独立测试分数。',
        '- 每外折 3 参数 × 3 内折 + 1 次重训 = 10 次拟合；每套额外全量调参/拟合 10 次，因此每套共 110 次。',
        '- 只加载可信来源的 pickle。模型文件随本次 Python/依赖版本一起使用。',
        '- 原始波形基线、MFCC 不带变化率消融不在本次执行范围。当前不能单独归因变化率的贡献。',
        '- 沿用此前探索选定的数据、特征和划分；重跑不是全新独立确认实验。',
        '- 题目是否接受、复杂模型是否课堂未讲过，仍需组员/教师确认；本次不写 report、不提交。','']
    (ROOT/'TRAINING_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','lightgbm','threadpoolctl']}
    save_json(ROOT/'RUN_MANIFEST.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'packages':packages,'complete_configurations':len(audits),
        'outer_models':10*len(audits),'full_data_models':len(audits),'total_model_fits':110*len(audits),
        'source_code_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},
        'result_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'raw_sha256':sha(ROOT/'data_raw'/'audiomnist.zip'),
        'verification_sha256':sha(ROOT/'VERIFICATION.json') if (ROOT/'VERIFICATION.json').exists() else None})
    print('SUMMARY_COMPLETE',len(audits),'configurations',flush=True)


if __name__=='__main__':summarize()
