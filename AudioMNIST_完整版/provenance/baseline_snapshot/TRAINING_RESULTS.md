# AudioMNIST 本次重新训练结果

这是数据/代码交接记录，不是课程四页 report。数值来自本目录新生成的结果；没有复制旧训练结果。
所有数值为未见说话人的 10 外折均值；± 为折间样本标准差，不是置信区间。
两种策略均按 clean/white10 验证 macro-F1 平均值选参；粉红噪声未参与训练或选参。
外折样本 10,800 训练 / 1,200 测试；增强训练用 21,600 行，但独立训练录音仍为 10,800。

## Macro-F1（均值 ± 标准差）

|批次|特征|模型|训练策略|干净|白噪声20dB|白噪声10dB|粉红噪声10dB|
|---|---|---|---|---:|---:|---:|---:|
|main|logmel|forest|augmented|0.930 ± 0.021|0.804 ± 0.047|0.893 ± 0.037|0.824 ± 0.045|
|main|logmel|forest|clean_matched|0.928 ± 0.021|0.268 ± 0.035|0.056 ± 0.018|0.104 ± 0.028|
|main|logmel|lightgbm_goss|augmented|0.949 ± 0.023|0.870 ± 0.043|0.912 ± 0.033|0.862 ± 0.043|
|main|logmel|lightgbm_goss|clean_matched|0.945 ± 0.023|0.365 ± 0.029|0.133 ± 0.032|0.196 ± 0.032|
|main|logmel|logistic|augmented|0.962 ± 0.021|0.949 ± 0.025|0.934 ± 0.030|0.882 ± 0.044|
|main|logmel|logistic|clean_matched|0.968 ± 0.009|0.296 ± 0.055|0.076 ± 0.022|0.247 ± 0.033|
|main|mfcc_delta|forest|augmented|0.959 ± 0.016|0.931 ± 0.029|0.931 ± 0.024|0.908 ± 0.029|
|main|mfcc_delta|forest|clean_matched|0.967 ± 0.012|0.691 ± 0.053|0.353 ± 0.037|0.504 ± 0.037|
|main|mfcc_delta|lightgbm_goss|augmented|0.973 ± 0.013|0.944 ± 0.025|0.948 ± 0.023|0.908 ± 0.035|
|main|mfcc_delta|lightgbm_goss|clean_matched|0.958 ± 0.016|0.573 ± 0.051|0.206 ± 0.029|0.500 ± 0.038|
|main|mfcc_delta|logistic|augmented|0.957 ± 0.021|0.954 ± 0.027|0.930 ± 0.035|0.917 ± 0.035|
|main|mfcc_delta|logistic|clean_matched|0.949 ± 0.047|0.676 ± 0.101|0.316 ± 0.051|0.431 ± 0.039|

## 超参数审查

中间值唯一众数与超过半数分开报告；未满足项不得标记为已完成 2.f。

|批次/特征/模型/策略|候选值|选择次数|中间唯一众数|中间超过半数|全量重训选参|收敛警告|
|---|---|---|---|---|---:|---:|
|main/logmel/forest/augmented|[4, 48, 128]|[0, 10, 0]|True|True|48|0|
|main/logmel/forest/clean_matched|[4, 48, 128]|[0, 10, 0]|True|True|48|0|
|main/logmel/lightgbm_goss/augmented|[3, 63, 255]|[0, 8, 2]|True|True|63|0|
|main/logmel/lightgbm_goss/clean_matched|[3, 63, 255]|[0, 7, 3]|True|True|255|0|
|main/logmel/logistic/augmented|[0.0001, 0.1, 100.0]|[0, 10, 0]|True|True|0.1|0|
|main/logmel/logistic/clean_matched|[0.0001, 0.1, 100.0]|[0, 6, 4]|True|True|0.1|0|
|main/mfcc_delta/forest/augmented|[4, 48, 128]|[0, 10, 0]|True|True|48|0|
|main/mfcc_delta/forest/clean_matched|[4, 48, 128]|[0, 10, 0]|True|True|128|0|
|main/mfcc_delta/lightgbm_goss/augmented|[3, 63, 255]|[0, 7, 3]|True|True|63|0|
|main/mfcc_delta/lightgbm_goss/clean_matched|[3, 63, 255]|[10, 0, 0]|False|False|3|0|
|main/mfcc_delta/logistic/augmented|[0.0001, 0.1, 100.0]|[0, 10, 0]|True|True|0.1|0|
|main/mfcc_delta/logistic/clean_matched|[0.0001, 0.1, 100.0]|[2, 0, 8]|False|False|100.0|0|

参数敏感性补充：中间值获选不一定表示它显著优于大值；达到树深度上限平台时，同分按预定规则选较小值。

- main/logmel/forest/augmented：内折均分最大与最小差的外折平均=0.2353；中间/最大值均分完全相同的折数=10/10。
- main/logmel/forest/clean_matched：内折均分最大与最小差的外折平均=0.0706；中间/最大值均分完全相同的折数=10/10。
- main/logmel/lightgbm_goss/augmented：内折均分最大与最小差的外折平均=0.0716；中间/最大值均分完全相同的折数=0/10。
- main/logmel/lightgbm_goss/clean_matched：内折均分最大与最小差的外折平均=0.0274；中间/最大值均分完全相同的折数=5/10。
- main/logmel/logistic/augmented：内折均分最大与最小差的外折平均=0.1410；中间/最大值均分完全相同的折数=0/10。
- main/logmel/logistic/clean_matched：内折均分最大与最小差的外折平均=0.0792；中间/最大值均分完全相同的折数=0/10。
- main/mfcc_delta/forest/augmented：内折均分最大与最小差的外折平均=0.1318；中间/最大值均分完全相同的折数=10/10。
- main/mfcc_delta/forest/clean_matched：内折均分最大与最小差的外折平均=0.0367；中间/最大值均分完全相同的折数=9/10。
- main/mfcc_delta/lightgbm_goss/augmented：内折均分最大与最小差的外折平均=0.0398；中间/最大值均分完全相同的折数=0/10。
- main/mfcc_delta/lightgbm_goss/clean_matched：内折均分最大与最小差的外折平均=0.0262；中间/最大值均分完全相同的折数=10/10。
- main/mfcc_delta/logistic/augmented：内折均分最大与最小差的外折平均=0.0730；中间/最大值均分完全相同的折数=0/10。
- main/mfcc_delta/logistic/clean_matched：内折均分最大与最小差的外折平均=0.0388；中间/最大值均分完全相同的折数=0/10。

## 增强的按折配对收益

|批次/特征/模型|测试条件|F1 增量均值 ± SD|正向折数|
|---|---|---:|---:|
|main/logmel/forest|clean|+0.002 ± 0.007|6/10|
|main/logmel/forest|white_20|+0.536 ± 0.043|10/10|
|main/logmel/forest|white_10|+0.837 ± 0.037|10/10|
|main/logmel/forest|pink_10|+0.721 ± 0.057|10/10|
|main/logmel/lightgbm_goss|clean|+0.004 ± 0.006|7/10|
|main/logmel/lightgbm_goss|white_20|+0.505 ± 0.037|10/10|
|main/logmel/lightgbm_goss|white_10|+0.779 ± 0.044|10/10|
|main/logmel/lightgbm_goss|pink_10|+0.666 ± 0.046|10/10|
|main/logmel/logistic|clean|-0.006 ± 0.014|5/10|
|main/logmel/logistic|white_20|+0.653 ± 0.073|10/10|
|main/logmel/logistic|white_10|+0.858 ± 0.044|10/10|
|main/logmel/logistic|pink_10|+0.635 ± 0.051|10/10|
|main/mfcc_delta/forest|clean|-0.008 ± 0.006|0/10|
|main/mfcc_delta/forest|white_20|+0.241 ± 0.042|10/10|
|main/mfcc_delta/forest|white_10|+0.578 ± 0.038|10/10|
|main/mfcc_delta/forest|pink_10|+0.404 ± 0.039|10/10|
|main/mfcc_delta/lightgbm_goss|clean|+0.015 ± 0.013|9/10|
|main/mfcc_delta/lightgbm_goss|white_20|+0.371 ± 0.037|10/10|
|main/mfcc_delta/lightgbm_goss|white_10|+0.742 ± 0.030|10/10|
|main/mfcc_delta/lightgbm_goss|pink_10|+0.409 ± 0.037|10/10|
|main/mfcc_delta/logistic|clean|+0.009 ± 0.032|3/10|
|main/mfcc_delta/logistic|white_20|+0.278 ± 0.081|10/10|
|main/mfcc_delta/logistic|white_10|+0.614 ± 0.078|10/10|
|main/mfcc_delta/logistic|pink_10|+0.485 ± 0.042|10/10|

## 训练产物与边界

- 每套配置保留 10 个外折模型及 1 个全量模型（models/fold_00.pkl）。全量模型另做三折选参，使用全部 12,000 条录音训练，没有独立测试分数。
- 每外折 3 参数 × 3 内折 + 1 次重训 = 10 次拟合；每套额外全量调参/拟合 10 次，因此每套共 110 次。
- 只加载可信来源的 pickle。模型文件随本次 Python/依赖版本一起使用。
- 原始波形基线、MFCC 不带变化率消融不在本次执行范围。当前不能单独归因变化率的贡献。
- 沿用此前探索选定的数据、特征和划分；重跑不是全新独立确认实验。
- 题目是否接受、复杂模型是否课堂未讲过，仍需组员/教师确认；本次不写 report、不提交。
