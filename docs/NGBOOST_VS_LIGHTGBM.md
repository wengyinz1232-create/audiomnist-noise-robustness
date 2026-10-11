# NGBoost 与 LightGBM–GOSS：候选模型取舍及证据边界

核查日期：2026-10-12。本文补充代码仓库的方法讨论，不增加第四个已完成实验，也不修改正式三模型的训练结果。

## 当前运行结果是否支持“NGBoost 不如 LightGBM”？

**不支持这一实证结论。** 正式云端运行 [38144868346](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38144868346) 的 12 组配置只包含 Logistic regression、Random forest 和 LightGBM–GOSS。入口 [`reproduce.py`](../reproduce.py) 的配置、依赖文件和保存的 [48 行结果汇总](../reproduced_results/github-actions-38144868346/summary_current_12.csv) 均没有 NGBoost。不能用 LightGBM 的单独分数推断两者的性能差、显著性或耗时比。

| 要判断的结论 | 当前证据 | 可以怎样表述 |
|---|---|---|
| LightGBM 在本流程中能完成复现 | 10×3 嵌套 CV、逐配置审计及已保存结果 | 已完成并核验 |
| NGBoost 的 F1 / log loss 更差 | 无匹配的、可核验的 NGBoost 结果 | 尚不能判断 |
| NGBoost 更慢或更不稳定 | 无匹配硬件计时、逐折日志和失败记录 | 只有机制上的成本/风险分析，不能报告实测差距 |
| NGBoost 有额外自然梯度与线搜索计算 | 官方实现可核查 | 属于实现机制，不等于已证明总时间更长 |
| 选择 LightGBM 作为正式复杂模型 | 已有完整复现证据；NIPS 2017 方法来源 | 是当前项目的证据与实现成熟度选择，不是 NGBoost 普遍劣势的证明 |

用户提供的早期报告曾描述 NGBoost 的三个外折、500 stages、耗时及数值异常。本次在当前代码仓库与正式复现证据中没有找到支持这些数字的 NGBoost 训练脚本、逐折预测、版本和日志。因此不把“所有条件都更差”“约慢 100 倍”“需要约 40 小时”“学习率 0.1 必然不稳定”等初稿说法当作复现结论。即使找回那批证据，三个探索折、默认参数和读取多个测试阶段，也不能替代与当前 LightGBM 配置匹配的十折内层选参比较。

## 方法差异与 NGBoost 的相关局限

NGBoost 是 ICML 2020 方法，通过自然梯度学习预测分布参数。[原论文](https://proceedings.mlr.press/v119/duan20a.html)提供方法依据；官方提供 `NGBClassifier(Dist=k_categorical(10))`，所以它**可以用于本项目的十分类**，不能以“只支持回归”作为不采用的理由。[分类用法](https://stanfordmlgroup.github.io/ngboost/1-useage.html)

### 1. 存在额外的分布几何计算，但总训练成本需要实测

官方 categorical 实现使用 K−1 个自由参数；十分类因此是九个参数。自然梯度使用每个样本的 9×9 Fisher 矩阵解线性系统；树基学习器设置下，每阶段为九个参数分别拟合树，并进行步长线搜索。这些是需要预算的计算环节。[categorical 源码](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/distns/categorical.py)、[梯度求解源码](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/scores.py)、[基学习器与线搜索源码](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/ngboost.py)

LightGBM 使用直方图分桶和高效的分裂统计；本项目还显式使用 GOSS。其多分类模型同样不是“一轮一棵树”：十类、70 轮对应 700 棵树。因此，不能仅凭 NGBoost 的“每轮九棵树”断言其更慢，也不能把 500 与 70 轮当作等预算实验。应在相同设备、线程限制和调参预算下测量时间、内存及预测延迟。[LightGBM 特性](https://lightgbm.readthedocs.io/en/stable/Features.html)、[参数说明](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

本节推断的是“自然梯度和线搜索可能使本任务的成本收益比不理想”，不是已测得的速度结论。所查 NGBoost 开发分支还提供基学习器并行路径，不能笼统声称它无法并行；正式实验必须锁定版本，而不能混用开发分支与旧版本行为。

### 2. 极端概率下需关注数值条件，但没有本项目失败证据

categorical Fisher 矩阵为 `diag(p) − p pᵀ`（省去参考类别）。当概率趋近边界时，该矩阵可能病态；实现涉及线性系统求解与步长选择，增加了需要监测的数值环节。这是数学和源码层面的风险分析，不能写成“NGBoost 在 AudioMNIST 上已经发散”。目前查看的求解源码有奇异系统的逐样本求解和伪逆回退，也不能把旧版本异常直接套用到当前版本。LightGBM 同样不能被视为永远数值稳定。[求解与回退](https://github.com/stanfordmlgroup/ngboost/blob/master/ngboost/scores.py)

### 3. 概率建模优势不会自动转化为更高 F1 或更好校准

在这里 NGBoost 的 categorical 分布就是十个类别概率；LightGBM 的 multiclass 也能输出类别概率并优化多分类对数损失。NGBoost 的通用分布建模能力在这个固定十分类任务中的额外价值，需要由 log loss、可靠性图和拒识效果证明。较低的 log loss 也不能单独证明校准更好。自然梯度不会自动保证更高 macro-F1、对白噪声到粉红噪声的迁移更强，或已解决分布外不确定性；同样，不能断言它在这些指标上一定更差。

### 4. 公平接入现有实验需要额外实现和选参工作

NGBoost 需要选择基学习器容量、学习率、轮数、采样率和早停规则；LightGBM 也有调参成本，因此这不是“NGBoost 独有缺陷”。在当前项目中，LightGBM 已有完整手写 10×3 流程、保存模型核验和依赖锁定，而 NGBoost 尚未接入并验证。为它提供同样标准的比较需要新增适配、锁定依赖、验证概率类别顺序、保存预测并在内折选参。NGBoost 支持 staged prediction 和早停，但应使用内层说话者隔离的验证集，不能依据外层粉红噪声分数挑轮数。[官方调参说明](https://stanfordmlgroup.github.io/ngboost/2-tuning.html)

这些理由足以说明**本次为何保留 LightGBM，暂不把 NGBoost 纳入正式结果**；不足以证明 NGBoost 不适合所有音频任务。若后续 NGBoost 在相同协议下改善概率质量或拒识收益且成本可接受，应据实修改模型选择。

## 当前可核验的 LightGBM 参照

下列仅是 LightGBM 增强训练的外层结果，来自链接的正式 CSV；均为均值 ± 十折样本标准差。它们不是 NGBoost 对照实验。

| 表示 | Clean macro-F1 | Pink-10 macro-F1 | Pink-10 log loss |
|---|---:|---:|---:|
| Log-Mel | 0.9489 ± 0.0229 | 0.8620 ± 0.0430 | 0.3971 ± 0.1066 |
| MFCC+delta | 0.9640 ± 0.0158 | 0.9071 ± 0.0399 | 0.3477 ± 0.1156 |

其他已验证模型中，增强 MFCC Logistic 的 Pink-10 macro-F1 为 0.9213，高于 LightGBM 的 0.9071。故本项目结果甚至不支持“更复杂的 LightGBM 对所有模型、所有条件都占优”，更不能用预设排名描述尚未验证的 NGBoost。

## 怎样获得可写入报告的公平比较

1. 使用相同的原始数据版本、两种特征、说话者外层十折、训练方式和噪声种子；每折重新拟合训练集标准化。保留现有结果，新增独立运行目录。
2. 预先登记 NGBoost 候选范围及资源预算，在手写内层三折中使用相同 clean/W10 平均 macro-F1 目标选择参数。轮数与早停均不得查看外层测试或粉红噪声。不要因为不喜欢测试结果再改网格并宣称独立确认。
3. 用相同预算规则比较调优后的系统；相同轮数不代表相同计算量。若只做少量折的预算试跑，应标为 pilot，不能伪装成完成正式比较。
4. 保存各折概率、预测、手写指标、参数、版本、警告及失败日志。计算 NGBoost−LightGBM 的配对折差、均值、样本 SD 和胜出折数；不要将重叠训练折当作十个独立实验来给未经校正的显著性结论。
5. 在同一机器控制线程，分别记录内层搜索、最终拟合、特征处理、推理延迟与峰值内存。重复随机种子；用预先保留的外部说话者和真实背景声验证最终结论。失败或超时也按预先规则报告，不能删掉不利折。

若 NGBoost 的差异落在折间波动内，就报告“未建立可靠优势”；若它更好，就报告更好，不能为了支持 LightGBM 而改写结果。

## 可用于英文报告的简短表述

> NGBoost was considered as a probabilistic alternative to LightGBM–GOSS. Its categorical natural-gradient updates and line search introduce additional computational steps, but these do not establish a runtime disadvantage without a matched benchmark. Both methods output class probabilities, so NGBoost's distributional formulation does not by itself establish better calibration or noise transfer. We retained LightGBM because it has a complete, verified nested cross-validation implementation in this project. The current evidence contains no matched NGBoost run and therefore does not support claims that NGBoost is less accurate, less stable or a specified factor slower. A fair comparison requires inner-fold tuning, paired unseen-speaker evaluation and controlled resource measurements.

源码链接指向核查时的开发分支，仅作为机制说明；本文没有声称已安装或运行该 NGBoost 版本。方法论文、官方实现与本项目实测证据属于不同证据层级。
