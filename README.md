# AudioMNIST noise robustness

根目录 reproduce.py 是干净克隆的复现入口。AudioMNIST_完整版/ 保留原代码和历史结果；新入口不依赖历史模型或诊断 JSON。

## 本地运行

需要 Python 3.12、网络和数 GB 空间。在仓库根目录：

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -m unittest discover -s tests -v
python reproduce.py self-test
python reproduce.py all --jobs 2
```

完整运行下载约 1 GB 的公开 AudioMNIST 数据并重新提取特征和训练，需要数十分钟或更久。也可添加 --raw-archive /path/to/audiomnist.zip，指定原始数据 ZIP（不是整个项目的 ZIP）。不要并发向同一运行目录启动多个完整实验。

数据来自 https://github.com/soerenab/AudioMNIST ，遵循上游许可证。固定版本 630d7dab4c040882834de6fa21baf9a60372accd；60 位说话者、10 个数字，各前 20 条，共 12,000 条；种子 20261009。未修改的训练核心复制到独立 .reproduction/，不覆盖历史结果。中断后可继续相同命令；核心代码变化后用新的 --work-dir。

## GitHub Actions

推送运行轻量 Code checks。Actions → Full reproduction → Run workflow 手动启动完整复现。新的 Ubuntu runner 克隆代码、下载数据、训练验证；工作流仅有 contents: read 权限，无需仓库密钥。运行使用账户 Actions 配额，不会自行启用付费服务。

下载 reproduction-evidence 工件（保留 14 天）查看报告、逐配置核验、数据清单及误差棒图。原始音频和模型不上传。绿色工程检查不等于课程全部合规，还要查看 reports/status.json 的 midpoint_failures。

## 配置和核验范围

较早的 RF 深度诊断入口：`optimize_rf.py` 使用训练内诊断后选定的共同 `[4,8,16]` 深度范围，记录实际树深、叶子数、内层分数和并列情况。`diagnose_rf_depth.py` 只使用指定外折的训练说话人检查深度曲线。该范围用于容量受控的 RF，四组中的三组筛查最优仍为上界16；它不是已确认的通用最优网格，也未证明中间值最常选。完整命令见 [RF 深度诊断说明](docs/RF_OPTIMIZATION.md)。当前特征采样比例方案使用下文的 `optimize_rf_midpoint.py`。

原始 12 组复现（`reproduce.py`）= 2 特征 × 3 类模型 × 2 训练方式。说话者隔离的外层 10 折、内层 3 折，共 1,320 次拟合、120 外层模型和 12 全数据模型。该历史复现入口保留原 RF 网格；下文的新 RF 方案使用独立入口。

### 新增 NGBoost 配对训练实验

新增 `extended800` 协议：仅将 NGBoost 轮数网格扩展为 **200 / 400 / 800**，保持学习率 0.03、树结构和十折外层／三折内层选参目标不变。Actions 手动运行时选择该协议；内层选择最佳轮数后才拟合外折模型，并非强制所有外折使用 800 轮。原 `legacy200` 协议和以下历史结果继续保留，新结果已在独立目录完成归档。

[实验说明与运行命令](docs/NGBOOST_EXPERIMENT.md) · [训练及比较入口](compare_ngboost.py) · [GitHub Actions](.github/workflows/ngboost.yml)

以 MFCC+delta＋白噪声增强为范围，NGBoost 和 LightGBM 在相同外层十折、内层三折上选参并训练；每对模型在同一 runner 顺序运行，保存逐折预测、模型重载核验、配对分数差和计时。它是独立补充实验，不改写原 12 组配置。结果以实际训练和完整十折核验为准，不预设 NGBoost 更差；未完成的折不能被删除后用其余折冒充完整比较。

**最新扩展实验已完成：[200/400/800 中文结果与完整证据](reproduced_results/ngboost-38332286883/NGBOOST_RESULTS_ZH.md)**。十折内层全部选择 800 轮；相比旧 200 轮，Clean/W20/W10/P10 的平均 F1 分别提高 0.42/1.01/0.89/1.03 个百分点，仍由 LightGBM 在每种条件十折全部胜出。

| 条件 | 旧 NGBoost 200 | 新 NGBoost 800 | 同轮 LightGBM |
|---|---:|---:|---:|
| Clean macro-F1 | 0.9275 | 0.9317 | 0.9640 |
| White-20 macro-F1 | 0.8772 | 0.8873 | 0.9357 |
| White-10 macro-F1 | 0.8689 | 0.8778 | 0.9373 |
| Pink-10 macro-F1 | 0.8466 | 0.8569 | 0.9071 |

本次平均最终拟合为 NGBoost 3074.39 秒/折、LightGBM 8.24 秒/折（约 373 倍）；NGBoost 的干净/白噪声 log loss 略升，粉红噪声略降，不能把 F1 增益解释成概率质量全面改善。800 仍是上界，仍未满足中间参数检查。

[运行 38332286883](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38332286883) 的十个训练任务全部成功；最后汇总任务因账户付款/消费上限问题未启动，整体页面显示 failure。已从十个折工件本地恢复并核验完整汇总，20 个模型哈希和重载检查通过，预测长期保存；课程报告正文未修改。

**历史预算结果：**

**已完成：[十折实际对比报告](reproduced_results/ngboost-38150310147/NGBOOST_RESULTS_ZH.md) · [成功运行 38150310147](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38150310147)**。20 个外折模型全部通过保存/重载核验，逐折预测已复算；预测与 JSON 长期保存在结果目录，可独立重建汇总。

| 指标（MFCC+delta＋白噪声增强） | NGBoost | LightGBM–GOSS |
|---|---:|---:|
| Clean macro-F1 | 0.9275 ± 0.0282 | 0.9640 ± 0.0158 |
| Pink-10 macro-F1 | 0.8466 ± 0.0374 | 0.9071 ± 0.0399 |
| Pink-10 log loss | 0.4910 ± 0.1063 | 0.3477 ± 0.1156 |
| 平均最终拟合秒/折 | 740.44 | 8.20 |

四种条件的 F1 均为 LightGBM 十折全部胜出；粉红噪声配对 F1 差（NGBoost−LightGBM）为 −0.0605 ± 0.0168。该结论限于这次配置和预算：NGBoost 十折全部选中 200 轮上限，**未通过“中间参数最常选”检查，不能宣称已充分调参或全面满足课程要求**。两者均无训练警告；NGBoost 的 Pink-10 F1 折间 SD 反而略小，所以不能写成“NGBoost 更不稳定”。耗时是同机配对测量，不代表所有实现或硬件的普遍速度比。

| 特征 | 模型 | 网格 |
|---|---|---|
| LogMel | Logistic | C = 0.0001, 0.1, 100 |
| MFCC+delta | Logistic clean | C = 0.0001, 0.003, 0.01 |
| MFCC+delta | Logistic augmented | C = 0.01, 0.03, 0.1 |
| 两种 | Random forest | max_depth = 4, 48, 128 |
| LogMel | LightGBM GOSS | num_leaves = 3, 63, 255 |
| MFCC+delta | LightGBM GOSS | lambda_l2 = 0, 0.1, 1；固定 7 leaves |

MFCC 四组采用最近修订网格，与历史最初 12 组不同。之前已探索数据和网格，这是既有实验复现，不是独立确认实验。不同训练方式的 Logistic 网格不同，应在报告披露。

核验数据/源码/模型哈希、说话者划分、超参数选择、训练折标准化、模型参数，以及重新加载模型预测与保存概率/指标的一致性。integrity_status: PASS 只表示工程核验通过。中间参数唯一众数与超过半数分别报告。跨平台可能有小数值或边界预测差异，不保证逐位一致。

.reproduction/reports/ 中有 status.json、summary_current_12.csv、robustness.png、audits/。误差棒为外层折样本标准差，不是独立置信区间。课程报告页数、会议记录和课堂未讲模型等条件需另外核对。

支持阶段 prepare、train、verify、summarize；list 显示配置名。只核验一个配置可执行：

```bash
python reproduce.py prepare
python reproduce.py train --config logmel__logistic__clean_matched
python reproduce.py verify --config logmel__logistic__clean_matched
```

全部 12 组通过核验后才能 summarize。仅加载自己本次生成的模型，不要将他人提供的 pickle 放入运行目录。

云端完整工作流：一次准备特征 → 12 个独立配置任务并行训练和核验 → 汇总报告。临时 prepared-features 工件保留 1 天；每组及汇总证据保留 14 天。数据缓存经过 SHA-256 核对后才用于训练。


## 随机森林四组中间值方案（本地探索）

当前四组随机森林调参入口为 `optimize_rf_midpoint.py`，说明见
`rf_midpoint_experiment/使用说明.txt`。它调节 `max_features`，固定 60 棵树、
最大深度 16、最小叶子样本数 2；外层十折、内层三折，每折仍按验证分数最大值选择。
安装依赖后，可用 `python reproduce.py prepare --work-dir .reproduction` 准备特征缓存。

```bash
python optimize_rf_midpoint.py --cache-dir .reproduction/data --output .rf-midpoint --workers 4
```

四组范围不同：Log-Mel clean 使用 `[0.1,0.3,1.0]`，其余三组使用
`[0.01,0.1,0.3]`。本轮中间候选获选次数依次为 5、10、10、10；
Log-Mel clean 的完整次数为 `[2,5,3]`，满足唯一众数，但不严格超过半数。
这些范围来自同一数据的事后筛查，属于探索性结果；范围设计使用了跨折信息，
不能当作独立确认或无偏嵌套交叉验证。完整四点候选均予保留，最终网格没有丢失
本轮四点候选中的逐折最高内层分数。旧 `optimize_rf.py` 保留为深度诊断入口。

## 已验证的云端运行

2026-10-12（悉尼时间），[完整复现 #2](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38144868346) 成功，耗时约 18 分钟（包含资源排队）。被验证代码提交为 14e1eb55dc21a2d858e66c87db80c58b207ecf99；本节与 reproduced_results/ 是随后保存的结果证据。

12 组配置、1,320 次拟合、132 个保存模型核验全部通过，训练记录警告数为 0。全部配置的中间参数都是唯一众数；MFCC LightGBM clean 的计数为 [2,5,3]，中间参数没有严格超过半数。工程 PASS 不表示课程全部要求通过。

结果长期保存在 [reproduced_results/github-actions-38144868346](reproduced_results/github-actions-38144868346/)。云端 Ubuntu/Python 3.12.15 与本机 Windows 重跑相比，48 个“配置×条件”的平均 macro-F1 最大绝对差为 0.00076123，约 0.076 个百分点。跨平台结果接近，但并非逐位一致。

本次 Actions 页面另有 artifact action 的 Node 20 兼容性及 ubuntu-latest 未来迁移提示；运行成功。上面的 0 警告指训练记录，环境版本见保存的 status.json。

