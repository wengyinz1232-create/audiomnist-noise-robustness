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

当前 12 组 = 2 特征 × 3 类模型 × 2 训练方式。说话者隔离的外层 10 折、内层 3 折，共 1,320 次拟合、120 外层模型和 12 全数据模型。

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


## 已验证的云端运行

2026-10-12（悉尼时间），[完整复现 #2](https://github.com/wengyinz1232-create/audiomnist-noise-robustness/actions/runs/38144868346) 成功，耗时约 18 分钟（包含资源排队）。被验证代码提交为 14e1eb55dc21a2d858e66c87db80c58b207ecf99；本节与 reproduced_results/ 是随后保存的结果证据。

12 组配置、1,320 次拟合、132 个保存模型核验全部通过，训练记录警告数为 0。全部配置的中间参数都是唯一众数；MFCC LightGBM clean 的计数为 [2,5,3]，中间参数没有严格超过半数。工程 PASS 不表示课程全部要求通过。

结果长期保存在 [reproduced_results/github-actions-38144868346](reproduced_results/github-actions-38144868346/)。云端 Ubuntu/Python 3.12.15 与本机 Windows 重跑相比，48 个“配置×条件”的平均 macro-F1 最大绝对差为 0.00076123，约 0.076 个百分点。跨平台结果接近，但并非逐位一致。

本次 Actions 页面另有 artifact action 的 Node 20 兼容性及 ubuntu-latest 未来迁移提示；运行成功。上面的 0 警告指训练记录，环境版本见保存的 status.json。

