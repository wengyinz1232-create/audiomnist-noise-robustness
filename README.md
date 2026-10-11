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
