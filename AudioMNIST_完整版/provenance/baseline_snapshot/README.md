# AudioMNIST：代码与原始数据交接包

本目录只保留实验源代码、原始音频及必要说明/课程原文。**不附带训练模型、特征缓存、预测、结果图表、日志或中断批次。**这些产物由队友运行后重新生成。旧 APS 和候选筛选目录未改动。

## 一次运行完整流程

推荐 Python 3.12。在本文件夹打开终端：

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_pipeline.py all --jobs 3
```

主入口是 `run_pipeline.py`。上述命令自动读取原始录音、计算特征、完成手写10×3嵌套交叉验证、调参和训练，再核验并汇总结果。无需其他项目文件夹，无需重新下载音频；安装依赖需要网络。

运行后会生成 `data/`（特征）、`runs/`（模型/逐折预测）、`results/`（汇总），以及 `TRAINING_RESULTS.md`、`VERIFICATION.json`、`RUN_MANIFEST.json`。**现在没有这些文件，是因为交接包已精简，不代表历史实验没跑过。**完整运行会重新占用磁盘空间。

## 实验内容

- 研究问题：未见说话人的数字识别中，声音表示如何影响抗噪能力？白噪声增强的收益能否迁移到未用于训练或选参的粉红噪声，是否损害干净声音表现？
- 原始 AudioMNIST：60人、10个数字、共30,000条真实录音。完整归档 `data_raw/audiomnist.zip` 约995 MB；代码固定选每人每数字编号0–19，共12,000条。人工噪声副本不计独立样本。
- 两套特征：Log-Mel 256维；MFCC＋一阶变化率320维，由原始WAV计算。
- 三模型：逻辑回归、随机森林、LightGBM-GOSS。
- 两种训练方式：只训练干净录音、干净＋白噪声10dB增强。两者均按干净/白噪声验证macro-F1平均值选参；粉红噪声仅测试。
- 主流程12套配置，每套10外折×3内折，按说话人隔离；共1,320次拟合，包括另行调参训练的全量模型。

## 保留的文件

|文件|用途|
|---|---|
|`run_pipeline.py`|唯一主入口|
|`prepare_data.py`、`features.py`|读取录音、加噪、计算特征|
|`evaluation.py`|手写切分、调参、指标；调用分类器拟合|
|`verify.py`、`summarize.py`|核验和结果汇总|
|`predict.py`、`check_prediction_cli.py`|训练后加载可信模型、检查WAV预测入口|
|`plot_training.py`、`package_code.py`|可选绘图、生成无数据代码ZIP|
|`requirements*.txt`、`Readme.txt`|依赖与简短启动说明|
|`METHODS_AND_SOURCES.md`|特征算法、模型设置、来源和复查依据|
|`REQUIREMENTS_CHECKLIST.md`、`CLASSROOM_SOURCE_CHECK.md`|课程要求与待确认事项|
|`data_raw/`|原始音频、原始许可证与来源说明|
|`assignment_sources/`|课程原文、教授反馈，不是实验生成物|

## 分步与可选运行

安装依赖后，可以在当前Python环境执行：

```text
python run_pipeline.py prepare
python run_pipeline.py train --jobs 3
python run_pipeline.py verify
python run_pipeline.py summarize
```

- 相同配置可断点恢复；代码或数据指纹不一致会拒绝混用。
- 原始归档缺失时：`python run_pipeline.py prepare --download`，从固定版本下载。
- 绘图：安装 `requirements-plots.txt`，再运行 `python plot_training.py`。
- 预测接口检查：完成训练后运行 `python check_prediction_cli.py`。例子属于已用数据，只检查接口，不是独立测试准确率。
- 无数据代码包：`python package_code.py`。只在本地打包，不会上传。

历史上的两套参数范围复查不属于默认all命令。如需复现：

```text
python run_pipeline.py train --run-id mfcc_logistic_range_check --representations mfcc_delta --models logistic --modes clean_matched --grid 0.1 100 10000 --jobs 1
python run_pipeline.py train --run-id mfcc_lightgbm_range_check --representations mfcc_delta --models lightgbm_goss --modes clean_matched --grid 2 3 7 --jobs 1
python run_pipeline.py verify
python run_pipeline.py summarize
```

## 必须保留的限制说明

1. 精简前曾完成12套主配置和2套复查、1,540次拟合，工程核验通过。产物现已移除，新运行以新生成记录为准。
2. 历史主批次中，MFCC干净训练的逻辑回归、LightGBM未满足“中间参数最常选”，复查也未解决；不能将这些探索对照声称为全面合规。其他主配置的历史参数审查通过。
3. 新题目和论文“课堂未讲过”的解释仍需组员/教师确认。没有写report或提交。
4. 数据、表示和划分已被探索过，重跑不是独立确认实验；人工合成噪声不能代表所有真实环境。
5. 没有原始波形基线或MFCC无变化率消融，不能单独归因变化率贡献。
6. 全量模型没有独立测试分数；评价应使用外折结果。只加载可信来源的pickle。
7. **不要把含数据的交接包整体上传Canvas**；课程明确禁止提交数据。
