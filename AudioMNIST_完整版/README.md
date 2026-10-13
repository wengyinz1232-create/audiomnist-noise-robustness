# AudioMNIST：实验代码、数据与结果交接

## RF 主训练代码更新（2026-10-13）

`evaluation.py` 中的随机森林现已直接采用验证过的特征采样比例方案：60棵树、最大深度16、最小叶子样本数2，调节 `max_features`。四组实际网格由 `rf_settings.json` 指定：Log-Mel干净训练为 `[0.1,0.3,1.0]`，其余三组为 `[0.01,0.1,0.3]`。
`run_pipeline.py train` 和仓库根目录的 `reproduce.py train/all` 都会使用这些设置；内层仍按实际验证分数最大值选参。
下文2026-10-11的结果索引和归档描述是历史记录，不会因源码更新而自动重算。新训练应使用全新的工作目录或 run-id，不能续接旧深度网格的模型。
本轮范围来自事后探索；Log-Mel两种训练方式的网格不同，不能将全部差异解释为纯增强效应。主流程仍包含额外全数据拟合；已验证的40折RF实验本身没有全数据模型。

## 当前结果入口（2026-10-11）

### 最新调参状态

MFCC LightGBM 已增加 `lightgbm_l2_selected_20261011` 完整替代批次：固定7个叶子和70轮，共用 `reg_lambda=0/0.1/1`，干净/增强训练的十折选择次数为 `2/5/3`、`3/6/1`，两组均通过中间值唯一众数检查。详见 `results/lightgbm_l2_selected_20261011/summary.json`。这是此前探索后的完整重跑，不是独立确认实验，也不保证全部性能提高。
当前十二组视图已采用新逻辑回归和新LightGBM，共十二组均通过该中间值检查；这不等于课程全部要求已验收。原失败批次仍保留。
L2完整重跑入口为 `python rerun_lightgbm_l2.py`；不要在其运行时同时启动 `complete_lightgbm_l2_parallel.py`。所有重跑批次均检查源码指纹，源码变更后不能直接恢复旧批次。新增L2分支前的逻辑回归评估源码已按原哈希保存在 `provenance/pre_l2_snapshot/evaluation.py`。

### 先前选用范围与图表（L2重跑前快照）

`results/selected_experiments/selection.csv` 保留L2重跑前的选用清单：8组共用网格的严格配对、2组分别调参的新逻辑回归、2组单独展示的LightGBM配置。它及对应图表尚未换入新的L2批次，最新数值应查看 `results/current/`。
它不同于保留原十二个位置的 `summary_current_12.csv`；选用清单中的两叶干净LightGBM替代配置不能与原增强模型解释为增强单因素对照。
运行 `python prepare_selected_experiments.py` 生成清单及四张 PNG/PDF 科学图，需先安装 `requirements-plots.txt`。图表底层数值在 `figure_data.csv`，配对差值在 `paired_effects.csv`，限制和未选用实验见 `selection.json`。
图表和清单不代表所有原始配置通过，也不构成课程报告或独立确认实验。

先查看 `results/current/status.json` 和 `results/current/summary_current_12.csv`。
运行 `python build_experiment_index.py` 可从保存结果重建这些文件，不训练模型、不覆盖原 `results/summary.json`。

- `catalog.json`：20 组完整实验的来源、哈希、参数选择次数、用途及限制。
- `summary_current_12.csv`：十二个配置位置，MFCC逻辑回归采用 `logistic_selected_grids_20261011`，MFCC LightGBM采用 `lightgbm_l2_selected_20261011`；原批次保留在完整清单中。
- `summary_all_runs.csv`：全部完整实验的逐条件指标均值和折间样本标准差。
- `paired_comparisons.csv`：同折增强差值，分别标明是否共用网格、双方是否通过中间值检查。新逻辑回归是分别调参后的流程比较。
- `diagnostics.json`：内层诊断来源，含提前停止的轮次实验；不混入外层性能汇总。
- `supplementary.json`：原三模型集成的来源。它仍使用旧逻辑回归，不能将 0.9414 当作新逻辑回归参与后的集成成绩。

新逻辑回归两组已完整重跑并核验，中间值选择次数为 0/10/0、1/8/1。其网格来自此前探索，不是独立确认实验。复现专用入口：`python rerun_logistic_selected_grids.py`。
LightGBM 两叶干净替代配置与原增强配置分别通过检查，但设置不同，不是仅改变增强方式的对照。
`results/summary.json` 是保留的历史主批次汇总。原 `run_pipeline.py summarize` 会覆盖它；查看当前结果请用上述新索引入口。

## 2026-10-10 后续代码补充

`python run_pipeline.py compare` 读取完整的 12 组主批次逐折 JSON，核对折配对与内折选参，输出 `results/comparison/` 中的配置汇总和同折差值；不重新训练。另增加可选的 `lightgbm_goss_min_child` 配置：固定 2 个叶子，调 `min_child_samples=5/20/80`，需在独立 `--run-id alt_min_child` 中运行。它与原主批次使用相同外层说话人划分及选参目标，但它是看过旧结果后设计的探索性复查，**不自动取代主批次**。本次实跑的干净训练中间值被选 6/10 折，增强训练为 3/10 折，因此成对比较仍不能作为全面满足 2.f 的正式结果。原 12 组主批次及其已知限制不变。

还加入了可选的 `multi_snr_augmented` 模式（干净＋白噪声 20 dB＋白噪声 10 dB），以及 `compare_training_modes.py`、`verify_saved_run.py`。在 MFCC＋delta 与 Log-Mel 逻辑回归上各完成一套新的 10×3 实验，均通过选参中间值检查。MFCC＋delta 的粉红噪声 macro-F1 仅从 0.9166 升至 0.9197，Log-Mel 则从 0.8818 降至 0.8783；这不支持把双强度增强宣称为普遍优越。可选 LightGBM 双强度训练只完成两折后因耗时停止，不能作为结果。

`python run_pipeline.py ensemble` 对原主批次三种 **MFCC＋delta、白噪声增强** 模型的同折概率做固定等权平均，不训练权重。对应干净/粉红噪声 macro-F1 为 0.9755/0.9414；比其中最好的单模型对应值分别高 0.0029/0.0248（不同条件下的最佳单模型不同）。`predict_ensemble.py` 可用三个可信的全量模型对单条 WAV 预测。**这是看过主批次结果后添加的补充分析，不是第四种正式调参算法或独立确认实验。**

当前目录包含实验源代码、原始音频、特征缓存、模型、逐折预测和分析结果；它是组内交接目录，不是可整体提交的代码包。早期精简包的描述不再代表当前目录状态。

## 安装环境并查看已有结果

推荐 Python 3.12。在本文件夹打开终端：

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python build_experiment_index.py
```

上述命令只读取已有实验记录生成当前索引。需要重训原方案时使用 `python run_pipeline.py train --run-id rerun_latest --jobs 3`；新逻辑回归两套网格使用 `python rerun_logistic_selected_grids.py`。原始音频已附带，安装依赖需要网络。

当前已经保存 `data/`、`runs/`、`results/`。历史源码与当前代码存在版本差异，不要直接对旧 main 批次执行 all/train/verify；重新训练应使用新 run-id。旧的全流程核验不适合混合版本归档，逐批核验使用 `verify_saved_run.py`。历史源码及核验记录见 `provenance/baseline_snapshot/`。

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

1. 历史资料记载过12套主配置及2套范围复查；当前完整实验清单以 `results/current/catalog.json` 为准，不把未保留的历史复查或内层诊断计为完整外层实验。
2. 历史主批次中，MFCC干净训练的逻辑回归、LightGBM未满足“中间参数最常选”，复查也未解决；不能将这些探索对照声称为全面合规。其他主配置的历史参数审查通过。
3. 新题目和论文“课堂未讲过”的解释仍需组员/教师确认。没有写report或提交。
4. 数据、表示和划分已被探索过，重跑不是独立确认实验；人工合成噪声不能代表所有真实环境。
5. 没有原始波形基线或MFCC无变化率消融，不能单独归因变化率贡献。
6. 全量模型没有独立测试分数；评价应使用外折结果。只加载可信来源的pickle。
7. **不要把含数据的交接包整体上传Canvas**；课程明确禁止提交数据。
