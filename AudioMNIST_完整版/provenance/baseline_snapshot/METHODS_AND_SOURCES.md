# 方法和来源：供队友理解、复现，不是 report

交接包已按用户要求精简：下文提及的缓存、模型、逐折结果均需运行代码后生成，目前不随包附带。关于历史选参结果的说明仅保留实验设计依据和已知限制，不代表对应结果文件仍在目录中。

## 声音 → 特征

源文件是单声道 48 kHz、16-bit PCM WAV。代码以 32768 缩放为波形浮点数，使用 `scipy.signal.resample_poly` 重采样到 16 kHz，末尾补零到 1 秒。每条产生 16,000 个采样点。此步骤不借助其他录音或标签。

短时傅里叶变换：Hann 窗长 400 点（25 ms），重叠 240 点（步长 10 ms），FFT 512 点，不额外边界填充，得到 98 个时间帧。幅度平方得到功率。32 个三角 Mel 频带覆盖 20–8000 Hz，每个滤波器权重之和归一为 1；采用 `2595 log10(1+f/700)` 的 Mel 刻度，对能量取自然对数（下限 1e-12）。这不是 librosa 默认参数的逐位复刻。

- **Log-Mel 256 维**：按帧顺序切成 25/25/24/24 帧四段；每段对 32 频带计算均值、总体标准差（ddof=0）。32 × 4 × 2 = 256。
- **MFCC＋delta 320 维**：在尚未池化的逐帧 Log-Mel 上做正交归一的 II 型 DCT，取第 0–19 个系数。用 `numpy.gradient` 沿时间计算一阶差分（内部用前后帧中心差分，端点单边差分，单位为每帧，非每秒）。20 系数＋20 delta，再同样 4 段 × 2 统计量 = 320。

两者是替代输入表示，不是 576 维拼接，不是从最终 256 维统计向量再生成 MFCC。差分不等于标准差：差分描述局部方向/速度，标准差概括一段内数值的离散程度。所有列名与真正数组顺序匹配，见 `data/feature_names.json`。

经典方法来源（不是声称本项目发明）：

- Davis & Mermelstein (1980), *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*. https://doi.org/10.1109/TASSP.1980.1163420 。用于 MFCC 方法背景，不用来满足 2017 年后的困难模型要求。
- 官方可对照实现文档：https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html 与 https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html 。本项目实际使用 NumPy/SciPy 组合实现，未调用 librosa。

## 噪声与训练干预

white_20 / white_10 为合成高斯白噪声，SNR 分别 20/10 dB。pink_10 对白噪声的频域幅度除以频率索引平方根，再反变换、缩放，使功率谱大致随 1/f 变化。SNR 按补齐后的整个录音 RMS 定义。生成后不额外截幅或归一化。

原始录音、白噪声版本、粉红噪声版本共享源样本 ID 和说话人。先根据说话人指定折，再仅索引对应人员的版本；即使预先批量计算缓存也没有跨样本学习操作，不会因此造成标签泄漏。

两种训练模式的 inner validation 都只看 clean 和 white_10 两个条件的 macro-F1 平均值。clean_matched 只训练干净录音；augmented 加上同一训练录音的 white_10 副本。二者参数网格相同。white_20 和 pink_10 均不参与选参；pink_10 从未作为训练输入。无法把 clean_matched 描述为“从未接触噪声信息”：它的拟合没用噪声，但参数选择参考了白噪声验证表现。

## 模型、复杂度和调参

|模型|角色|固定设置|主批次调参范围|
|---|---|---|---|
|Multinomial logistic regression|简单：线性分类边界|L2、lbfgs、最多2000轮、tol=1e-3|C：0.0001 / 0.1 / 100|
|Random forest|中等：并行自助采样树的非线性集成|60 棵树、最小叶样本2|最大深度：4 / 48 / 128|
|LightGBM-GOSS|困难：逐步优化的梯度提升树与 GOSS|70 轮、学习率0.08、min_child_samples=20、L2=1、top_rate=.2、other_rate=.1|叶子数：3 / 63 / 255|

这是模型机制/实现层次的三档解释，不是严格证明所有随机森林的假设空间都小于所有提升树。困难模型不是通过调大同一种模型参数凑出来的。

困难模型来源：Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NIPS 2017 主会，非 workshop。
https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

本次显式设置 `data_sample_strategy='goss'`，保留大梯度样本并对子采样的小梯度样本补偿。论文也提出 EFB，但本项目是稠密声音特征，不声称验证了 EFB 收益，也不声称复现整篇论文实验。官方实现参数：https://lightgbm.readthedocs.io/en/stable/Parameters.html 。

参数范围参考旧探索，以及本目录中断批次的内折选择记录后重新指定，没有把旧外折模型或结果复制进来。每个范围都必须经实际众数审查；失败时保留原批次，以新批次重定网格，不手工改选参。不把网格探索解释为独立预注册。树深度上限超过实际树深度时可能形成平台；中间值胜出若包含同分，应同时解释平台，而不能宣称该值显著更优。

### 额外范围检查

主批次 MFCC＋delta / logistic / clean_matched 的内折结果在10折中有8折选最大 C=100，因此另开 `mfcc_logistic_range_check` 批次，比较 C=0.1/100/10000，完整重做10×3与全量拟合。原主批次不覆盖。该批次仅检查参数边界，不取代共同网格下的增强配对实验，也不能把它与原 augmented 模式之差宣称是完全隔离的增强收益。实际是否中间最常选以结果为准。

本次范围检查的实际结果为 C=100、10000 各选5折，没有中间唯一众数。两者内折均分差大多很小，但不能因此改写成符合要求。这组 MFCC 干净训练应保持探索标签；正式主方案优先看 Log-Mel 三模型的完整参数审查。没有继续反复调整候选集来制造通过记录。

主批次 MFCC＋delta / LightGBM / clean_matched 的10折全部选择最小叶子数3，因此另开 `mfcc_lightgbm_range_check`，使用合法的相邻小范围2/3/7，完整重做10×3与全量拟合。这同样是单独参数范围检查，不替换主批次的匹配增强比较。

该复查实际10/10折选择叶子数2，已经达到该参数的合法最小值，仍不满足中间值要求。不能用非法叶子数1作下界来制造中间值。若未来要将这项 MFCC 干净训练变为正式方法，需要另行选择有意义的调参维度，并重新跑验证；本次保留为已训练的探索对照，而不宣称全部配置都满足2.f。

## 验证、标准化、指标和保存

外层 10 折：每折 54 位训练人员（10,800 录音）、6 位测试人员（1,200 录音）。内层 3 折：在这 54 人内进一步按人切分。全部配置用同一划分种子 20261009。增强版本始终跟随其原始说话人，绝不跨折。

每个内折训练集单独计算均值/标准差；验证集只使用这些参数变换。最终外折重训时重新拟合标准化。不得使用整个 12,000 样本共同标准化再切分。

手写切分循环、超参数枚举、混淆矩阵和以下指标：accuracy、macro precision、macro recall（balanced accuracy）、macro-F1、log loss、每类 precision/recall/F1。优化指标为两个验证条件的平均 macro-F1，参数同分选列表中第一个低复杂度值。外折结果保存均值和样本标准差（ddof=1）；折间训练集重叠，所以 SD 不是独立样本置信区间。

另用全部 60 人做三折选参，再训练全量模型，存为 fold_00.pkl。它是可复用产物而非新的评估结果。每折模型保存后重新加载，确认预测一致；逐样本预测概率保留，可重算全部指标。

## 数据来源与许可

AudioMNIST 官方：https://github.com/soerenab/AudioMNIST 。固定归档版本 `630d7dab4c040882834de6fa21baf9a60372accd`，原始许可随包保留。官方 README 给出的数据论文：Becker et al., *AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark*, DOI https://doi.org/10.1016/j.jfranklin.2023.11.038 。请按出版页面最终书目信息整理报告引用，不将 GitHub README 的年份与正式卷期信息混为一谈。

12,000 条为真实不同录音，不把人工加噪当成扩大独立样本数；原始录音索引0–19的选择保留录制顺序偏差可能。重跑与旧探索共享人员和录音，不是新的独立测试数据。
