# Raabin-WBC 无条件 DDPM 基线结果分析

分析日期：2026-11-06
实验：`wbc-raabin-256-baseline-v1`，256×256，EMA checkpoint，200,000 updates
数据：冻结的 Raabin-WBC Train 双标注数据；8,139 train / 1,017 validation / 1,017 test

## 结论摘要

正式训练和 2,000 张生成样本均已完成。当前结果**不支持直接进入条件生成**：通用 Inception 特征空间中的 precision 和 coverage 明显低于真实 train-vs-test / test-vs-test 参照，16 张训练预览中也可见颜色、模糊和多细胞伪影。这个判断仍是“暂缓/no-go”，因为三位评审的正式盲评分数还没有填写，人工质量门槛尚未计算。

训练/验证 loss 下降说明去噪训练目标得到优化，不能单独说明生成图像质量好。以下 FID/KID/PRDC 全部基于通用 TorchMetrics/torch-fidelity Inception-v3 768 维特征；该特征网络不是经 WBC 形态验证的指标，因此只作分布差异证据，不解释为临床或病理真实性。

## 训练结果

| 项目 | 结果 |
|---|---:|
| 训练更新 | 200,000 / 200,000，Slurm 245428 正常结束 |
| 硬件 | NVIDIA RTX PRO 6000 Blackwell Max-Q |
| 训练耗时 | 2 小时 40 分 59 秒 |
| 最大已分配显存 | 约 3.58 GiB |
| 最终验证 epsilon MSE | 0.0041909（固定 seed 2027，全量 1,017 张 validation） |
| 最终 checkpoint | step 200,000 EMA |

最终 validation MSE 相对早期大幅下降，但它衡量的是固定验证噪声任务的 epsilon 预测误差，不是感知质量分数。

## 生成分布与覆盖

最终 EMA checkpoint 以 seed 2026 生成 2,000 张 256×256 RGB 图像。

| 指标 | 生成 vs test | 真实参照 | 解释 |
|---|---:|---:|---|
| FID ↓ | 0.7547 | train vs test：0.0180 | 生成分布与真实 test 的特征差距明显较大 |
| KID mean ↓ | 0.0013485 | train vs test：0.00000133 | 100 个、每个 1,000 张子集的平均 MMD；`std` 是子集间标准差，不是置信区间 |
| PRDC precision ↑ | 0.0865 | test 二分参照：0.8900 | 只有较小比例的生成点落在真实局部流形内 |
| PRDC recall ↑ | 0.5290 | test 二分参照：0.9114 | 生成流形覆盖的真实样本比例偏低 |
| PRDC density ↑ | 0.0447 | test 二分参照：0.9654 | 真实局部邻域中生成点密度很低 |
| PRDC coverage ↑ | 0.2380 | test 二分参照：0.9626 | 真实样本被生成流形覆盖的比例很低，提示明显覆盖不足/模式收缩 |

这些数字在同一通用特征空间和同一预处理下相较真实参照差距很大，是当前暂缓条件生成的主要定量依据。它们不能单独证明白细胞结构是否正确；仍需盲评，并结合预览图及近邻复核解释。

## 记忆化筛查

- 对 2,000 张生成图与全部 8,139 张 train 图按固定 RGB/Lanczos 256×256 处理后，精确像素 SHA-256 命中：**0**。
- 64-bit dHash 汉明距离 ≤4 候选：**0**。
- 另导出每张生成图在通用 Inception 特征空间中的 top-5 train 近邻。top-1 距离中位数 0.8628，5th–95th percentile 为 0.5761–1.3056；目前没有经校准的 WBC 距离阈值或 real-test-to-train 距离对照，不能仅凭这些距离断言复制或排除记忆化。
- 因此当前结论是“这两种哈希检查未发现命中”，不是“已证明不存在记忆化”。对特征近邻的进一步可视复核仍可加强证据。

## 图像质量盲评与当前 go/no-go

为方便直接查看，已从 2,000 张结果中用固定选择 seed 抽出 100 张，导出单张 PNG 和总览图。对总览的非盲初步观察是：部分图像能看到 WBC 样结构，但也有不少明显色偏、模糊、细胞过多或结构混杂的图像；不据此估算通过比例，正式质量判断留给盲评。

盲评包已为 3 位评审各准备 200 张生成图 + 200 张真实 test 图，随机混排并隐藏来源。校准包另含 20 张 train 图，不进入盲评计分。三位评审的正式 `scoring_sheet.csv` 目前仍为空，因此以下预登记人工门槛都**尚未判定**：

- 至少 80% 生成图被 ≥2/3 评审评为可辨认为 WBC（评分 ≥4）；
- 至少 80% 生成图被 ≥2/3 评审评为单细胞主体完整（评分 ≥4）；
- 严重伪影图像比例 ≤15%（≥2/3 评审给反向伪影严重度评分 ≤2）。

**当前决定：暂缓进入条件生成。** 现有机器指标显示很低的 precision/coverage，且训练预览显示可见伪影；同时人工盲评和特征近邻复核尚缺。下一步先收齐盲评表、解盲并运行 `ddpm.summarize_wbc_blind_review`，再结合近邻复核完成正式 go/no-go。若门槛失败，应先排查采样/训练配置并改进无条件基线。

## 结果文件

- 指标与训练曲线诊断图：[diagnostics.svg](diagnostics.svg)；可用浏览器或 VS Code 打开。
- 生成图 100 张预览：[总览图](generated_preview_100/contact_sheet.jpg)、[单张 PNG 文件夹](generated_preview_100/images/)；对应索引见 `generated_preview_100/manifest.csv`。
- 训练日志：`/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/train.csv`、`validation.csv`
- 2,000 张生成图：`/projects/keyi005ssd/WBCDiffusion/generated_samples/wbc-raabin-256-baseline-v1/samples_ema_step0200000_n2000_seed2026.npy`（约 376 MiB，uint8 NHWC）
- FID/KID/PRDC JSON：`/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/evaluation/fid_kid_step200000_n2000.json`
- 精确哈希/dHash 报告与 CSV：`/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/evaluation/memorization/`
- 特征近邻 top-5：`/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/evaluation/generated_feature_nearest_train_step200000.csv`
- 盲评材料：`/projects/keyi005ssd/WBCDiffusion/results/wbc-raabin-256-baseline-v1/evaluation/blind_review_v1/`

## 限制

1. 通用 ImageNet-derived Inception 特征不是 WBC 领域验证过的感知指标。
2. 本报告没有估计 FID/KID bootstrap 置信区间；KID 的 `std` 仅是随机子集间标准差。
3. 患者/玻片 ID 不可用，冻结拆分只控制已识别重复组，不代表患者级独立。
4. 人工盲评未完成，不能把机器结果或训练 loss 描述成专家认可、临床真实或病理有效。
