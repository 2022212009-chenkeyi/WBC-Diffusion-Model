zai# WBCDiffusion Research Plan v2.0

WBCDiffusion Research Plan v2.0
High-Quality Cell Image Generation · NTU MSc Research Roadmap
## 1. Overall Research Roadmap

## 2. Research Habits — 每一步都必须执行

实验记录: 每个正式实验建立 Experiment ID，并记录日期、数据集、模型、超参数、GPU、seed、训练时间、结果和观察。
控制变量: 一次实验尽量只改变一个核心变量；否则不能把结果差异归因到某个因素。
Baseline 意识: 任何新模型/新方法都先明确比较对象，优先与 WBC-DDPM baseline 对比。
Hypothesis: 实验前先写清楚假设：我认为改变 X 会导致 Y，原因是什么。
Evidence: 不要只凭生成图片下结论；结合 FID、KID、diversity、morphology、速度、显存等证据。
Negative Results: 失败实验也要记录；重点分析为什么失败，而不是删除失败记录。
Paper → Equation → Code → Experiment: 读论文时建立方法、公式、代码和实验之间的对应关系。
Git / Version Control: 代码通过 Git 管理，commit 信息清晰，避免 final.py / final2.py 等不可追踪文件。
Data Management: Raw data 不修改；processed data、metadata、checkpoints、results 分开管理。
Reproducibility: 固定并记录 random seed、环境、依赖版本、配置文件和运行命令。
Weekly Research Summary: 每周记录：做了什么、成功/失败、学到了什么、证据是什么、剩余问题、下一步。
Mentor Communication: 与导师沟通前准备 What I did / What I found / What I need。
Research Question: 每个实验都问：这个实验到底证明了什么？下一步问题是什么？
## 3. Mandatory Experiment Template

1. Why? — 为什么做这个实验？
2. Hypothesis — 我预测什么？
3. Setup — 数据、模型、超参数、GPU、seed 等是什么？
4. Experiment — 如何实施？
5. Record — 如何记录配置与日志？
6. Result — 得到了什么定量/定性结果？
7. Analysis — 为什么出现这个结果？
8. Conclusion — 这个实验真正证明了什么？
9. Next Experiment — 根据证据下一步做什么？
## 4. Project Structure

WBCDiffusion/
├── 00_environment/
├── 01_ddpm_reproduction/
├── 02_dinobloom_analysis/
├── 03_wbc_ddpm/
├── 04_ddim/
├── 05_ldm/
├── 06_dit/
├── 07_jit/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── checkpoints/
├── experiments/
├── results/
├── generated_samples/
├── scripts/
├── paper_notes/
└── README.md
## 5. Storage Strategy — NTU GPU Cluster

SSD: 代码、Git repository、Conda environment、常用依赖/cache、训练时频繁读写的配置/日志、当前活跃 checkpoint、实验结果等对 I/O 速度敏感的内容。
HDD: 原始数据集、数据集归档、大型长期保存数据、较少访问的历史结果和归档 checkpoint。
Home: 只保留必要的个人配置和极少量小文件；不放 dataset、checkpoint、Conda environment 等大文件。
重要：具体 SSD/HDD 用户目录必须先确认 NTU EEE Cluster 的实际 storage allocation 和挂载路径，不能根据目录名猜测，也不能直接在 /projects/_ssd 或 /projects/_hdd 下创建用户目录。
## 6. Environment Strategy

ddpm-original：用于 Ho et al. 原始 DDPM 代码与旧依赖；不要与现代 WBC 环境混用。
wbcdiffusion：Python 3.11 + modern PyTorch，用于 WBC、DDPM、DDIM、LDM、DiT、JiT。
当前 wbcdiffusion 已建立在 Home 中，但正式科研环境应迁移到确认后的 SSD；storage allocation 未确认前不要删除现有 Home 环境。
## 7. Phase 0 Checklist

□ 确认 SSD / HDD 实际 allocation 与用户目录
□ 建立 SSD 项目目录
□ 建立 HDD 数据目录
□ 确认 WBCDiffusion 代码位置
□ 将/重新创建 wbcdiffusion 到 SSD
□ 验证 Python / PyTorch / CUDA / GPU
□ 建立 Git repository
□ 建立 experiment logging 规范
□ 写下 Phase 0 Research Summary
## 8. Researcher Training Rule

从 Phase 0 开始，每推进一个步骤，都必须回答：我现在做什么？为什么做？我记录了什么？这一步如何保证可复现？这一步完成后获得了什么证据？下一步为什么这样做？
长期目标：提出问题 → 阅读文献 → 设计实验 → 控制变量 → 获取证据 → 分析结果 → 形成结论 → 与导师讨论 → 再提出问题。

<!-- Table 1 -->

| 阶段 | 内容 | 阶段目标 |
| --- | --- | --- |
| Phase 0 | GPU / Environment / Project Structure | 建立稳定、可复现且不依赖 Home 的科研环境 |
| Phase 1 | DDPM Original Reproduction | 理解并复现 DDPM 核心流程；先 CIFAR-10，再扩展 |
| Phase 2 | DinoBloom Dataset Analysis | 分析 WBC 数据分布，并构建约 10K 子集 |
| Phase 3 | WBC-DDPM Baseline | 建立 WBC 图像生成 baseline |
| Phase 4 | DDIM | 比较采样速度、质量与 diversity |
| Phase 5 | LDM | 比较 latent-space diffusion 与 pixel-space DDPM |
| Phase 6 | DiT | 研究 Transformer-based diffusion |
| Phase 7 | JiT | 在 baseline 稳定后研究新一代 denoising generative model |
| Phase 8 | WBC-specific Research Question | 围绕 morphology / diversity / imbalance / controllability 等形成研究问题 |
