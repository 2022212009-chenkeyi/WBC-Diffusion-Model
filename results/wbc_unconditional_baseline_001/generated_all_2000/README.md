# 全部 2,000 张 DDPM 生成图

- `contact_sheets/`：20 张总览页，每页按样本编号展示 100 张；建议先从这里快速浏览。
- `images/`：全部 2,000 张独立 PNG，编号 `generated_0001.png` 至 `generated_2000.png`。
- 图像来自最终 EMA checkpoint，分辨率 256×256 RGB；索引顺序与 SSD 上原始 NPY 数组一致。
- 结果解释和实验限制见上一级 `analysis.md`。

此目录是便于本地浏览的导出副本；原始数组仍保存在 SSD 的 `samples_ema_step0200000_n2000_seed2026.npy`。
