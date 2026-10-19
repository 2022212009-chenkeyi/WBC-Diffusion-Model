# DDPM 核心数学流程：11 个 Steps

> **项目：WBCDiffusion — High-Quality Cell Image Generation**  
> **论文：Denoising Diffusion Probabilistic Models (DDPM)**  
> **用途：DDPM 理论学习笔记 / 后续代码实现参考**

---

## Step 1 — 定义噪声强度：βₜ

DDPM 首先定义 variance schedule（方差/噪声调度）：

β₁, β₂, …, βₜ, …, β_T

其中 βₜ 控制第 t 个 timestep 加入多少噪声。

原论文实验使用线性 schedule：

β₁ = 10⁻⁴，β_T = 0.02

因此：

**βₜ = noise strength at timestep t**

---

## Step 2 — 定义 αₜ

为了简化公式：

αₜ = 1 − βₜ

其中：

- αₜ：当前 timestep 保留的 signal 比例
- βₜ：当前 timestep 加入的 noise 比例

---

## Step 3 — 定义累计系数 ᾱₜ

ᾱₜ = ∏ₛ₌₁ᵗ αₛ

即：

ᾱₜ = α₁ α₂ … αₜ

含义：

**ᾱₜ = 从第 1 步到第 t 步累计保留的 signal。**

注意：

- αₜ：只描述第 t 步
- ᾱₜ：描述前 1 到 t 步的累计效果

---

## Step 4 — Forward Process（前向扩散过程）

单步前向过程：

q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ)xₜ₋₁, βₜI)

利用 αₜ = 1−βₜ：

q(xₜ | xₜ₋₁) = N(xₜ; √αₜ xₜ₋₁, (1−αₜ)I)

等价的 reparameterization（重参数化）形式：

xₜ = √αₜ xₜ₋₁ + √(1−αₜ) ε

其中：

ε ~ N(0, I)

直观理解：

**当前 noisy image = 一部分上一时刻 image + 一部分 Gaussian noise**

---

## Step 5 — 整个 Forward Markov Chain

因为前向过程是 Markov Chain（马尔可夫链）：

x₀ → x₁ → … → x_T

满足：

q(xₜ | xₜ₋₁, …, x₀) = q(xₜ | xₜ₋₁)

所以：

q(x₁:T | x₀) = ∏ₜ₌₁ᵀ q(xₜ | xₜ₋₁)

其中：

- x₀：真实数据
- x₁,…,x_T：逐步加噪的数据
- x_T：最终接近 Gaussian noise

---

## Step 6 — 任意 timestep 直接得到 xₜ

DDPM 可以直接得到：

q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1−ᾱₜ)I)

因此：

**xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε**

其中：

ε ~ N(0, I)

这是 DDPM 实现中最重要的 forward formula 之一。

---

## Step 7 — Reverse Process（反向去噪过程）

前向：

x₀ → x₁ → … → x_T

由我们人为规定的 q 表示。

反向：

x_T → xₜ₋₁ → … → x₀

需要通过神经网络学习：

p_θ(x₀:T) = p(x_T) ∏ₜ₌₁ᵀ p_θ(xₜ₋₁ | xₜ)

其中：

p(x_T) = N(0, I)

也就是说，生成时从 Gaussian noise 开始。

---

## Step 8 — 神经网络预测 Noise

DDPM 采用 epsilon prediction（噪声预测）：

ε_θ(xₜ, t) ≈ ε

训练输入：

(xₜ, t)

训练目标：

ε

其中真实噪声 ε 是我们在 Step 6 中自己采样得到的，因此训练时知道 ground-truth noise。

---

## Step 9 — Training Objective（训练目标）

核心简化目标：

L_simple = E_{t,x₀,ε}[ || ε − ε_θ(xₜ,t) ||² ]

即：

**Loss = MSE(true noise, predicted noise)**

对应代码：

```python
noise = torch.randn_like(x0)

xt = (
    sqrt_alpha_bar_t * x0
    + sqrt_one_minus_alpha_bar_t * noise
)

pred_noise = model(xt, t)

loss = mse_loss(pred_noise, noise)
```

---

## Step 10 — Sampling（采样/生成）

训练完成以后，不再需要真实图片 x₀。

首先：

x_T ~ N(0, I)

然后反复执行反向去噪：

x_T → x_T−1 → x_T−2 → … → x₁ → x₀

每一步使用：

ε_θ(xₜ,t)

预测当前图像中的噪声，再根据 DDPM reverse transition 得到 xₜ₋₁。

---

## Step 11 — 最终得到生成图像

完整 reverse process：

x_T → x_T−1 → … → x₁ → x₀

最终：

**x₀ = Generated Image**

对于 WBCDiffusion：

**x₀ = Generated WBC / Cell Image**

因此整个 DDPM 可以压缩成：

x₀ --add noise--> x_T

x_T --denoise--> x₀_generated

---

# 一张总流程图

```text
                 TRAINING
                    │
                    ▼
              Real Image x₀
                    │
              randomly choose t
                    │
                    ▼
              Sample ε ~ N(0,I)
                    │
                    ▼
     xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε
                    │
                    ▼
          ┌──────────────────┐
          │      U-Net       │
          │ εθ(xₜ, t)        │
          └──────────────────┘
                    │
                    ▼
          Predicted Noise
                    │
                    ▼
       MSE(ε, εθ(xₜ,t))
                    │
                    ▼
                 Update θ


                 SAMPLING
                    │
                    ▼
              xT ~ N(0,I)
                    │
                    ▼
          U-Net predicts noise
                    │
                    ▼
              Denoise → xT-1
                    │
                    ▼
          U-Net predicts noise
                    │
                    ▼
              Denoise → xT-2
                    │
                    ▼
                   ...
                    │
                    ▼
                    x₀
                    │
                    ▼
          Generated Cell Image
```

# 公式之间的逻辑关系

βₜ

↓

αₜ = 1 − βₜ

↓

ᾱₜ = ∏ₛ₌₁ᵗ αₛ

↓

xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε

↓

ε_θ(xₜ,t) ≈ ε

↓

L_simple = E[||ε − ε_θ(xₜ,t)||²]

↓

x_T ~ N(0,I) → x₀_generated

---

# WBCDiffusion 中的对应关系

| DDPM | WBCDiffusion |
|---|---|
| x₀ | 真实 WBC / cell image |
| xₜ | 加噪后的 cell image |
| x_T | 接近纯 Gaussian noise 的 cell image |
| ε | 人工采样的 Gaussian noise |
| ε_θ | U-Net 预测的 noise |
| t | diffusion timestep |
| βₜ | noise schedule |
| L_simple | DDPM training loss |
| reverse process | 从 noise 生成 WBC image |
| x₀_generated | 最终生成的 WBC image |

---

# 论文学习提醒

这一页解决的是：

**DDPM 的数学主线是什么？**

后续还要继续理解：

- 为什么 reverse process 可以写成 Gaussian？
- q(xₜ₋₁ | xₜ, x₀) 怎么推出来？
- ELBO 是什么？
- KL divergence 为什么出现？
- 为什么复杂的 variational objective 最终变成 MSE？
- 为什么网络输入必须包含 timestep t？
- U-Net 如何预测 ε？

这些属于后续的 **DDPM Training Objective 推导**。

---

# 推荐存放位置

建议放在：

```text
/projects/keyi005ssd/WBCDiffusion/paper_notes/DDPM/
```

推荐目录结构：

```text
paper_notes/
└── DDPM/
    ├── 01_DDPM_core_math_11_steps.md
    ├── 02_DDPM_training_objective.md
    ├── 03_DDPM_sampling.md
    ├── 04_DDPM_experiments.md
    └── README.md
```

原则：

**paper_notes/** 保存论文阅读与理论笔记；  
**01_ddpm_reproduction/** 保存复现代码、配置、日志和实验结果。

这样可以把“论文为什么这样做”和“我实际上怎么做、结果怎么样”分开。
