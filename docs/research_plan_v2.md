# WBCDiffusion Research Plan v2.0

**High-Quality Cell Image Generation · NTU MSc Research Roadmap**

## 1. Overall research roadmap

| Phase | Focus | Goal |
|---|---|---|
| 0 | GPU, environment, and project structure | Establish a stable, reproducible research environment that does not depend on Home for large or active research files. |
| 1 | Original DDPM reproduction | Understand and reproduce the core DDPM workflow, first on CIFAR-10 and then on WBC data. |
| 2 | DinoBloom dataset analysis | Analyze WBC data distributions and construct a documented subset of about 10K images. |
| 3 | WBC-DDPM baseline | Establish a reproducible WBC image-generation baseline. |
| 4 | DDIM | Compare sampling speed, quality, and diversity. |
| 5 | LDM | Compare latent-space diffusion with pixel-space DDPM. |
| 6 | DiT | Study Transformer-based diffusion. |
| 7 | JiT | Study newer denoising generative models after the baseline is stable. |
| 8 | WBC-specific research question | Develop a focused question around morphology, diversity, imbalance, controllability, or another evidence-supported issue. |

## 2. Required research habits

- **Experiment records:** Give every formal experiment a unique ID. Record date, dataset, model, hyperparameters, GPU, seed, training time, results, and observations.
- **Control variables:** Change one core variable at a time where possible. If multiple variables change, record the reason and avoid attributing the outcome to one factor.
- **Baseline:** Name the comparison baseline before testing a new model or method; prefer the WBC-DDPM baseline.
- **Hypothesis:** Before running, state what change X is expected to do to outcome Y and why.
- **Evidence:** Do not draw conclusions from sample images alone. Use appropriate quantitative and qualitative evidence, including FID, KID, diversity, morphology, speed, and memory use.
- **Negative results:** Preserve failed runs and analyze plausible causes; do not erase inconvenient outcomes.
- **Paper → Equation → Code → Experiment:** Map each paper's method and equations to the implementation and the experiment that evaluates it.
- **Git:** Track code with clear commits. Avoid untraceable names such as `final.py` and `final2.py`.
- **Data management:** Never modify raw data. Keep processed data, metadata, checkpoints, and results separate.
- **Reproducibility:** Fix and record random seeds, environment and dependency versions, config files, and exact run commands.
- **Weekly summary:** Record what was done, what succeeded or failed, what was learned and its evidence, open questions, and next steps.
- **Mentor communication:** Prepare “What I did / What I found / What I need” before meeting with the mentor.
- **Research question:** For every experiment, ask what it demonstrates and what question follows from it.

## 3. Mandatory experiment template

Use [`experiments/experiment_template.md`](../experiments/experiment_template.md) for every formal run. It covers:

1. Why — Why run this experiment?
2. Hypothesis — What is predicted, and why?
3. Setup — Dataset, model, hyperparameters, GPU, seed, and software environment.
4. Experiment — Exact procedure and command.
5. Record — Config, logs, and artifact locations.
6. Result — Quantitative and qualitative outcomes.
7. Analysis — Why might these outcomes have occurred?
8. Conclusion — What does the evidence support?
9. Next experiment — What should be changed or investigated next, based on evidence?

## 4. Project structure

```text
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
```

The current learning scripts remain at the repository root until they are deliberately organized. Do not put large dataset or checkpoint files in Git.

## 5. Storage strategy — NTU GPU Cluster

- **SSD:** Code and Git repository, Conda environment, frequently used dependencies/cache, active configs and logs, active checkpoints, and I/O-sensitive experiment outputs.
- **HDD:** Raw datasets, dataset archives, large long-term data, infrequently accessed historical results, and archived checkpoints.
- **Home:** Necessary personal configuration and a small number of small files only; do not store datasets, checkpoints, or Conda environments there for formal work.

**Allocation gate:** Confirm the actual SSD/HDD allocation and user paths from NTU EEE Cluster documentation or the account administrator before creating storage directories or moving data. Do not guess from names such as `/projects/_ssd` or `/projects/_hdd` and do not create user directories there based on those names alone.

## 6. Environment strategy

- `ddpm-original`: isolated environment for Ho et al.'s original DDPM code and older dependencies. Keep it separate from the modern WBC environment.
- `wbcdiffusion`: Python 3.11 and modern PyTorch for WBC, DDPM, DDIM, LDM, DiT, and JiT work.
- The current `wbcdiffusion` environment was established under Home. Once the SSD allocation is confirmed, migrate or recreate the formal environment there and record its package lock/export.
- Do not remove the current Home environment until the new environment has been verified.

## 7. Phase 0 checklist

- [ ] Confirm actual SSD/HDD allocation and user directory paths with an authoritative cluster source.
- [ ] Create the SSD project directory after its path is confirmed.
- [ ] Create the HDD data directory after its path is confirmed.
- [ ] Confirm the WBCDiffusion code location and Git remote/backup plan.
- [ ] Migrate or recreate `wbcdiffusion` on confirmed SSD storage; retain the Home environment until verification.
- [ ] Record Python, PyTorch, CUDA, driver, and GPU details from the cluster.
- [ ] Initialize/verify Git and make a clean baseline commit.
- [ ] Adopt Experiment IDs, configuration capture, log/artifact naming, and the experiment template.
- [ ] Write a Phase 0 research summary with evidence, unresolved items, and next steps.

### Phase 0 evidence to retain

Record the authoritative source and confirmed paths; directory ownership and available capacity; environment export; `python --version`; PyTorch version and CUDA availability; GPU model; repository commit; and the exact commands used. Redact account credentials and secrets from logs.

## 8. Researcher training rule

At each step, answer: What am I doing? Why? What did I record? How is this reproducible? What evidence did this step produce? Why is the next step justified?

Long-term loop: pose a question → read literature → design a controlled experiment → gather evidence → analyze results → form a conclusion → discuss with the mentor → pose the next question.
