# Experiment record

## Identity

- Experiment ID:
- Title:
- Date (UTC or timezone noted):
- Status: planned / running / complete / failed
- Code commit:
- Config path:
- Run command:

## 1. Why?

Why is this experiment needed? What baseline or open question motivates it?

## 2. Hypothesis

I expect changing **[X]** to affect **[Y]** by **[direction/magnitude]**, because **[reason/evidence]**.

## 3. Setup

- Dataset/version and split:
- Dataset manifest/metadata:
- Model and code revision:
- Baseline/comparison:
- Hyperparameters:
- Random seed(s):
- Hardware/GPU:
- Python/PyTorch/CUDA/driver:
- Dependency lock or environment export:
- Training duration / compute budget:

## 4. Experiment

Describe the controlled change and procedure. Note deviations from the plan.

## 5. Record

- Config:
- Logs:
- Checkpoint(s):
- Generated samples:
- Metrics:
- Exact reproduction command:

## 6. Result

| Measure | Baseline | This run | Notes |
|---|---:|---:|---|
| FID (protocol/sample count): | | | |
| KID (protocol/sample count): | | | |
| Diversity/morphology: | | | |
| Sampling speed: | | | |
| Peak memory: | | | |

Qualitative observations (include sample grid location and avoid cherry-picking):

## 7. Analysis

What evidence supports or contradicts the hypothesis? What confounders, failures, or limitations remain?

## 8. Conclusion

What does this experiment establish, and what does it not establish?

## 9. Next experiment

What is the next evidence-based step, and which variable will it change?
