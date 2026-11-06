# Raabin-WBC Train QC: goal progress

## Scope and current decision

The first unconditional DDPM baseline remains the Raabin-WBC cropped, double-labeled Train component. The three newly requested sources are kept outside this baseline until each source has its own audit. The class folders are useful for stratified quality review and post-generation analysis; the unconditional model input will contain images only.

The official Raabin page describes the data as free access and requests citation of the original publication, but it does not state a separate dataset license identifier. The dataset paper and download terms should be checked again before redistribution. See [Raabin official data page](https://raabindata.com/) and [original dataset paper](https://www.nature.com/articles/s41598-021-04426-x).

## Inventory and duplicate screening

- 10,175 JPEGs; the existing inventory reports 575×575 RGB and successful decoding for all files.
- SHA-256 was computed for every image. All 10,175 hashes are unique; no byte-identical duplicates were found.
- The original Raabin paper explains that repeated cells can occur in the broader cropped collection, and also says the curated double-labeled five-class subset does not contain repeated samples. The Train component is part of that double-labeled subset, so the paper's curation is relevant evidence; local duplicate checks are still retained as a reproducibility check.
- A 64-bit difference-hash screen at Hamming distance ≤4 found 160 candidate pairs (320 images). Pairwise pixel-difference ranking and larger contact sheets were generated and visually reviewed. The pairs are very similar in many cases, but the visual evidence does not establish identity; treat them as candidate lookalikes, not confirmed duplicates. No images were removed. Pair IDs are included in the QC decision register so each pair can be assigned to the same split conservatively; this may group some distinct but similar cells.
- Filenames fall into numeric/date-time patterns, but no patient or smear identifier has been established. Do not interpret filename tokens as patient IDs. Patient-level split isolation cannot currently be claimed.

## Technical triage and visual sample

- A deterministic 64×64 triage measured grayscale brightness and contrast, edge energy, and extreme-pixel fractions for all 10,175 images. It flagged no images under the deliberately broad extreme-value thresholds. This is a technical screen only; it cannot judge cell identity, morphology, focus, truncation, or staining quality.
- A reproducible sample of 20 images per native class (seed `20261104`) was visually reviewed. After checking the 12 initially flagged samples at original size, 98 were marked `keep`, 1 `review`, and 1 `exclude` for multiple target cells. These are image-usability decisions, not medical label validation; the single review case needs a second reviewer.
- The full image-level register has 10,173 `keep`, 1 `review`, and 1 `exclude`. The 10,075 unsampled images were retained under a documented dataset-level rule: they decode, match the verified 575×575 RGB inventory and broad technical checks, and belong to the curated double-labeled Train component. They were not individually viewed; this is recorded on each row.
- One image is marked `exclude` in the decision register; it remains on the HDD and is omitted from the frozen train/validation/test manifest. `keep / review / exclude` decisions are preserved with an image-level reason.

## Artifacts

- `../../../datasets/metadata/dinobloom/raabin-wbc-train-sha256-manifest.csv` — per-image relative path, class folder, size, and SHA-256.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-dedupe-report.json` — exact duplicate summary and filename-pattern audit.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-neardup-candidates.csv` — 160 candidate pairs requiring review.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-neardup-report.json` — perceptual-hash method and limitations.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-neardup-pixel-review.csv` — normalized pixel-difference ranking for candidate pairs.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-qc-decisions.csv` — image-level `keep / review / exclude` register with decision basis recorded per row.
- `../../../datasets/metadata/dinobloom/raabin-wbc-train-split-manifest.csv` — frozen grouped train/validation/test assignment for 10,173 kept images.
- `../../../datasets/metadata/dinobloom/raabin-wbc-preprocessing-v1.json` — fixed RGB resize and normalization settings; includes the split-manifest SHA-256.
- `split_summary.json` — split counts, independent verification, and limitations.
- `balanced_sample_visual_review.csv` — review reasons for the 100 stratified sample images.
- `balanced_random_samples_20_per_class_seed20261104.jpg` and `balanced_random_samples_manifest.csv` — balanced visual sample.
- `highres_class_samples/` — one higher-resolution sample sheet per native class.
- `technical_screening_metrics.csv` and `technical_screening_summary.json` — deterministic image-level technical triage.
- `neardup_review/` — side-by-side contact sheets grouped by hash distance.

## Goal status

1. Data source: complete — Raabin Train is the initial baseline; additional sources stay separate.
2. Source/terms, grouping, and duplicate audit: complete with limitations — official sources describe free public research access and request citation, but show no separate data-file license or redistribution terms (`datasets/metadata/dinobloom/raabin-wbc-terms-review-2026-11-04.md`); exact hashes are unique; 160 lookalike candidate pairs are provisionally grouped for split protection; patient/smear grouping is unavailable and cannot be inferred from filenames.
3. Image QC: complete with limitations — 10,173 kept, 1 review, 1 excluded. The review case is withheld from training pending a second look; unsampled keep decisions are rule-based rather than individual visual inspection.
4. Split: complete with limitations — 8,139 train / 1,017 validation / 1,017 test; SHA-256 values are unique and no candidate pair crosses splits. Patient/smear separation cannot be guaranteed.
5. Preprocessing: fixed as v1 — RGB, LANCZOS square resize without cropping, float32 `[-1,1]`, no color normalization or augmentation.
6. Smoke/run comparison: A6000 smoke job 245357 completed successfully (2 updates, finite losses, checkpoint and sample grid saved); recovery job 245400 loaded checkpoint step 2. The matched 128 and 256 pilots (jobs 245361 and 245363) both completed 200 updates. Their losses stayed finite and fell to 0.0129 / 0.0156, respectively; peak allocated VRAM was 0.8 / 2.7 GiB. Both 200-step sample grids are noise-like, so they are pipeline/resource checks only, not quality evidence.
7. Formal baseline: complete — job 245428 finished 200,000 updates on RTX PRO 6000 Blackwell Max-Q in 2:40:59; final validation epsilon MSE was 0.0041909. The final EMA checkpoint and training logs are retained on the SSD.
8. Evaluation: machine stage complete — job 245465 generated 2,000 EMA images; job 245467 computed FID/KID/PRDC; job 245482 screened exact-pixel/dHash neighbors and prepared three blinded scoring packets. The machine metrics show a large generated-vs-real feature gap and low coverage; see `results/wbc_unconditional_baseline_001/analysis.md`. Exact-pixel and dHash≤4 screens found no matches. All three human scoring sheets are still blank, so the preregistered image-quality gates and final decision on conditional generation remain pending.
