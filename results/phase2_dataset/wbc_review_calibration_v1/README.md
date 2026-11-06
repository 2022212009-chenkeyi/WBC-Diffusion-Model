# WBC Reviewer Calibration Set

This reproducible set contains 20 QC-kept real training images, selected with seed `20261106` from the frozen training split and resized using the baseline RGB/Lanczos preprocessing (256×256).

Use these labeled real examples to align how reviewers interpret the scoring scale before they receive any generated images. Discuss score meanings only; do not change the preregistered pass thresholds based on this set. These examples are from the training split and are not part of the 200-image held-out test review.

Artifact severity is reverse-coded: 1=severe/unusable, 3=visible but not severe, 5=no obvious artifact. The other 1–5 scales use 1=poor/unrecognizable and 5=clear/complete.

`calibration_manifest.csv` records source paths and hashes. `calibration_scoring_sheet.csv` is optional for individual practice; it is not part of the formal blind-review analysis.
