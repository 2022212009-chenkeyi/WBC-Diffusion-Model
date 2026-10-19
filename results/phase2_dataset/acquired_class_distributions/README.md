# Acquired dataset category distributions (interim)

This is a first-pass summary from sources with usable native class or object
counts currently recorded in `class_counts.csv`. Each chart preserves that
source's label names and count unit. Counts from different datasets are not
pooled.

## Charts and detailed tables

See [the source-separated report](charts/class_counts_report.md). It contains
one bar chart and count/share table for each source, split, task, and label
schema represented in the CSV. The SVG figures are in `charts/figures/`.

## Included distributions

| Source | What the chart counts | Interpretation |
|---|---|---|
| Acevedo V1 | 17,092 decodable cell images across 8 native groups | Kept as an external holdout; do not add to the WBCDiffusion training subset by default. |
| Raabin WBC Train | 10,175 cell images across 5 classes | Train component only, not the separate Raabin Leukemia package. |
| KRD-WBC V2 | 600 labeled images across 5 cell classes | Segmentation dataset; class counts are labeled images. |
| SSL Seg | Dataset 1: 300; Dataset 2: 100 labeled images | Two source splits are charted separately. |
| Aslan | 2,339 bounding boxes | Object counts, not image counts. |
| BCCD | 4,888 bounding boxes | Object counts, not image counts. |
| APL_AML | 15,632 signed cell images across 21 native classes; 106 patient diagnoses | Cell morphology and patient diagnosis are charted separately. Unsigned slides are unlabeled and excluded from class charts. |
| Warty pig | 1,464 augmented and 1,408 cropped-classified images | Derivative groups are charted separately; the 667 `Original` images are not assigned inferred class labels. |
| AML Matek | 18,365 annotation records across 15 morphology labels | **Annotation-only:** image package is not acquired yet, so these are label records rather than locally verified image counts. |

## Acquired sources without a class histogram yet

- **White-Blood-Cell-dataset:** 367 images and 367 masks are available, but no
  morphology class table is supplied.
- **AML Hehr:** 81,214 TIFF images are present. Its class labels are
  patient-level; patient metadata still needs to be parsed and linked to the
  image set.
- **Raabin Leukemia:** the 35,057,436,145-byte archive passed the safe member
  scan and its outer layer has been extracted to the HDD. It contains nested
  ZIP/RAR archives; those still need inspection and class-image counting.
- **BMC:** the TCIA transfer is incomplete and is resuming from Aspera
  checkpoints; no class histogram is included yet.

NuClick and LISC have no local image data because the official source links
remain inaccessible; see
[`GITHUB_OVERVIEW_SOURCES.md`](../../../02_dinobloom_analysis/GITHUB_OVERVIEW_SOURCES.md).

