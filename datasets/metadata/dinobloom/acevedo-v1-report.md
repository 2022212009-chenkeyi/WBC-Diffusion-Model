# Acevedo V1 inventory report

**Retrieved:** 2026-10-12  
**Source:** [Mendeley Data, version 1](https://data.mendeley.com/datasets/snkd93bnjr/1)  
**License shown on source page:** CC BY 4.0  
**Role in DinoBloom v1:** external evaluation set, held out from model training.

## Storage and integrity

- HDD archive: `/projects/_hdd/keyi005hdd/dinobloom/archives/acevedo-v1-PBC_dataset_normal_DIB.zip`
- Archive bytes: `281366219` (matches the archive server's Content-Length)
- SHA-256: `6c50b82d1ab499b0b5a6beda0d0f75e6013df4a7233f116da354351ac10ca4f8`
- ZIP integrity: passed (`unzip -t`, no compressed-data errors)
- Extracted source files are kept unmodified at
  `/projects/_hdd/keyi005hdd/dinobloom/raw/acevedo-v1/PBC_dataset_normal_DIB/`.

## Categories found in the source directory structure

The Mendeley record describes eight groups. The archive has these eight
category folders; counts below exclude one non-image metadata artifact that
has a `.jpg` suffix but cannot be decoded.

| Folder label | Valid image files | Share |
|---|---:|---:|
| neutrophil | 3,329 | 19.48% |
| eosinophil | 3,117 | 18.24% |
| ig | 2,895 | 16.94% |
| platelet | 2,348 | 13.74% |
| erythroblast | 1,551 | 9.07% |
| monocyte | 1,420 | 8.31% |
| basophil | 1,218 | 7.13% |
| lymphocyte | 1,214 | 7.10% |
| **Total valid images** | **17,092** | **100.00%** |

`ig` is retained as the source label (“immature granulocyte” in the Mendeley
description); this inventory does not split it into promyelocyte, myelocyte,
and metamyelocyte subtypes. Erythroblasts and platelets also appear in the
source, so not all eight folders are conventional mature WBC classes.

## File quality

- The extraction contains 17,093 `.jpg`-suffixed files under the category
  root. `neutrophil/.DS_169665.jpg` is a 6,148-byte Apple metadata artifact,
  not a decodable JPEG. It remains untouched in the raw data. This leaves
  17,092 decodable images, matching the source's reported total.
- Exact SHA-256 comparison found 17 duplicate groups, each containing two
  identical paths. Raw files are retained; any future duplicate exclusion
  must be represented in a derived manifest. This corresponds to 17,075
  distinct decodable image contents by exact byte hash.
- The archive also contains a separate `__MACOSX/` sidecar directory. It was
  preserved during extraction and excluded from class counts; the analysis
  root is the dataset's `PBC_dataset_normal_DIB/` directory.
- All decodable images are RGB JPEGs. 16,639 are 360×363 pixels; the remaining
  453 have small dimension variations (359–366 pixels wide and 360–369 pixels
  high).
- Class folders are imbalanced: largest-to-smallest valid category ratio is
  about 2.74 (neutrophil/lymphocyte).

## Interpretation and limitation

The folder labels provide eight observed source categories, whereas the
DinoBloom v1 paper describes Acevedo as 11 classes and the Mendeley record
describes eight groups, including an `ig` group that covers several immature
granulocyte subtypes. The counts above describe the downloaded V1 archive, not
a harmonized 11-class relabeling. This source is evaluation-only when following
the paper's protocol and should not be used in the planned training subset.

The full machine-readable file inventory, dimensions, decode result, and hashes
are in [`acevedo-v1-inventory.json`](acevedo-v1-inventory.json).
