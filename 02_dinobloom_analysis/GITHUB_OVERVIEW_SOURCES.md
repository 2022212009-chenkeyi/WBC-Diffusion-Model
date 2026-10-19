# DinoBloom GitHub dataset table: acquisition tracker

The live [DinoBloom repository](https://github.com/marrlab/DinoBloom) describes
the model as trained on 13 datasets, but its current “Overview of publicly
available Blood Cell Datasets” table lists 15 rows. This tracker follows the
table as requested and keeps those 15 entries separate from the paper's 13
source cohort. The table also includes segmentation, detection, patient-level
labels, non-human data, and additional disease datasets; these are acquired
and inventoried as their own sources, never merged into a common class count.

“Listed count” is copied from the repository and is not a verified local count.
The detailed DinoBloom cohort interpretation remains in
[`COHORT_CLASSES.md`](COHORT_CLASSES.md).

| GitHub table source | Listed images | Native task/labels | Original source | HDD status |
|---|---:|---|---|---|
| BMC | 171,373 | Bone marrow; 21 cell labels | [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) | Official page confirms 6.8 GB JPG / CC BY 4.0; partial transfer has 95,259 complete JPGs plus partial/checkpoint files; Aspera checkpoint resume restarted 2026-10-19 |
| AML Hehr | 101,949 | Blood; patient-level AML subtype labels | [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) | 81,214 TIFF images present, matching current official TCIA image count (GitHub count differs); inventory being generated; official page reports CC BY 4.0 |
| AML Matek | 18,365 | Blood; 15 cell morphology labels | [TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/) | Annotations acquired; official page confirms 11 GB TIFF / CC BY 3.0; current Faspex receive attempt rejected its public passcode, needs link recheck |
| Acevedo | 17,092 | Blood; repository lists 10 labels; Mendeley groups into eight | [Mendeley V1](https://data.mendeley.com/datasets/snkd93bnjr/1) | V1 downloaded and inventoried; external holdout |
| Raabin WBC | 10,175 | Blood; five WBC classes | [Raabin](https://raabindata.com/) | Train component downloaded, extracted, and inventoried |
| NuClick | 2,933 | Blood-cell segmentation | [NuClick repository](https://github.com/navidstuv/NuClick) | Not acquired; author-linked Warwick archive redirects to institutional login |
| Warty pig | 2,871 | WBC classification; raw, augmented, and cropped images | [Google Drive](https://drive.google.com/drive/folders/1CsDoL448kvAtFVd5jowVJGKjFLv3qjz4) | Downloaded and verified 3,541 files (3,539 decodable images; 0 decode errors). Folder-label counts and an interim chart are in `results/phase2_dataset/interim_warty_pig/`; raw Original (667) and derivative augmented/cropped groups remain distinct. |
| LISC | 2,263 | Five-class WBC images and segmentation | [Original page](https://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm) | Official directory lists 214 MB ZIP; repeated direct author-host requests return HTTP 400; no mirror used |
| KRD-WBC | 601 | Five-class WBC segmentation, 600 images per Mendeley description | [Mendeley V2](https://data.mendeley.com/datasets/jzdj6h7gms/2) | V2 downloaded, extracted, and inventoried; CC BY 4.0 |
| SSL Seg | 400 | WBC segmentation | [GitHub repository](https://github.com/zxaoyou/segmentation_WBC) | Repository archive downloaded, extracted, and inventoried |
| BCCD | 364 | Detection: WBC, RBC, platelet | [Kaggle](https://www.kaggle.com/datasets/konstantinazov/bccd-dataset) | Downloaded/extracted; 364 annotated images (plus 1 example preview); 4,888 native boxes; bundled MIT file may cover repository code, image rights need confirmation |
| Aslan | 100 | Detection: WBC, RBC | [GitHub repository](https://github.com/draaslan/blood-cell-detection-dataset) | Repository archive downloaded, extracted, and inventoried; MIT license |
| Raabin Leukemia | ? | Patient labels: four leukemia groups | [Raabin](https://raabindata.com/) | Official `LeukemiaData.rar` downloaded at expected 35,057,436,145 bytes; safe archive scan/listing is in progress before extraction; verify license and inventory after scan |
| APL_AML | 25,915 | Patient labels plus cell labels | [Kaggle](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl) | Downloaded/extracted; 25,724 JPGs, 3 undecodable; 21 native cell folders plus 10,089 decodable unsigned images; patient table has 34 APL / 72 AML; license unknown |
| White-Blood-Cell-dataset | 376 | WBC segmentation | [GitHub repository](https://github.com/arbackes/White-Blood-Cell-dataset) | Repository archive downloaded, 367 images and 367 masks inventoried; no license file found |

Archive byte counts, SHA-256 hashes, extracted locations, image totals, decode
status, and verified native class/box counts for acquired sources are in
[`github-overview-acquisition.json`](../datasets/metadata/dinobloom/github-overview-acquisition.json).

## Download and preservation rules

- Download only through the listed official or author-linked source. Record the
  release URL, access date, license/terms, archive size, and SHA-256.
- Keep every source in its own directory under
  `/projects/_hdd/keyi005hdd/dinobloom/`; never combine archives before creating
  an unchanged-source inventory.
- Do not infer image labels from folder names until the source documentation or
  annotation schema confirms their meaning. Preserve patient-level labels and
  split/group identifiers when available.
- If a source requires an account, access agreement, or software transfer route
  unavailable on the cluster, record that limitation and continue with other
  sources rather than substituting an unofficial mirror.
- Do not include Acevedo in the WBCDiffusion training subset by default; retain
  it as the DinoBloom external holdout.
