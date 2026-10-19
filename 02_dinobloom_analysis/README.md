# DinoBloom cohort: acquisition and analysis

Start with [`PLAN.md`](PLAN.md) and [`COHORT_CLASSES.md`](COHORT_CLASSES.md).
DinoBloom is a pretrained model family; its
training data is a heterogeneous cohort assembled from independently hosted
source datasets. There is no single source archive to download as a unified
"DinoBloom Dataset". We track each source separately and preserve its own
labels and splits. Sources and counts below are reported by the authors and are
not a substitute for a verified file-level inventory.

Acquisition scope covers all 13 paper-listed sources when their public release
and terms permit access. Task differences determine how a source can be used
later; they do not remove a source from the cohort-wide inventory. The
approximately 10K WBCDiffusion subset is selected as a separate derived
dataset after that full-source review.

## Verified storage

On 2026-10-12 the active host was `login-3.cluster02.eee.ntu.edu.sg`. The HDD
mount is `/projects/_hdd`; the storage manager reported user `keyi005` has
`/projects/_hdd/keyi005hdd`, with 400 GB assigned and 0 B used. The directories
below have been created:

```text
/projects/_hdd/keyi005hdd/dinobloom/
├── archives/
├── raw/
└── manifests/
```

Recheck quota before every transfer. The mount-level `df` capacity is shared
filesystem capacity and is not the user's individual quota.

## DinoBloom study cohort (source-reported overview)

| Source | Paper/repository count | Label/task notes | Acquisition status |
|---|---:|---|---|
| BMC | 171,373 | Bone marrow, 21 cell categories | Verify source release and terms |
| AML Hehr | 101,949 in figure/repository; 81,214 in paper text | Patient labels: four AML subtypes plus healthy; image-level label semantics differ | Verify TCIA collection and terms |
| LA (LabAnonymous) | over 40,000 | Internal dataset in the user-provided v1 paper; planned separate release | Not available for download from this cohort plan |
| AML Matek | 18,365 | 15 highly imbalanced cell categories | Official image package pending; annotations acquired and counted |
| Acevedo | 17,092 | Mendeley says eight groups; paper v1 says 11 classes, authors' repository lists 10 labels; held out as external evaluation | V1 downloaded and inventoried; keep test-only |
| Raabin | 10,175 | Five WBC types | Train component downloaded, extracted, and inventoried |
| NuClick | 2,933 in figure | Segmentation resource, rather than a harmonized classification cohort | Verify which image set/purpose applies |
| Warty pig | 2,871 | Four WBC categories; includes augmented/cropped images | Verify source release and terms |
| LISC | 2,263 | Five WBC categories; augmented images included in total | Verify current source availability and terms |
| SSLSeg | 400 | WBC segmentation; two image sources | Verify source release and terms |
| BCCD | 364 | Detection labels include WBC, RBC, and platelet | Verify dataset version and terms |
| Aslan | 100 | Detection labels include WBC and RBC | Verify source release and terms |
| Chula | 706 | Red blood cell segmentation, not WBC cell-type classification | Verify source release and terms |

The current repository's overview table contains 15 rows, although its
introduction says the model cohort contains 13 datasets. Our acquisition
checklist follows all 15 table entries and records source access/terms for each
one in [`GITHUB_OVERVIEW_SOURCES.md`](GITHUB_OVERVIEW_SOURCES.md). This table
also lists supplementary public resources that are not necessarily among the
13 sources used to train DinoBloom.

The user-provided [arXiv v1 HTML](https://arxiv.org/html/2404.05022v1) names LA
(LabAnonymous) as its internal dataset; later copies may use a different name.
Several reported counts and class descriptions differ between the paper text,
figure, author repository table, and original dataset hosts (including AML
Hehr and Acevedo). We record these discrepancies and use downloaded source
manifests as the final count authority. Cohort membership does not mean that
every source is suitable for the later WBC image-generation subset:
detection/segmentation sets, RBC data, patient-level labels, augmented
variants, and held-out evaluation data need separate treatment.

## Downloaded and inventoried: Acevedo V1

- Official host: [Mendeley Data V1](https://data.mendeley.com/datasets/snkd93bnjr/1)
- License shown by the host: CC BY 4.0.
- Archive: `/projects/_hdd/keyi005hdd/dinobloom/archives/acevedo-v1-PBC_dataset_normal_DIB.zip`
- Archive size: 281,366,219 bytes; ZIP integrity check passed.
- SHA-256: `6c50b82d1ab499b0b5a6beda0d0f75e6013df4a7233f116da354351ac10ca4f8`
- Extracted, unchanged source: `/projects/_hdd/keyi005hdd/dinobloom/raw/acevedo-v1/PBC_dataset_normal_DIB/`
- File inventory: [`acevedo-v1-inventory.json`](../datasets/metadata/dinobloom/acevedo-v1-inventory.json)
- Analysis: [`acevedo-v1-report.md`](../datasets/metadata/dinobloom/acevedo-v1-report.md)
- Interim table and chart: [`class_counts_report.md`](../results/phase2_dataset/interim_acevedo_v1/class_counts_report.md)

The study treats Acevedo as external test data. Preserve this source as
evaluation-only for any reproduction of the DinoBloom training protocol; do
not include it in a training subset by default.

## Downloaded and inventoried: Raabin-WBC Train component

The [Raabin official site](https://raabindata.com/) calls this a free-access
dataset and asks users to cite the original publication. Its component index
offers `Train.rar` (292,300,976 bytes). This is the cropped, double-labeled
classification component corresponding to the 10,175-image training split
reported in the DinoBloom cohort. The host does not show a separate license
identifier on the download page; preserve the source citation and do not
redistribute it. The completed archive has SHA-256
`ac41ebc1547d9c5c70153fe1b5a1303630ad44643b8fa99cc72dccbe31b9ca31`; the full
archive was read and safely extracted to
`/projects/_hdd/keyi005hdd/dinobloom/raw/raabin-wbc-train/`. The inventory found
10,175 JPEGs, all decodable, and no exact duplicates were computed. Class counts,
inventory, source metadata, and interim chart are linked here:
[`raabin-wbc-train-inventory.json`](../datasets/metadata/dinobloom/raabin-wbc-train-inventory.json),
[`raabin-wbc-train-source.json`](../datasets/metadata/dinobloom/raabin-wbc-train-source.json),
[`interim_raabin_wbc_train`](../results/phase2_dataset/interim_raabin_wbc_train/).

## Next source: BMC

The [TCIA collection record](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
reports 6.8 GB, CC BY 4.0, and 21 native cell labels. Its page points to an
Aspera-based collection download; TCIA's current documentation describes the
NBIA Data Retriever for public collections. The cluster has neither client
installed. A metadata API request returned HTTP 200 with an empty body, so no
BMC transfer has started. Continue with TCIA's documented public download route
and record its manifest/checksums before transferring.

## TCIA AML-Matek access check

The official [AML-Cytomorphology_LMU collection record](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
reports 18,365 TIFF images, an 11 GB image release, CC BY 3.0, and separate
annotation/abbreviation files. The official annotation ZIP (550,950 bytes,
SHA-256 `4db78414ad9c3a2e493a6516be92d093992b310060c9575364dc502ff4e4cc31`)
and 541-byte abbreviation file have been downloaded to the HDD; the ZIP passed
integrity validation and `annotations.dat` was extracted. The annotation table
has 18,365 image records and 15 native primary classes. It also has two extra
annotation columns; 1,905 records have at least one non-`nan` reannotation.
The class counts are in [`aml-matek-v1-annotations.json`](../datasets/metadata/dinobloom/aml-matek-v1-annotations.json)
and `class_counts.csv`, with an interim table and SVG chart in
[`interim_aml_matek_v1`](../results/phase2_dataset/interim_aml_matek_v1/).
This is an annotation-derived count, not proof that all corresponding TIFFs
have been acquired or decoded. The official public image package is linked
through Aspera Faspex (11 GB). The cluster has no `ascp` client, and the TCIA
manifest query currently returns HTTP 500 with an empty series match. The image
transfer route remains unresolved; retain the annotation files and continue
with an official TCIA-supported download method.

## Reproducible local inventory

Once a source has been downloaded and extracted, inventory it without modifying
the source files:

```bash
python 02_dinobloom_analysis/inventory.py /projects/_hdd/keyi005hdd/dinobloom/raw/<source> \
  --output datasets/metadata/dinobloom/<source>-inventory.json
```

Add `--hash-files` to calculate SHA-256 values and find exact duplicate files;
this reads every file and may take time. The tool's top-level directory counts
describe folder structure only. It does not claim folder names are semantic
classes. The source-specific annotation reader and category report must follow
after its schema has been documented.

## Category tables and charts

The derived manifest contract is documented in
[`subset_manifest_schema.md`](subset_manifest_schema.md); it keeps source
labels, explicit analysis classes, model splits, and group IDs separate for
later WBCDiffusion loading.

`class_counts.csv` is the reviewed input for category reporting. Each row keeps
source/release, split, task, native label scheme, native category, count basis,
and data role explicit. Add rows only after confirming labels and counts from
source annotations or a verified inventory. Leave `analysis_class` and
`harmonization_id` blank until an explicit, documented cross-source mapping is
approved.

Generate tables and native-label SVG bar charts with the Python standard
library:

```bash
python 02_dinobloom_analysis/summarize_classes.py \
  02_dinobloom_analysis/class_counts.csv \
  --output results/phase2_dataset/class_report
```

The output includes `class_counts_summary.csv`, a Markdown report, and one
chart for each source/release/split/task/count-basis group. Interim source-only
reports are not the final Phase 2 cohort summary. Native categories from
unrelated sources are not combined by default.

When an explicit mapping is supplied, the generator also writes source-by-class
count and percentage matrices, separated by mapping version, split, and data
role so training and external-evaluation counts cannot be silently pooled.

## RAR archive handling

`libarchive_rar.py` uses the installed system `libarchive.so.13`, without a
root-level extractor install. After a transfer completes, record its SHA-256
and inspect the archive in full before extraction:

```bash
python 02_dinobloom_analysis/libarchive_rar.py \
  /projects/_hdd/keyi005hdd/dinobloom/archives/raabin-wbc-train.rar \
  --list-json datasets/metadata/dinobloom/raabin-wbc-train-members.json
```

After reviewing the member list, extract into a new source-specific directory
with a byte cap based on inspected contents and the confirmed HDD quota. The
tool rejects traversal/absolute paths, links, special files, duplicate paths,
and an existing destination. It never overwrites raw files.

The Raabin host supports byte ranges. The resilient transfer uses the existing
contiguous prefix, 1 MiB ranges, a small worker pool, and validates every
`Content-Range` and response length before assembly. Persistently failing
ranges are divided into smaller exact ranges after retrying:

```bash
python 02_dinobloom_analysis/download_ranges.py \
  'https://dl.raabindata.com/Index%20of%20WBC%20Data/Index%20of%20WBC%20Cropped_double_labeled/Train.rar' \
  /projects/_hdd/keyi005hdd/dinobloom/archives/raabin-wbc-train.rar \
  /projects/_hdd/keyi005hdd/dinobloom/archives/raabin-wbc-train.rar \
  --expected-size 292300976 \
  --work-dir /projects/_hdd/keyi005hdd/dinobloom/archives/.raabin-train-ranges \
  --chunk-size 1048576 --workers 4 --attempts 10 --timeout 600
```

## References

- [DinoBloom authors' repository and cohort table](https://github.com/marrlab/DinoBloom)
- [DinoBloom user-provided arXiv v1 HTML](https://arxiv.org/html/2404.05022v1)
- [DinoBloom MICCAI 2024 paper](https://papers.miccai.org/miccai-2024/paper/3584_paper.pdf)
- [BMC source record, TCIA](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/)
- [AML-Matek source record, TCIA](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/)
- [TCIA NBIA Advanced REST API guide](https://wiki.cancerimagingarchive.net/display/Public/NBIA+Advanced+REST+API+Guide)
- [TCIA Data Retriever CLI](https://github.com/TCIA/data-retriever)
- [Acevedo dataset, Mendeley Data](https://data.mendeley.com/datasets/snkd93bnjr/1)
