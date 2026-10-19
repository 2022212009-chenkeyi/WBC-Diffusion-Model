# Phase 2 Plan: DinoBloom Source Audit and WBC Subset Construction

## Goal

Acquire and analyze the original releases for all 13 sources named in the
DinoBloom v1 cohort wherever the source is publicly available and its access
terms permit acquisition. Preserve each release unchanged and report its
native categories, task, split, and data quality. Record internal, unavailable,
or inaccessible sources explicitly instead of treating them as downloaded.
After the cohort-wide source inventory, construct an approximately 10K-image
derived WBCDiffusion subset from eligible WBC morphology data. Keep the source
audit comprehensive while keeping the training subset scientifically scoped.

**Terminology correction:** DinoBloom itself is a family of pretrained vision
models, not a single dataset archive. The user-provided v1 paper describes a
13-source cohort; its LA (LabAnonymous) source is internal and expected to be
released separately. The work therefore tracks source datasets individually
instead of assuming a single downloadable "DinoBloom Dataset". The study held
Acevedo out as an external evaluation dataset, so it must be kept distinct
from any training-subset recommendation.

## Rules from the research plan

- Confirm actual SSD/HDD allocations and user paths from NTU EEE Cluster
  documentation or the account administrator. Never infer a user path from a
  mount name, and do not create a storage directory before confirmation.
- Keep raw data immutable. Keep raw data, derived metadata, and analysis output
  separate; do not put large data files in Git or Home.
- Record the source, release/version, license or access conditions, checksums,
  commands, code revision, and analysis decisions so the work can be reproduced.
- Do not declare category names from folder names alone until the release
  documentation and annotation schema confirm their meaning.
- Keep native labels and tasks intact in the source inventory; put any label
  harmonization in a separate, versioned mapping.
- Separate WBC morphology image-level data from patient diagnosis, detection,
  segmentation, RBC-only, internal, and external-test data.

## Execution stages and gates

For each completed stage, record in the run report: what was done, why, the
evidence obtained, how it can be reproduced, and what question or decision is
next. This carries the research plan's Phase 2 researcher-training rule into
the dataset workflow.

### 1. Confirm storage and transfer constraints

The active NTU EEE cluster session reports `/projects/_hdd` as a Ceph HDD mount.
The cluster `storagemgr -user keyi005` reports `/projects/_hdd/keyi005hdd` with
400 GB assigned and 0 B used (checked 2026-10-12). The account owns this
directory. The planned `dinobloom/{archives,raw,manifests}` directories have
been created there. Before each dataset transfer, check this allocation's
remaining capacity and that dataset's expected compressed and extracted size.
Do not use the mount-wide free-space figure as the user's quota.

### 2. Identify and qualify each source release

Use the user-provided DinoBloom v1 paper and authors' repository to identify
the 13 cited sources, then use each original dataset host as the authority for
download instructions and terms. Record exact release/version, URL or access
procedure, license or access terms, archive size, expected extracted size,
published checksums, documentation, and annotations. INT/LA is not currently a
public download target unless its authors publish it. Do not assume the
DinoBloom code/model license also licenses source images.

Classify every cited source as (a) WBC morphology image-level data, (b) a
different task/domain such as detection or segmentation, (c) external
evaluation-only, or (d) internal/unavailable/inaccessible. The acquisition
scope is all public, eligible cohort sources, not only the sources later
chosen for WBCDiffusion training. For each source, record exact release,
access route, terms, size, and label documentation; mark it acquired, in
progress, blocked, or unavailable. **Gate:** acquire only individually
verified releases under their stated terms and within the confirmed personal
HDD quota. An inaccessible or internal source stays explicitly recorded as
unavailable; it is not silently dropped from the cohort-wide report.

### 3. Plan HDD layout and provenance

After the storage gate, use a layout such as:

```text
<confirmed HDD path>/dinobloom/
├── archives/       # downloaded release, preserved as received
├── raw/            # extracted original release, left unchanged
└── manifests/      # file/checksum manifest for this release
```

Keep scripts and compact reports in this project directory; keep generated
inventories and derived metadata in a clearly separate analysis output folder.
Record the actual paths in the run report rather than assuming the example
layout above exists.

### 4. Download and verify

Check capacity immediately before transfer. Use resumable transfer if the host
supports it. Save the exact command and timestamps. Verify the publisher's
checksum when available; otherwise calculate and record a local cryptographic
checksum. Inspect the archive member list before extraction and reject unsafe
paths or unexpected contents. Extract only after verification, then compare
the extracted file inventory with the archive listing.

### 5. Inventory and establish category semantics

Read the release documentation, directory layout, and annotation schema. Map
each image to its documented category, split, and any other relevant labels.
Do not treat split directories as classes. If labels are ambiguous or absent,
report the ambiguity instead of guessing. Record the mapping rule and how many
files could not be mapped.

### 6. Analyze distribution and data quality

The analysis report is a Phase 2 deliverable for downstream WBCDiffusion
training. Present results as both tables and statistical charts. At minimum,
include per-source/native-label counts and proportions, the final candidate
training subset's class counts and proportions, and a source-by-class cross
tabulation where labels can be mapped defensibly. Charts should show class
imbalance clearly (for example, sorted count bars and percentage bars); state
whether each figure describes raw files, decodable images, deduplicated
content, or the selected subset. Include split/group counts where available.
Do not present preliminary literature counts as measured results.

Report:

- total files and image files, with counts by documented category and split;
- per-category counts and proportions, plus unlabeled/unmapped counts;
- formats, dimensions, and file sizes where they can be read reliably;
- unreadable/corrupt images, duplicate files (at least exact byte duplicates),
  missing or malformed labels, and unexpected files;
- counting rules, excluded items, and known limitations.

Do not silently remove, rewrite, or deduplicate raw files. Any proposed
exclusion or deduplication belongs in a derived manifest with a reason.

### 7. Construct and validate the approximately 10K Phase 2 subset

First produce a cohort-wide source audit report with source/version/access
terms, confirmed HDD location, download and checksum evidence, native
category/split tables for every acquired source, and a status/reason for every
source not acquired. Include data-quality findings, reproducible commands, and
unresolved issues. Then specify the subset objective and inclusion policy from
the observed evidence:
eligible tasks/domains, native labels retained, augmented-copy handling,
patient/source grouping, class balancing or natural prevalence, and
exclusions. Do not use Acevedo (external holdout) in training. Do not turn
patient diagnosis or detection/segmentation labels into cell morphology.

Build the approximately 10K dataset as a derived dataset, not by modifying raw
sources. Save a fixed-seed inclusion manifest mapping each selected item to its
source, original relative path, native label, split/group identifier when
available, and derived label if applicable. Document duplicate and leakage
controls, split policy, transformations, and counts. Prefer links or a
manifest over a second physical copy when practical; record the actual
implementation and storage footprint. **Completion gate:** count and
category/split totals reconcile to the manifest; source terms permit planned
use; known patient/source leakage does not cross splits; and the dataset can be
reconstructed from unchanged raw releases and recorded configuration.

Make the derived manifest usable by the later WBCDiffusion data loader: define
stable class names and integer IDs, train/validation/test assignments, a
source-relative path convention, and the meaning of every manifest field.
Record preprocessing assumptions (such as color mode, image size, and pixel
range) as recommendations/configuration, while preserving source pixels in
raw storage. Include a small schema/example so the Phase 3 loader can consume
the dataset without relying on directory-name guessing.

## Planned project artifacts

- `02_dinobloom_analysis/PLAN.md` — this execution plan.
- `02_dinobloom_analysis/inventory.py` — local, read-only filesystem inventory
  utility; it does not download or modify data.
- `02_dinobloom_analysis/libarchive_rar.py` — list or extract RAR using the
  already-installed system libarchive; rejects unsafe paths and links.
- `02_dinobloom_analysis/download_ranges.py` — resumable HTTP range transfer
  utility that validates each server range before assembling the archive.
- `02_dinobloom_analysis/class_counts.csv` and
  `02_dinobloom_analysis/summarize_classes.py` — reviewed per-source count
  inputs and deterministic table/SVG report generation.
- `02_dinobloom_analysis/subset_manifest_schema.md` — data-loader-facing
  manifest and class-map contract for the derived Phase 2 data.
- `02_dinobloom_analysis/README.md` — run instructions and analysis notes,
  updated as authoritative release and allocation facts are established.
- `datasets/metadata/dinobloom/` — compact manifests and final analysis report,
  once there is verified data to analyze.
- `datasets/processed/dinobloom-wbc-10k/` (or another quota-appropriate,
  documented location) — derived Phase 2 subset and inclusion manifest; choose
  final layout after confirming capacity and whether physical image copies are
  necessary.
- `02_dinobloom_analysis/report/` (or `results/phase2_dataset/`) — final
  tables, chart source/data, rendered figures, and a concise report connecting
  the verified category distribution to downstream WBCDiffusion data loading.

## Current status

**Raabin transfer (2026-10-14):** the existing 50,401,722-byte prefix is being
combined with 1 MiB HTTP ranges. Forty-two exact ranges are verified and
stored; transient resets are retrying. Final archive size must equal
292,300,976 bytes, then the complete RAR must pass a full content read before
extraction.

- The plan, inventory tool, and analysis instructions are in
  `02_dinobloom_analysis/`.
- The active cluster mount, personal HDD directory, ownership, and allocation
  are verified: `/projects/_hdd/keyi005hdd`, 400 GB assigned, 0 B used at
  verification time. The archive/raw/manifest directories are created.
- DinoBloom is a model family trained on a heterogeneous multi-source cohort,
  not one dataset. The user-provided v1 paper and author repository identify
  the source list; v1 calls the internal data LA, while later material may use
  another name. Keep this naming/version discrepancy explicit.
  paper-reported and source-reported counts do not always match, so final counts
  must come from downloaded manifests and not be copied from overview tables.
- Acevedo V1 has been downloaded to HDD, verified at 281,366,219 bytes, and
  passed ZIP integrity checks. Its local SHA-256 and file-level inventory are
  recorded under `datasets/metadata/dinobloom/`.
- The official Raabin-WBC cropped double-labeled `Train.rar` transfer is
  partially complete and running through verified HTTP ranges. The site
  labels the dataset free-access and asks users to cite the publication; it
  does not show a distinct license identifier. Do not extract or count it
  before integrity verification.
- The first interrupted transfer was preserved with the `.incomplete` suffix;
  it was not extracted or used. The verified archive is the canonical `.zip`.
- Remaining cohort sources still require individual release, terms, size, and
  access checks before transfer.
- The Phase 2 approximately 10K subset is an explicit planned deliverable, not
  merely a recommendation. It remains gated on source audit, documented
  selection rules, use terms, leakage checks, and inclusion-manifest
  reconciliation.
