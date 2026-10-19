# DinoBloom dataset download handoff — 2026-10-14

This note records where the authorized acquisition task stands so it can be
resumed tomorrow. Dataset files are being kept on the HDD at
`/projects/_hdd/keyi005hdd/dinobloom/`.

## Continuation update — 2026-10-15

- Warty pig finished: all 3,541 manifest files (2,128,515,810 bytes) passed
  Drive size/MD5 checks. Inventory: 3,539 decodable images, zero decode
  errors, and two Excel feature workbooks. Folder-level class counts and an
  interim SVG chart/table are in
  [`interim_warty_pig`](../results/phase2_dataset/interim_warty_pig/class_counts_report.md).
  Augmented and cropped-classified image counts are separate derivative
  groups; 667 `Original` images have no inspected class subfolder. The
  augmented feature workbook has 1,328 rows versus 1,464 files in its folder
  tree, so it was not used as the count source. License not established.
- Raabin Leukemia has resumed and verified 249/2,090 ranges so far (about
  3.7 GiB of range data; the complete archive is 35.1 GB). Current execution
  session: `91178`; range cache remains at
  `/projects/_hdd/keyi005hdd/dinobloom/archives/.raabin-leukemia-ranges`.
- TCIA BMC and AML Hehr transfers have started through Aspera and are still
  active. Current execution sessions: BMC `42594`, AML Hehr `70374`. They are
  writing to `raw/bmc-tcia-v1` and `raw/aml-hehr-tcia-v1` respectively.
- AML Matek's public package link was freshly retrieved from the official
  TCIA page but Aspera rejected its passcode as invalid; leave image files
  pending and recheck the official access link later. Its annotations remain
  acquired.
- NuClick remains behind the author's Warwick institutional login and LISC's
  official archive still returns HTTP 400.

The current transfer progress is also reflected in
[`GITHUB_OVERVIEW_SOURCES.md`](../02_dinobloom_analysis/GITHUB_OVERVIEW_SOURCES.md).

## Status check — 2026-10-19

- No transfer processes from the prior execution were still running at the
  start of this check.
- The complete Raabin Leukemia RAR is present at the expected byte size
  (35,057,436,145). A safe libarchive scan is running before extraction.
- AML Hehr has 81,214 TIFF files plus one `.sums` file, matching the current
  official TCIA image count. The image inventory process is still running.
- BMC had 95,259 complete JPGs, 160 partial files, and 159 Aspera checkpoint
  files when checked. Its public TCIA package transfer has been restarted to
  the same directory so Aspera can resume from checkpoints.
- Current execution sessions: BMC resume `41266`, AML Hehr inventory `82140`,
  and Raabin archive scan `88820`. Session IDs may not survive a reconnect;
  check process state and HDD contents before restarting.

## Class-chart request — 2026-10-19

- Generated a source-separated chart/table collection at
  `results/phase2_dataset/acquired_class_distributions/`. It currently has 12
  SVG bar charts across nine sources, based on 80 native-count rows. Its README
  distinguishes image, annotation-record, patient, and bounding-box counts and
  marks datasets without available class labels.
- The outer Raabin Leukemia RAR passed archive listing and was extracted to
  `/projects/_hdd/keyi005hdd/dinobloom/raw/raabin-leukemia-v1`; the extracted
  content is itself 50 nested ZIP/RAR archives and still needs class-level
  inspection.
- BMC Aspera session `41266` remained present, but no files in its directory
  had changed during the last 10-minute sample. Check progress before assuming
  it is transferring; its last observed state was Aspera waiting on socket I/O.
- AML Hehr's inventory was still running in storage I/O wait. Its 81,214 TIFFs
  are present, but the metadata and image inventory need to finish before
  adding any patient-level distribution to the chart set.

## Scope

The request follows all 15 rows in the current “Overview of publicly
available Blood Cell Datasets” table in the DinoBloom GitHub repository. The
paper's 13-dataset cohort is tracked separately. Keep each source's native
labels and task separate; do not combine classification, detection,
segmentation, or patient-level labels into one class distribution.

The source-by-source status and known access/license caveats are in
[`GITHUB_OVERVIEW_SOURCES.md`](../02_dinobloom_analysis/GITHUB_OVERVIEW_SOURCES.md).
Acquisition metadata and checksums are in
[`github-overview-acquisition.json`](../datasets/metadata/dinobloom/github-overview-acquisition.json).

## Status at handoff

- Eight table sources have been downloaded and inventoried: Acevedo V1,
  Raabin WBC Train, KRD-WBC, SSL Seg, Aslan, White-Blood-Cell-dataset, BCCD,
  and APL_AML. BCCD and APL_AML native counts/reports are already recorded.
- Warty pig is downloading from its public Drive folder. Its manifest contains
  3,541 files totaling 2,128,515,810 bytes. The downloader verifies file size
  and MD5 and can resume by skipping verified files. After completion, check
  the tree, inventory the images, inspect its spreadsheets for native label
  counts, and add results to the source metadata and charts.
- Raabin Leukemia is downloading from the official Raabin host. The expected
  RAR size is 35,057,436,145 bytes. It uses resumable, verified 16 MiB ranges
  with four workers; progress was slow (about 12 of 2,090 ranges verified at
  the last check). Keep the partial range cache. After completion, verify the
  archive, inspect/extract it safely, then inventory and count native groups.
- AML Matek annotation files are acquired and analyzed, but its 11 GB image
  package still needs to be received through the public TCIA Aspera/Faspex
  package.
- BMC and AML Hehr also have public TCIA Aspera/Faspex packages located, but
  image transfers have not started. AML Hehr's official TCIA page reports
  81,214 images / 13.3 GB, which differs from the GitHub table's 101,949.
  Its clinical metadata workbook is already on the HDD at
  `manifests/aml-hehr-clinical-metadata.xlsx`.
- NuClick remains inaccessible because the author-linked Warwick archive
  redirects to institutional login. LISC remains inaccessible from its
  official author host (HTTP 400); no mirrors have been substituted.

## Transfer sessions at handoff

These session IDs were returned by the current execution environment and may
not survive a reconnect:

- Warty pig Drive downloader: session `27127`.
- Raabin Leukemia range downloader: session `2853`.

If the sessions are still attached, poll them with `write_stdin`. If not, check
whether the downloads are still running before restarting them. Warty can be
rerun with `02_dinobloom_analysis/download_public_drive_folder.py`; it skips
files whose size and MD5 already match. Raabin can be rerun with
`02_dinobloom_analysis/download_ranges.py`, using the same URL, expected size,
`.raabin-leukemia.part` target, and
`.raabin-leukemia-ranges` work directory. The exact invocation is in the
conversation handoff context; preserve the range cache so completed chunks are
reused.

The portable Aspera CLI v4.27.5 is extracted under
`/tmp/aspera-cli-portable/` and can be used for the three public TCIA packages.
`/tmp` may be cleared between sessions, so if absent, reacquire the official
portable release and verify SHA-256
`97805ad0e7950dd540c5fd23397638dec971f486b591e17d71e3f02a216282bb` before
use. Do not print or commit public-link passcodes.

## Resume checklist

1. Check whether Warty pig and Raabin transfers are still running; inspect
   their current progress and the HDD mount before restarting anything.
2. Finish verification, extraction, inventory, native label counts, and
   source-specific charts for completed downloads.
3. Use Aspera CLI with the public TCIA packages for BMC, AML Hehr, and AML
   Matek; resume or retry without overwriting verified data.
4. Update the tracker and acquisition JSON after each source completes. Keep
   access failures and license uncertainties explicit.
5. Produce a final source-separated table and figures covering all sources
   that could be acquired, with task and label definitions shown alongside
   counts. Clearly mark unavailable sources and distinguish annotated images
   from previews, unlabeled images, masks, and bounding boxes.

