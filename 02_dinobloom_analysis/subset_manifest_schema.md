# Derived WBC subset manifest schema

This schema is for the derived Phase 2 subset and its later WBCDiffusion data
loader. It does not change source files or infer labels. The final class map,
split policy, and selected rows will be filled only after source inventories
and use terms have been reviewed.

## Required manifest fields

| Field | Meaning |
|---|---|
| `sample_id` | Stable identifier derived from source ID, release, and source-relative path. |
| `source_id` | Dataset source key, such as `raabin` or `aml_matek`. |
| `source_release` | Exact source release/version. |
| `source_relpath` | Path relative to that source's immutable raw root; no machine-specific absolute path. |
| `sha256` | Content hash for integrity and exact-duplicate tracking. |
| `task` | Source annotation task, such as cell classification, detection, or segmentation. |
| `native_label` | Original label exactly as supplied by the source. |
| `analysis_class` | Explicit Phase 2 class mapping; blank for unharmonized or excluded data. |
| `class_id` | Stable integer from the versioned class map; blank when not a selected class sample. |
| `source_split` | Original source split, or `not-reported`. |
| `model_split` | Phase 2 `train`, `validation`, `test`, `external_test`, or `excluded`. |
| `group_id` | Pseudonymous patient/smear/slide grouping key for leakage control, if available. |
| `domain` | Source domain such as peripheral blood, bone marrow, or animal. |
| `augmentation_status` | `original`, `augmented`, or `unknown`. |
| `selection_status` | `included` or `excluded`. |
| `selection_reason` | Auditable reason for inclusion/exclusion. |

The loader resolves `source_id` to a raw root in configuration, then appends
`source_relpath`. It must never depend on folder names to assign `class_id`.
Keep source-native splits separate from model splits, and assign all samples
from a known group to one model split. The external Acevedo holdout remains
distinct from training and from any internal random split.

## Versioned class map

Keep a separate `class_map.json` with a mapping ID, stable integer IDs, class
names, the source-native labels accepted for each class, and the rule/evidence
for each mapping. Changing a mapping creates a new mapping ID and regenerated
manifest; do not silently reuse IDs for different meanings.

## Example (illustrative only)

```csv
sample_id,source_id,source_release,source_relpath,sha256,task,native_label,analysis_class,class_id,source_split,model_split,group_id,domain,augmentation_status,selection_status,selection_reason
raabin-v1-abc123,raabin,Train-component-v1,Train/eosinophil/example.jpg,<sha256>,cell-classification,eosinophil,eosinophil,2,train,train,smear-007,peripheral-blood,original,included,eligible native WBC class
```

The example row is schema-only and is not a real selected sample. Replace it
with rows generated from verified source inventories before training.
