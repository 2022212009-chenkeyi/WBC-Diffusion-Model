# DinoBloom study cohort: dataset and label overview

This table follows the user-provided [arXiv v1 paper](https://arxiv.org/html/2404.05022v1)
and the [authors' dataset overview](https://github.com/marrlab/DinoBloom). The
13 sources do **not** share one class system: they mix cell morphology,
patient-level disease subtypes, object detection, and segmentation. Treat each
dataset's native labels separately until an explicit harmonization is designed.

| Source | Documented labels/task | How to interpret it for this project |
|---|---|---|
| **BMC** | 21 bone-marrow cell labels: ABE abnormal eosinophil, ART artefact, BAS basophil, BLA blast, EBO erythroblast, EOS eosinophil, FGC faggot cell, HAC hairy cell, KSC smudge cell, LYI immature lymphocyte, LYT lymphocyte, MMZ metamyelocyte, MON monocyte, MYB myelocyte, NGB band neutrophil, NGS segmented neutrophil, NIF not identifiable, OTH other, PEB proerythroblast, PLM plasma cell, PMO promyelocyte. | Cell morphology classification; largest source. Labels include artefacts and non-WBC/ambiguous classes. Source: [TCIA collection](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/). |
| **AML Hehr** | Five patient-level labels: CBFB::MYH11, NPM1, PML::RARA, RUNX1::RUNX1T1, healthy. | AML subtype labels belong to patients/smears, not individual cells. Do not treat as image-level cell classes. Source: [TCIA collection](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/). |
| **LA (LabAnonymous)** | Internal peripheral-blood images; v1 says over 40K and planned a separate release. | Not presently in our public-source download scope. |
| **AML Matek** | 15 morphology labels: BAS basophil, EBO erythroblast, EOS eosinophil, KSC smudge cell, LYA atypical lymphocyte, LYT typical lymphocyte, MMZ metamyelocyte, MOB monoblast, MON monocyte, MYB myelocyte, MYO myeloblast, NGB band neutrophil, NGS segmented neutrophil, PMB bilobed promyelocyte, PMO promyelocyte. | Image-level morphology classification; strongly imbalanced and includes abnormal/immature cells. Source: [TCIA collection](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/). Verify the collection's exact release and label files before transfer. |
| **Acevedo** | Mendeley describes eight groups: neutrophil, eosinophil, basophil, lymphocyte, monocyte, immature granulocytes (including promyelocytes/myelocytes/metamyelocytes), erythroblasts, platelets. The downloaded V1 archive has eight folders, with `ig` kept as one folder. The paper says 11 classes. | External evaluation source in DinoBloom; exclude from training by default. Actual file inventory is recorded in [`acevedo-v1-report.md`](../datasets/metadata/dinobloom/acevedo-v1-report.md). Sources: [Mendeley V1](https://data.mendeley.com/datasets/snkd93bnjr/1), [paper v1](https://arxiv.org/html/2404.05022v1). |
| **Raabin WBC** | Five cell types: eosinophil, lymphocyte, monocyte, neutrophil, basophil. | Image-level WBC classification. The official site offers a cropped, double-labeled Train component; source: [Raabin-WBC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8782871/) and [official data page](https://raabindata.com/). |
| **NuClick** | WBC images from four source classes used to generate artificially overlapping-nucleus segmentation examples. | Segmentation task, not necessarily a four-class classifier dataset. Preserve masks and source labels if present. Source: [NuClick repository](https://github.com/navidstuv/NuClick). |
| **Warty pig** | Basophil, eosinophil, monocyte, neutrophil (four classes); cropped and augmented juvenile Visayan warty-pig WBC images. | Species and augmentation domain differ from human WBC sources; keep domain identity and original/augmented status. Source: [author dataset link](https://drive.google.com/drive/folders/1CsDoL448kvAtFVd5jowVJGKjFLv3qjz4). |
| **LISC** | Five types: basophil, eosinophil, monocyte, neutrophil, lymphocyte. | WBC images plus segmentation information; paper count includes augmented variants. Source: [original dataset page](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm). |
| **Chula** | Red-blood-cell segmentation. | Not a WBC type-classification source; keep only if the intended target includes blood-cell context/background segmentation. |
| **SSLSeg** | Segmentation masks for 300 WBC images from two sources (200 and 100). | Pixel-level segmentation labels; no cell-type classes specified in the paper. Source: [dataset repository](https://github.com/zxaoyou/segmentation_WBC). |
| **BCCD** | Detection labels for WBC, RBC, and platelet. | Object detection labels rather than one class per cropped WBC. Source: [author-linked Kaggle dataset](https://www.kaggle.com/datasets/konstantinazov/bccd-dataset). |
| **Aslan** | Detection labels for WBC and RBC. | Object detection labels, not a WBC morphology taxonomy. Source: [dataset repository](https://github.com/draaslan/blood-cell-detection-dataset). |

## Implications for the approximately 10K subset

- The raw cohort cannot be pooled under one class mapping without losing task
  and domain information. Store source name, native label, patient/split when
  available, and original-versus-augmented provenance for every image.
- For a WBC image-generation subset, candidate image-level morphology sources
  include BMC, AML Matek, Raabin, LISC, and Warty pig, subject to individual
  data-use terms and scientific scope. Exclude Acevedo from training to honor
  the paper's external holdout protocol. Do not include RBC-only Chula or
  detection/segmentation sources unless the research question calls for them.
- Before balancing to 10K, inspect class counts and patient/image leakage at
  source level. Record any label harmonization as a separate derived mapping;
  retain original labels in the source manifest.
- These are literature-level descriptions. Actual downloaded annotations and
  file inventories remain the authority for exact class counts and mappings.
