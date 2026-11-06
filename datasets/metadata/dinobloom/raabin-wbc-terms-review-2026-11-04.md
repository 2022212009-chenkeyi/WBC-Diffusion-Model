# Raabin-WBC access and reuse terms review

Checked: 2026-11-04. Scope: the cropped, double-labeled Raabin-WBC Train archive used for the unconditional image-model baseline.

## Evidence reviewed

1. **Official Raabin page:** [raabindata.com](https://raabindata.com/) labels the resource “Free Normal WBC Dataset”, describes the available components and intended machine-learning tasks, and asks users to cite the original dataset publication. It also asks users who use segmentation ground truth to cite an additional segmentation paper. The page does not display a separate license identifier or explicit redistribution terms for the image files.
2. **Official component index:** [Cropped double-labeled archive index](https://dl.raabindata.com/Index%20of%20WBC%20Data/Index%20of%20WBC%20Cropped_double_labeled/) lists `Train.rar`, `TestA.rar`, and `TestB.zip`; the directory listing does not include a license or terms file.
3. **Original dataset paper:** [Scientific Reports 12, 1123 (2022)](https://www.nature.com/articles/s41598-021-04426-x) describes the dataset as free access, states that it is publicly available at Raabin's page, and describes use for machine-learning model development and testing. Its data-availability statement does not specify a distinct dataset license.

## Findings and limits

- Evidence supports public access, research/model-development use, and a citation request. The official source does **not** identify a specific license for the dataset images or set out explicit redistribution permissions.
- An open-access license attached to the paper, if shown by the publisher, applies to the paper text and figures; it must not be assumed to license the separate image archive.
- The current project trains on images and uses directory labels only for audit/split reporting. It does not use segmentation masks. Cite the Raabin-WBC dataset paper in any publication or report that uses this dataset.
- This review is not legal advice and does not establish permission to redistribute raw images, derived crops, checkpoints that may memorize images, or generated samples. Keep source images and model outputs private to the research project unless the provider's terms are clarified.

## Decision for this experiment

Proceed with the planned non-commercial academic baseline as a publicly accessible dataset research use, while recording the missing explicit dataset license as a limitation. Before public release or redistribution of raw or generated images, obtain clarification from the dataset provider or an institutional data steward. Do not describe the images as CC BY based only on a paper/article license.

Related acquisition record: `raabin-wbc-train-source.json`.
