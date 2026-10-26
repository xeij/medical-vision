# Label mapping

Six labels present in every dataset in scope. Each is a binary column; an image can be positive
for several pathologies. Code: `data/labels.py`.

| shared label   | NIH ChestX-ray14 (`Finding Labels`) | IU / Open-i (manual MeSH major terms)                 | VinDr-CXR (dormant) | CheXpert / MIMIC-CXR (v2) |
|----------------|-------------------------------------|-------------------------------------------------------|---------------------|---------------------------|
| atelectasis    | Atelectasis                         | Pulmonary Atelectasis                                 | Atelectasis         | Atelectasis               |
| cardiomegaly   | Cardiomegaly                        | Cardiomegaly; Cardiac Shadow/enlarged                 | Cardiomegaly        | Cardiomegaly              |
| consolidation  | Consolidation                       | Consolidation                                         | Consolidation       | Consolidation             |
| effusion       | Effusion                            | Pleural Effusion                                      | Pleural effusion    | Pleural Effusion          |
| pneumothorax   | Pneumothorax                        | Pneumothorax; Hydropneumothorax; Hemopneumothorax     | Pneumothorax        | Pneumothorax              |
| no_finding     | No Finding                          | normal                                                | No finding          | No Finding                |

## Decisions and ambiguities

**`no_finding` keeps its native meaning.** In every source it means "none of that dataset's
full label set", not "none of our five". An NIH image labelled only `Infiltration` is therefore
negative on all six shared columns: it is neither normal nor positive for a target pathology.
Redefining `no_finding` as "none of our five" would have labelled images with other pathologies as
normal, which is wrong. Consequence: the six columns are not a partition and a row can be all zeros.

**Out-of-scope but related findings are left as negatives.** NIH `Pneumonia` and `Infiltration`,
VinDr `Pneumonia` (global) and `Lung Opacity`, overlap clinically with `consolidation` but are not
mapped. Some consolidation-like images are therefore negatives; this adds label noise in both
directions and is a limitation, not a bug.

**IU MeSH terms are matched on the term before the first `/`**, case-insensitively, over the
`;`-separated list of manually assigned major terms. Modifiers are ignored except for
`Cardiac Shadow/enlarged`, which is counted as cardiomegaly (43 reports); `Cardiac Shadow` with
other modifiers (obscured, irregular) is not. `Hydropneumothorax` and `Hemopneumothorax` count as
pneumothorax. The 86 reports whose MeSH is `No Indexing` were never manually coded; their labels
are unknown, so the loader drops them rather than treating them as negatives. Related terms left as
negatives: `Opacity`, `Airspace Disease`, `Infiltrate`, `Pneumonia`, `Pulmonary Edema`,
`Pulmonary Congestion`, `Heart Failure`. Counts (frontal images, after dropping unindexed):
pulmonary atelectasis 310, cardiomegaly 319 (+43 cardiac shadow/enlarged), consolidation 30,
pleural effusion 149, pneumothorax 22 (+5 compound), normal 1343. Consolidation and pneumothorax
are rare in IU, so their out-of-distribution estimates will carry wide confidence intervals.

**NIH `Effusion` is pleural effusion** (the ChestX-ray14 paper defines it as such).

**VinDr-CXR (dormant: requires PhysioNet credentials, which this project does not have).**
Training labels would be aggregated by majority vote. Each training image was labelled
independently by 3 radiologists (one CSV row each); a label is positive when more than half of
that image's raters marked it. The test set is a 5-radiologist consensus with one row per image,
so the same rule is the identity there. If a positive pathology majority coexists with a
`No finding` majority (only possible with internally inconsistent raters), the pathology wins and
`no_finding` is set to 0.

**NIH labels are NLP-mined from reports; IU labels are manually assigned MeSH codes.** Any
NIH-to-IU degradation therefore mixes image-domain shift with label-quality shift.

**IU ships no patient identifier.** Each report is one de-identified visit with no patient
linkage, so `patient_id := report uid` (which at least keeps a visit's images together).
Patient-level guarantees hold for NIH only. The same applies to VinDr-CXR.

**VinDr source column names** (`Pleural effusion`, `No finding`, ...) are taken from the PhysioNet
description; the loader raises if a column is missing, so a naming mismatch fails loudly on first
load rather than silently dropping a label.

**Dropped from the shared set:** Edema and Pneumonia exist in NIH, VinDr (PhysioNet version),
CheXpert and MIMIC but are absent from the Kaggle release of VinDr; kept out of v1 to stay at the
six labels specified.
