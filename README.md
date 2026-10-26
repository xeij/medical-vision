# medical-vision

Do frozen medical vision foundation-model embeddings transfer to a new hospital better than a
fully fine-tuned CNN, and does retrieval-based diagnosis (kNN over an embedding index) degrade
less under distribution shift than a trained classifier head? This repository is the research code
for that question on frontal chest X-rays, evaluated leave-one-dataset-out across hospitals.

Nothing here is a clinical tool and no clinical utility is claimed. All results are research
measurements on public datasets.

## Status

All five phases are done. NIH ChestX-ray14 and the Indiana University (Open-i) collection
are downloaded, label-mapped and split; embeddings from the four frozen encoders are cached for
both; the full grid (both transfer directions, four encoders, linear probe and kNN heads, four
label regimes, patient-level bootstrap intervals) plus a fully fine-tuned ResNet-50 on the same
cells is run and reported in `reports/results.md`; class activation maps are scored against the
NIH radiologist boxes in `reports/gradcam_boxes.md`; the paper in `paper/` compiles with every
table generated from those results, with author details, the related-work section and the
discussion marked as to-do. A VinDr-CXR loader exists but is dormant: it needs PhysioNet
credentials this project does not have.

## Quickstart

Prerequisites: Python 3.11 (uv installs it), [uv](https://docs.astral.sh/uv/), curl, ~15 GB of
disk for the converted NIH images. `make paper` also needs pdflatex and bibtex with the IEEEtran
class (`tlmgr --usermode install ieeetran` on a basic TeX Live).

```bash
make setup                # uv sync
make data-nih             # ~45 GB download, converted to 518-px PNGs, zips deleted as it goes
make data-iu              # ~2 GB download, frontal images only
make reproduce            # metadata parquet + splits CSV + reports/eda.md
make embeddings           # 4 encoders x 2 datasets, cached under artifacts/embeddings (hours on MPS)
make sanity               # in-distribution linear probe on NIH -> reports/sanity_probe_nih.md
make grid                 # fit probe + kNN for every (direction, encoder, regime, seed); cache scores
make finetune             # fine-tune ResNet-50 on the same cells (17 runs, ~8 h on an M-series GPU)
make evaluate             # AUROC/AUPRC/ECE with 1,000-resample patient bootstrap -> metrics.csv
make tables               # reports/results.md, results_macro.csv, figures/, paper/tables/*.tex
make gradcam              # class activation maps vs NIH boxes -> reports/gradcam_boxes.md
make paper                # paper/main.pdf
make test                 # label mapping, splits, image conversion, extraction, metrics, heads
make lint
```

`make reproduce` runs metadata, EDA, grid, finetune, evaluate, tables, gradcam and paper in
sequence from the cached embeddings. The frozen grid takes a few minutes of CPU, the bootstrap
evaluation about ten, the fine-tuning grid 7.6 hours on an M-series GPU; every step is resumable.

Run scripts through `make` or with `PYTHONPATH=. uv run python scripts/<name>.py`: the Makefile
puts the repo root on `sys.path` so the scripts do not depend on the editable install.

## How it works

Every dataset is reduced to the same two things:

- `datasets/<name>/images_518/<image_id>.png`: 8-bit grayscale, shortest side 518 px (RAD-DINO's
  native input; the other encoders downsample from there), aspect ratio preserved. Raw archives
  and DICOMs are never kept on disk.
- `artifacts/metadata/<name>.parquet`: one row per image with `image_id`, `patient_id`, `view`,
  `age`, `sex`, `split` and six binary labels (`atelectasis`, `cardiomegaly`, `consolidation`,
  `effusion`, `pneumothorax`, `no_finding`). The label mapping and its ambiguities are in
  `LABELS.md`.

Splits are patient-level. Each dataset's official test list is kept as the test set; validation is
carved from the remaining *patients* with a fixed seed. The resulting `splits/<name>.csv` files are
committed and `tests/test_split_integrity.py` asserts no patient appears in two splits. VinDr-CXR
ships no patient identifier, so its splits are image-level by necessity (documented in
`LABELS.md`).

Embeddings are computed once per (encoder, dataset) by `scripts/extract_embeddings.py` and cached
as `artifacts/embeddings/<dataset>/<encoder>.npy` (float32, one row per image) with a parquet
mapping `image_id` to row. Extraction writes to a memmap with a progress marker, so a multi-hour
run resumes where it stopped. Each encoder runs at its own native resolution and normalisation:
RAD-DINO at 518 px, DINOv2 and BiomedCLIP and ResNet-50 at 224 px. One vector per image: the CLS
token for the ViTs, BiomedCLIP's projected image embedding, ResNet-50's global average pool.
The three ViTs run in float16 on the GPU (2 to 4 times faster on Apple MPS; cosine similarity to
the float32 embeddings is above 0.99999 on a 128-image check) and are stored as float32.
Everything downstream (linear probes, kNN retrieval, bootstrap CIs) runs on these cached arrays so
the experiment grid fits a laptop CPU.

Two heads sit on the frozen embeddings. The linear probe is one L2-regularised logistic
regression per label on standardised embeddings, with C chosen on the in-distribution validation
split by macro AUROC over the five pathologies. The retrieval head is a FAISS flat inner-product
index over L2-normalised training embeddings; a test image's score for a label is the fraction of
its k nearest training images carrying that label, for k in {5, 20, 50}. Label regimes (1%, 10%,
100%, plus NIH at IU's training size) subsample whole patients with three seeds; the validation
split is subsampled at the same fraction so model selection never sees more labels than training.

Metrics are AUROC, AUPRC and ECE (15 equal-width bins) per label and macro over the five
pathologies. `eval/metrics.py` implements them with integer sample weights vectorised over
resamples, so the 1,000-resample patient-level bootstrap of a 25k-image test set is one pass. The
same resampled patients are used for every run, which makes head-vs-head and
in-domain-vs-transferred comparisons paired. Two gaps are reported. The ID-vs-OOD gap of the
specification compares one model on its own test split and the other dataset's test split; it
turned out to be dominated by how hard each test split is (the IU split, with manual labels and
mostly normal studies, is easier than the NIH official split), so the primary measure is the
same-test-set transfer gap: on the target's test split, the in-domain model trained on the target
minus the model trained on the source. Positive means the transferred model falls short of the
in-domain ceiling.

The fully fine-tuned baseline is the same ResNet-50 trained end to end on the same cells of the
grid: ImageNet initialisation, every layer trained with binary cross-entropy over the six labels,
AdamW at 1e-4, learning rate divided by ten on a validation plateau, early stopping on validation
macro AUROC, random resized crops and small rotations for augmentation. Evaluation uses the frozen
ResNet-50's Resize(256)+CenterCrop(224) transform so the two differ only in whether the weights
moved. Its test-set scores enter the same evaluation and tables as a fifth head, so every number
and interval is computed by the same code on the same resamples.

Localisation is checked against the 984 radiologist boxes shipped with ChestX-ray14, all of which
lie in the test split; 577 are for labels in the shared set (no boxes exist for consolidation). For
a network that ends in global average pooling and a linear layer, Grad-CAM at the last
convolutional block is exactly the class activation map, so it is computed in closed form from
the fine-tuned network's output weights and, for the frozen encoder, from the probe's
coefficients divided by the standardiser's scale. Agreement is scored by the pointing game and by
the fraction of map energy inside the box, each reported next to the chance level of a uniform
map, plus IoU at half maximum (`reports/gradcam_boxes.md`).

On macOS the torch and faiss-cpu wheels each bundle an OpenMP runtime, and a process that imports
torch before faiss crashes at the first search. The grid runner never imports torch, and
`tests/conftest.py` imports faiss first.

## Results so far

Macro AUROC over the five pathologies at 100% of the source's training labels. "In-domain" is
trained and tested on the same dataset; "transferred" is trained on the other dataset and tested
on this one. Intervals are in `reports/results.md`.

| encoder | head | NIH test, in-domain | NIH test, transferred from IU | IU test, in-domain | IU test, transferred from NIH |
|---|---|---|---|---|---|
| ResNet-50 (ImageNet) | probe | 0.714 | 0.590 | 0.748 | 0.826 |
| ResNet-50 (ImageNet) | kNN k=50 | 0.642 | 0.567 | 0.629 | 0.673 |
| DINOv2 ViT-B | probe | 0.750 | 0.628 | 0.718 | 0.795 |
| DINOv2 ViT-B | kNN k=50 | 0.646 | 0.554 | 0.607 | 0.604 |
| BiomedCLIP | probe | 0.771 | 0.696 | 0.848 | 0.832 |
| BiomedCLIP | kNN k=50 | 0.728 | 0.678 | 0.760 | 0.730 |
| RAD-DINO | probe | 0.815 | 0.717 | 0.893 | 0.869 |
| RAD-DINO | kNN k=50 | 0.770 | 0.698 | 0.801 | 0.817 |
| ResNet-50 fine-tuned | end-to-end | 0.814 | 0.651 | 0.788 | 0.877 |

Four things the grid shows. Retrieval loses less of its in-domain ceiling under transfer than the
probe does (on the NIH test set the paired contrast is positive for every encoder at 10% and
100% source labels), but its ceiling is lower, so the probe remains the better transferred model
in most cells. With 78k NIH labels, NIH-trained probes match or beat IU-trained ones on the IU
test set for three of four encoders; at equal training size the gap is near zero for all but
RAD-DINO, whose in-domain IU probe benefits most from IU's cleaner labels. The fully fine-tuned
ResNet-50 ties the best frozen pipeline when it has 78k labels (0.814 against RAD-DINO's probe at
0.815 on NIH, 0.877 against 0.869 on IU) but falls well behind it with fewer labels: on the NIH
test split it trails the RAD-DINO probe by 0.11 AUROC at 1% of the labels, 0.09 at 10% and 0.12
at IU's training size, and when trained on IU's 2.6k images its shortfall against the
NIH-trained ceiling is the largest of any model. The naive ID-minus-OOD gap is negative for most NIH-trained
models, which says more about the two test splits than about the encoders.

Against the radiologist boxes, the NIH-trained fine-tuned network puts its activation maximum
inside the box for 45% of boxes (chance 11%, cardiomegaly 93%, pneumothorax at chance), the
IU-trained one for 34%, and the frozen ImageNet ResNet-50 with the NIH probe for 22%
(`reports/gradcam_boxes.md`).

## Layout

```
data/        loaders (nih.py, iu.py, vindr.py), label mapping, split assignment, image conversion
encoders/    frozen encoders (frozen.py), resumable extraction (extract.py), fine-tuning (finetune.py)
heads/       probe.py (per-label logistic regression), knn.py (FAISS cosine retrieval)
eval/        metrics.py (weighted AUROC/AUPRC/ECE, patient bootstrap), localization.py (CAM vs box)
configs/     data.yaml
scripts/     download_*.py, build_metadata.py, eda.py, extract_embeddings.py, sanity_probe.py,
             run_grid.py, finetune_grid.py, evaluate.py, make_tables.py, gradcam_boxes.py
splits/      committed patient-level split CSVs
reports/     eda.md, sanity_probe_nih.md, results.md, results_macro.csv, gradcam_boxes.md, figures/
tests/
paper/       main.tex (IEEE two-column conference format), main.pdf, refs.bib, tables/ written by
             make_tables.py and gradcam_boxes.py
```

## Data

| dataset | source | access | frontal images | labels |
|---|---|---|---|---|
| NIH ChestX-ray14 | NIH Clinical Center, via the Hugging Face mirror of the official Box release | public | 112,120 / 30,805 patients | NLP-mined from reports |
| IU chest X-ray (Open-i) | two Indiana hospital systems; NLM Open-i, via the Hugging Face mirror of raddar's Kaggle conversion (CC BY-NC-ND 4.0) | public | 3,603 reports (one frontal each, 86 unindexed dropped) | manually assigned MeSH terms |

Both datasets are used without any account. VinDr-CXR, CheXpert, MIMIC-CXR and PadChest all
require registration and are out of scope.

A known caveat: the public RAD-DINO checkpoint was pretrained on all NIH ChestX-ray14 images
(and CheXpert and MIMIC-CXR). The IU collection is unseen by every encoder in the study, so
IU-as-held-out is the clean cross-hospital direction. IU is small and consolidation (30) and
pneumothorax (27) are rare in it, so those out-of-distribution estimates carry wide intervals.
See `reports/eda.md`.
