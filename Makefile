# Repo root on sys.path regardless of the editable install (macOS can flag .pth files hidden,
# which Python >= 3.11.16 then ignores).
export PYTHONPATH := .

.PHONY: setup data-nih data-iu data-vindr metadata eda embeddings sanity grid finetune evaluate tables gradcam paper test lint reproduce

setup:
	uv sync

data-nih:
	uv run python scripts/download_nih.py

data-iu:
	uv run python scripts/download_iu.py

data-vindr:
	uv run python scripts/download_vindr.py

embeddings:
	uv run python scripts/extract_embeddings.py

sanity:
	uv run python scripts/sanity_probe.py --dataset nih --encoders resnet50 dinov2 biomedclip rad_dino

# Phase 3: fit heads and cache predictions (resumable), bootstrap metrics, tables and figures.
grid:
	uv run python scripts/run_grid.py

# Phase 4: fine-tuned ResNet-50 on the same cells (hours on MPS; resumable), then Grad-CAM vs boxes.
finetune:
	uv run python scripts/finetune_grid.py

evaluate:
	uv run python scripts/evaluate.py

tables:
	uv run python scripts/make_tables.py

gradcam:
	uv run python scripts/gradcam_boxes.py

# Phase 5: LNCS paper from the generated tables (needs pdflatex, bibtex and the llncs class).
paper: tables
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main | grep -E "Output written|Warning: (Citation|Reference)"

metadata:
	uv run python scripts/build_metadata.py

eda:
	uv run python scripts/eda.py

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

# Regenerates every derived table/figure. Frozen grid: minutes; fine-tuning: hours; both resumable.
reproduce: metadata eda grid finetune evaluate tables gradcam paper
