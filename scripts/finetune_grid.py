"""Phase 4 baseline: fully fine-tuned ResNet-50 on the same (train dataset, regime, seed) cells
as the frozen-head grid. Caches scores on the ID and OOD test splits in the shared predictions
format (head "ft") so scripts/evaluate.py and scripts/make_tables.py treat it like another head.
Weights of the 100% runs are saved for Grad-CAM. Resumable.

    PYTHONPATH=. uv run python scripts/finetune_grid.py --train iu nih
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.labels import LABELS, PATHOLOGIES
from data.regimes import (
    DATASETS,
    REGIMES,
    SEEDS,
    config,
    image_paths,
    metadata,
    regime_fraction,
    subset,
)
from encoders.finetune import predict, train

ENCODER = "resnet50_ft"
MAX_EPOCHS = {"1pct": 40, "10pct": 20, "matched": 20, "100pct": 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=list(DATASETS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="artifacts/results")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="float16 autocast for the forward pass")
    ap.add_argument("--max-epochs", type=int, default=None, help="override per-regime budget")
    args = ap.parse_args()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    cfg = config()
    out = Path(args.out)
    preds_dir, weights_dir = out / "preds", out / "weights"
    preds_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    runs_csv = out / "runs_ft.csv"
    metas = {ds: metadata(cfg, ds) for ds in DATASETS}
    n_train = {ds: int((m["split"] == "train").sum()) for ds, m in metas.items()}
    for train_ds in args.train:
        ood_ds = DATASETS[1 - DATASETS.index(train_ds)]
        meta = metas[train_ds]
        tests = {}
        for name, ds in (("id", train_ds), ("ood", ood_ds)):
            m = metas[ds][metas[ds]["split"] == "test"]
            tests[name] = (
                image_paths(cfg, ds, m["image_id"]),
                m["image_id"].to_numpy().astype(str),
            )
        for regime in args.regimes:
            frac = regime_fraction(regime, train_ds, n_train)
            if frac is None:
                continue
            for seed in args.seeds if frac < 1 else [0]:
                run_id = f"{train_ds}_{ENCODER}_{regime}_s{seed}"
                if (preds_dir / f"{run_id}.npz").exists():
                    continue
                t0 = time.time()
                tr, va = subset(meta, "train", frac, seed), subset(meta, "val", frac, seed)
                torch.manual_seed(seed)
                print(f"{run_id}: n_train={len(tr)} n_val={len(va)} device={device}", flush=True)
                model, info = train(
                    image_paths(cfg, train_ds, meta.iloc[tr]["image_id"]),
                    meta.iloc[tr][list(LABELS)].to_numpy(),
                    image_paths(cfg, train_ds, meta.iloc[va]["image_id"]),
                    meta.iloc[va][list(LABELS)].to_numpy(),
                    device,
                    range(len(PATHOLOGIES)),
                    lr=args.lr,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs or MAX_EPOCHS[regime],
                    workers=args.workers,
                    amp=args.amp,
                    log=lambda s: print(s, flush=True),
                )
                if regime == "100pct":
                    torch.save(model.state_dict(), weights_dir / f"{run_id}.pt")
                model = model.to(device)
                arrays = {}
                for name, (paths, ids) in tests.items():
                    arrays[f"{name}_image_id"] = ids
                    arrays[f"{name}_ft"] = predict(model, paths, device, workers=args.workers)
                tmp = preds_dir / f"{run_id}.tmp.npz"
                np.savez(tmp, **arrays)
                tmp.rename(preds_dir / f"{run_id}.npz")
                row = {
                    "run_id": run_id,
                    "train_ds": train_ds,
                    "ood_ds": ood_ds,
                    "encoder": ENCODER,
                    "regime": regime,
                    "seed": seed,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "best_epoch": info["best_epoch"],
                    "epochs": info["epochs"],
                    "select": info["select"],
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "amp": args.amp,
                    "history": json.dumps(info["history"]),
                    "fit_seconds": round(time.time() - t0),
                }
                pd.DataFrame([row]).to_csv(
                    runs_csv, mode="a", header=not runs_csv.exists(), index=False
                )
                done = f"best epoch {info['best_epoch']}/{info['epochs']} {row['fit_seconds']}s"
                print(f"{run_id}: {done}", flush=True)


if __name__ == "__main__":
    main()
