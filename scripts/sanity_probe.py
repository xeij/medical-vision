"""Phase 2 sanity check: in-distribution linear probe on cached embeddings (the same head as
the Phase 3 grid). Picks C on val by macro AUROC over the five pathologies and reports AUROC per
label on val and test.

    uv run python scripts/sanity_probe.py --dataset nih --encoders resnet50 rad_dino
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.labels import LABELS, PATHOLOGIES
from eval.metrics import auroc
from heads.probe import fit_probe


def load_embeddings(cfg, dataset, encoder):
    d = Path(cfg["artifacts_dir"]) / "embeddings" / dataset
    x = np.load(d / f"{encoder}.npy", mmap_mode="r")
    ids = pd.read_parquet(d / f"{encoder}.parquet")
    meta = pd.read_parquet(Path(cfg["artifacts_dir"]) / "metadata" / f"{dataset}.parquet")
    meta = ids.merge(meta, on="image_id", how="left")
    return x, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nih")
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--max-train", type=int, default=None, help="subsample train rows for speed")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/data.yaml"))
    rows = []
    for enc in args.encoders:
        x, meta = load_embeddings(cfg, args.dataset, enc)
        idx = {s: np.flatnonzero(meta["split"] == s) for s in ("train", "val", "test")}
        if args.max_train:
            rng = np.random.default_rng(cfg["seed"])
            idx["train"] = np.sort(rng.choice(idx["train"], args.max_train, replace=False))
        y = {s: meta.iloc[idx[s]][list(LABELS)].to_numpy(np.int8) for s in idx}
        probe = fit_probe(
            x[idx["train"]], y["train"], x[idx["val"]], y["val"], pathology_cols=range(5)
        )
        for split in ("val", "test"):
            p = probe.predict(x[idx[split]])
            aucs = {lab: auroc(y[split][:, j], p[:, j]) for j, lab in enumerate(LABELS)}
            rows.append(
                {
                    "encoder": enc,
                    "split": split,
                    "C": probe.c,
                    **{k: round(v, 3) for k, v in aucs.items()},
                    "macro5": round(float(np.mean([aucs[p] for p in PATHOLOGIES])), 3),
                }
            )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    # Merge with earlier runs so encoders can be probed one at a time.
    csv = Path("reports") / f"sanity_probe_{args.dataset}.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        table = pd.concat([old[~old["encoder"].isin(args.encoders)], table], ignore_index=True)
    table.to_csv(csv, index=False)
    csv.with_suffix(".md").write_text(
        "# Linear-probe sanity check (AUROC)\n\n```\n" + table.to_string(index=False) + "\n```\n"
    )


if __name__ == "__main__":
    main()
