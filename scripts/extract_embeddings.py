"""Compute and cache embeddings: artifacts/embeddings/<dataset>/<encoder>.{npy,parquet,progress}.

uv run python scripts/extract_embeddings.py --encoders resnet50 dinov2 --datasets nih iu
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from encoders import ENCODERS
from encoders.extract import extract
from encoders.frozen import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--datasets", nargs="+", default=["nih", "iu"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/data.yaml"))
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    for enc in args.encoders:
        transform, forward, dim = load(enc, device)
        for ds in args.datasets:
            meta = pd.read_parquet(Path(cfg["artifacts_dir"]) / "metadata" / f"{ds}.parquet")
            if args.limit:
                meta = meta.head(args.limit)
            img_dir = Path(cfg["datasets_dir"]) / ds / f"images_{cfg['image_size']}"
            paths = [img_dir / f"{i}.png" for i in meta["image_id"]]
            out_dir = Path(cfg["artifacts_dir"]) / "embeddings" / ds
            if args.limit:
                out_dir = out_dir / "smoke"
            t0 = time.time()
            extract(
                paths,
                meta["image_id"].tolist(),
                transform,
                forward,
                dim,
                out_dir,
                enc,
                device,
                args.batch_size,
                args.workers,
            )
            print(
                f"{enc}/{ds}: {len(paths)} images, dim {dim}, {time.time() - t0:.0f}s", flush=True
            )


if __name__ == "__main__":
    main()
