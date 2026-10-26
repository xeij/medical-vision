"""Indiana University chest X-ray collection (Demner-Fushman et al., 2016; NLM Open-i), taken from
the public Hugging Face mirror of raddar's Kaggle conversion (CC BY-NC-ND 4.0), which converted the
Open-i DICOMs to PNG and hand-labelled each image as frontal or lateral. One example per report
(uid) with a frontal image; we keep the frontal image only, stored as an 8-bit grayscale PNG with
shortest side = image_size. Raw parquet shards are deleted after conversion. Re-runnable.
"""

import csv
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from data.images import to_gray_shortest_side

REV = "790f3151f867653cd90371e28cbca2f131dea8f2"  # pinned mirror commit
BASE = f"https://huggingface.co/datasets/ykumards/open-i/resolve/{REV}/data/"
SHARDS = [
    "train-00000-of-00005-c80629f6027d8eb1.parquet",
    "train-00001-of-00005-07e6c158a3f56170.parquet",
    "train-00002-of-00005-653f05c8b15c22c6.parquet",
    "train-00003-of-00005-1d66a487a8489b7f.parquet",
    "train-00004-of-00005-208b7d689f1c7b70.parquet",
]
TEXT_COLS = ["uid", "image", "MeSH", "Problems", "indication", "findings", "impression"]


def main():
    cfg = yaml.safe_load(open("configs/data.yaml"))
    root = Path(cfg["datasets_dir"]) / "iu"
    raw, meta, out_dir = root / "raw", root / "meta", root / f"images_{cfg['image_size']}"
    for d in (raw, meta, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    reports_csv, info_csv = meta / "iu_reports.csv", root / "image_info.csv"
    skipped = []
    for name in SHARDS:
        marker = out_dir / f".done_{name}"
        if marker.exists():
            continue
        path = raw / name
        print(f"downloading {name}", flush=True)
        subprocess.run(
            [
                "curl",
                "-sSL",
                "-C",
                "-",
                "--retry",
                "10",
                "--retry-all-errors",
                "-o",
                path,
                BASE + name,
            ],
            check=True,
        )
        pf = pq.ParquetFile(path)
        with open(reports_csv, "a", newline="") as fr, open(info_csv, "a", newline="") as fi:
            wr, wi = csv.writer(fr), csv.writer(fi)
            if fr.tell() == 0:
                wr.writerow(TEXT_COLS)
            if fi.tell() == 0:
                wi.writerow(["image_id", "width", "height", "mode"])
            for batch in tqdm(pf.iter_batches(batch_size=32, columns=[*TEXT_COLS, "img_frontal"])):
                for row in batch.to_pylist():
                    if not row["img_frontal"]:
                        skipped.append(row["uid"])
                        continue
                    png, info = to_gray_shortest_side(row["img_frontal"], cfg["image_size"])
                    (out_dir / f"{row['uid']}.png").write_bytes(png)
                    wr.writerow([row[c] for c in TEXT_COLS])
                    wi.writerow([row["uid"], info["width"], info["height"], info["mode"]])
        path.unlink()
        marker.touch()
    print(f"done; {len(skipped)} reports without a frontal image skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
