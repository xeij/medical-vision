"""Download NIH ChestX-ray14 (HF mirror of the official NIH Box release) and store every image as
an 8-bit grayscale PNG with shortest side = image_size (configs/data.yaml). Each zip is deleted
after conversion, so peak disk use is one zip (~4 GB) plus the converted PNGs. Re-runnable: zips
already converted (marker file) are skipped and curl resumes partial downloads.
"""

import csv
import subprocess
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import yaml
from tqdm import tqdm

from data.images import to_gray_shortest_side

BASE = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/"
META = ["Data_Entry_2017_v2020.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt"]
ZIPS = [f"images_{i:03d}.zip" for i in range(1, 13)]

_zf = None  # per-worker open zip handle


def _init(zip_path):
    global _zf
    _zf = zipfile.ZipFile(zip_path)


def _convert(args):
    member, out_dir, size = args
    png, info = to_gray_shortest_side(_zf.read(member), size)
    image_id = Path(member).name
    (out_dir / image_id).write_bytes(png)
    return [image_id, info["width"], info["height"], info["mode"]]


def curl(url, dest):
    subprocess.run(
        ["curl", "-sSL", "-C", "-", "--retry", "10", "--retry-all-errors", "-o", dest, url],
        check=True,
    )


def main():
    cfg = yaml.safe_load(open("configs/data.yaml"))
    root = Path(cfg["datasets_dir"]) / "nih"
    meta_dir, out_dir = root / "meta", root / f"images_{cfg['image_size']}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in META:
        if not (meta_dir / f).exists():
            curl(BASE + f, str(meta_dir / f))
    info_csv = root / "image_info.csv"
    for name in ZIPS:
        marker = out_dir / f".done_{name}"
        if marker.exists():
            continue
        zip_path = root / name
        print(f"downloading {name}", flush=True)
        curl(BASE + "images/" + name, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m for m in zf.namelist() if m.startswith("images/") and m.endswith(".png")
            ]  # skip __MACOSX/._ forks
        jobs = [(m, out_dir, cfg["image_size"]) for m in members]
        with Pool(initializer=_init, initargs=(str(zip_path),)) as pool:
            rows = list(tqdm(pool.imap_unordered(_convert, jobs, chunksize=64), total=len(jobs)))
        with open(info_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if fh.tell() == 0:
                w.writerow(["image_id", "width", "height", "mode"])
            w.writerows(sorted(rows))
        zip_path.unlink()
        marker.touch()
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
