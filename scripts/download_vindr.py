"""Download VinDr-CXR 1.0.0 from PhysioNet (credentialed access required) and store every image as
an 8-bit grayscale PNG with shortest side = image_size. Each DICOM is fetched into memory,
converted and discarded, so no raw DICOM is ever written to disk. Re-runnable: PNGs that already
exist are skipped.

Usage: PHYSIONET_USER=... PHYSIONET_PASSWORD=... make data-vindr

NOTE: written against the PhysioNet file layout (SHA256SUMS.txt listing, annotations/*.csv,
train/*.dicom, test/*.dicom) but not yet run on real data; the DICOM conversion is unit-tested.
"""

import base64
import csv
import os
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import yaml
from tqdm import tqdm

from data.images import dicom_to_gray_shortest_side

BASE = "https://physionet.org/files/vindr-cxr/1.0.0/"


def fetch(rel: str) -> bytes:
    token = base64.b64encode(
        f"{os.environ['PHYSIONET_USER']}:{os.environ['PHYSIONET_PASSWORD']}".encode()
    ).decode()
    req = urllib.request.Request(BASE + rel, headers={"Authorization": f"Basic {token}"})
    last = None
    for _ in range(3):
        try:
            return urllib.request.urlopen(req, timeout=300).read()
        except Exception as e:  # noqa: BLE001 - retry any transient network error
            last = e
    raise RuntimeError(f"failed to fetch {rel}: {last}")


def _convert(args):
    rel, out_dir, size = args
    png, info = dicom_to_gray_shortest_side(fetch(rel), size)
    image_id = Path(rel).stem
    (out_dir / f"{image_id}.png").write_bytes(png)
    return [image_id, *info.values()]


def main():
    if not {"PHYSIONET_USER", "PHYSIONET_PASSWORD"} <= os.environ.keys():
        sys.exit("set PHYSIONET_USER and PHYSIONET_PASSWORD (credentialed PhysioNet account)")
    cfg = yaml.safe_load(open("configs/data.yaml"))
    root = Path(cfg["datasets_dir"]) / "vindr"
    meta_dir, out_dir = root / "meta", root / f"images_{cfg['image_size']}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        line.split()[-1] for line in fetch("SHA256SUMS.txt").decode().splitlines() if line.strip()
    ]
    for rel in [p for p in paths if p.startswith("annotations/")]:
        (meta_dir / Path(rel).name).write_bytes(fetch(rel))
    dicoms = [p for p in paths if p.startswith(("train/", "test/"))]
    todo = [p for p in dicoms if not (out_dir / f"{Path(p).stem}.png").exists()]
    print(f"{len(dicoms)} images listed, {len(todo)} to download", flush=True)

    info_csv = root / "image_info.csv"
    new_file = not info_csv.exists()
    with open(info_csv, "a", newline="") as fh, Pool(8) as pool:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["image_id", "width", "height", "mode", "view", "age", "sex"])
        jobs = [(p, out_dir, cfg["image_size"]) for p in todo]
        for row in tqdm(pool.imap_unordered(_convert, jobs), total=len(jobs)):
            w.writerow(
                row
            )  # ponytail: png-then-row; a crash in between loses one csv row, rerun fixes
            fh.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
