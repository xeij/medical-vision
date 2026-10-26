"""VinDr-CXR (Nguyen et al., 2022), PhysioNet release: 15,000 train + 3,000 test DICOMs from two
Vietnamese hospitals. Train labels: one row per (image, radiologist), 3 radiologists; test labels:
5-radiologist consensus, one row per image. The release ships NO patient identifier (image_id is an
anonymised SOP Instance UID), so patient_id := image_id and splits are necessarily image-level."""

from pathlib import Path

import pandas as pd

from data.labels import vindr_labels


def load_vindr(meta_dir) -> pd.DataFrame:
    meta_dir = Path(meta_dir)
    parts = []
    for name, is_test in [("image_labels_train.csv", False), ("image_labels_test.csv", True)]:
        part = vindr_labels(pd.read_csv(meta_dir / name))
        part["official_test"] = is_test
        parts.append(part)
    labels = pd.concat(parts, ignore_index=True)
    labels["image_id"] = labels["image_id"].astype(str)

    # DICOM header fields written by scripts/download_vindr.py; absent until images are converted.
    info_csv = meta_dir.parent / "image_info.csv"
    if info_csv.exists():
        info = pd.read_csv(info_csv, dtype={"image_id": str})[["image_id", "view", "age", "sex"]]
        info["age"] = pd.to_numeric(
            info["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    else:
        info = pd.DataFrame(columns=["image_id", "view", "age", "sex"])
    df = labels[["image_id"]].merge(info, on="image_id", how="left")
    df["view"] = df["view"].fillna("UNK")
    df["patient_id"] = df["image_id"]
    df["official_test"] = labels["official_test"]
    cols = ["image_id", "patient_id", "view", "age", "sex", "official_test"]
    return pd.concat([df[cols], labels.drop(columns=["image_id", "official_test"])], axis=1)
