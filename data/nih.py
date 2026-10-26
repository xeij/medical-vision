"""NIH ChestX-ray14 (Wang et al., 2017): 112,120 frontal CXRs from 30,805 patients, labels NLP-mined
from radiology reports. The official patient-level train_val/test lists shipped with the data are
kept as-is (test = official test)."""

from pathlib import Path

import pandas as pd

from data.labels import nih_labels


def load_nih(meta_dir) -> pd.DataFrame:
    meta_dir = Path(meta_dir)
    raw = pd.read_csv(meta_dir / "Data_Entry_2017_v2020.csv")
    test = set((meta_dir / "test_list.txt").read_text().split())
    train_val = set((meta_dir / "train_val_list.txt").read_text().split())
    df = pd.DataFrame(
        {
            "image_id": raw["Image Index"].str.removesuffix(".png"),
            "patient_id": raw["Patient ID"].astype(str),
            "view": raw["View Position"],
            "age": raw["Patient Age"],
            "sex": raw["Patient Gender"],
            "official_test": raw["Image Index"].isin(test),
        }
    )
    df = pd.concat([df, nih_labels(raw["Finding Labels"])], axis=1)
    if not (raw["Image Index"].isin(train_val) ^ df["official_test"]).all():
        raise ValueError("every image must be in exactly one of train_val_list / test_list")
    overlap = set(df.loc[df.official_test, "patient_id"]) & set(
        df.loc[~df.official_test, "patient_id"]
    )
    if overlap:
        raise ValueError(f"{len(overlap)} patients appear in both official lists")
    return df
