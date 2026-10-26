"""Indiana University chest X-ray collection (Demner-Fushman et al., 2016), served by NLM Open-i and
taken here from the public HF mirror of raddar's Kaggle conversion. One frontal image per report;
labels are the manually assigned MeSH major terms of the report. No patient identifier is released
(one report = one de-identified visit), so patient_id := report uid. No official split exists."""

from pathlib import Path

import pandas as pd

from data.labels import IU_UNINDEXED, iu_labels


def load_iu(meta_dir) -> pd.DataFrame:
    raw = pd.read_csv(Path(meta_dir) / "iu_reports.csv", dtype={"uid": str})
    raw = raw[raw["MeSH"].str.lower().str.strip() != IU_UNINDEXED].reset_index(drop=True)
    df = pd.DataFrame(
        {
            "image_id": raw["uid"],
            "patient_id": raw["uid"],
            "view": "PA",  # frontal by construction (hand-labelled in the source conversion)
            "age": pd.Series(pd.NA, index=raw.index, dtype="Float64"),
            "sex": pd.Series(pd.NA, index=raw.index, dtype="string"),
            "official_test": False,
        }
    )
    return pd.concat([df, iu_labels(raw["MeSH"])], axis=1)
