import pandas as pd

from data.iu import load_iu
from data.labels import LABELS


def test_load_iu_builds_shared_schema(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "uid": [1, 2],
            "image": ["Xray Chest PA and Lateral"] * 2,
            "MeSH": ["normal", "Cardiomegaly/mild;Pleural Effusion/left"],
            "Problems": ["normal", "Cardiomegaly;Pleural Effusion"],
            "indication": ["", ""],
            "findings": ["", ""],
            "impression": ["", ""],
        }
    ).to_csv(meta / "iu_reports.csv", index=False)

    df = load_iu(meta)

    assert list(df.columns) == [
        "image_id",
        "patient_id",
        "view",
        "age",
        "sex",
        "official_test",
        *LABELS,
    ]
    assert df["image_id"].tolist() == ["1", "2"] and df["patient_id"].tolist() == ["1", "2"]
    assert df["view"].tolist() == ["PA", "PA"]
    assert not df["official_test"].any()
    assert df["no_finding"].tolist() == [1, 0]
    assert df["cardiomegaly"].tolist() == [0, 1] and df["effusion"].tolist() == [0, 1]


def test_load_iu_drops_unindexed_reports(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "uid": [1, 2],
            "image": ["x", "x"],
            "MeSH": ["normal", "No Indexing"],
            "Problems": ["normal", "No Indexing"],
            "indication": ["", ""],
            "findings": ["", ""],
            "impression": ["", "No active disease."],
        }
    ).to_csv(meta / "iu_reports.csv", index=False)
    assert load_iu(meta)["image_id"].tolist() == ["1"]
