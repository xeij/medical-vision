import pandas as pd

from data.labels import LABELS
from data.nih import load_nih


def test_load_nih_builds_shared_schema(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "Image Index": ["00000001_000.png", "00000001_001.png", "00000002_000.png"],
            "Finding Labels": ["Cardiomegaly", "Cardiomegaly|Emphysema", "No Finding"],
            "Follow-up #": [0, 1, 0],
            "Patient ID": [1, 1, 2],
            "Patient Age": [57, 58, 80],
            "Patient Gender": ["M", "M", "F"],
            "View Position": ["PA", "PA", "AP"],
        }
    ).to_csv(meta / "Data_Entry_2017_v2020.csv", index=False)
    (meta / "train_val_list.txt").write_text("00000001_000.png\n00000001_001.png\n")
    (meta / "test_list.txt").write_text("00000002_000.png\n")

    df = load_nih(meta)

    assert list(df.columns) == [
        "image_id",
        "patient_id",
        "view",
        "age",
        "sex",
        "official_test",
        *LABELS,
    ]
    assert df["image_id"].tolist() == ["00000001_000", "00000001_001", "00000002_000"]
    assert df["patient_id"].tolist() == ["1", "1", "2"]
    assert df["official_test"].tolist() == [False, False, True]
    assert df["cardiomegaly"].tolist() == [1, 1, 0]
    assert df["no_finding"].tolist() == [0, 0, 1]


def test_load_nih_rejects_patient_overlap_between_official_lists(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    pd.DataFrame(
        {
            "Image Index": ["a.png", "b.png"],
            "Finding Labels": ["No Finding"] * 2,
            "Follow-up #": [0, 1],
            "Patient ID": [1, 1],
            "Patient Age": [1, 1],
            "Patient Gender": ["M"] * 2,
            "View Position": ["PA"] * 2,
        }
    ).to_csv(meta / "Data_Entry_2017_v2020.csv", index=False)
    (meta / "train_val_list.txt").write_text("a.png\n")
    (meta / "test_list.txt").write_text("b.png\n")
    try:
        load_nih(meta)
    except ValueError as e:
        assert "patient" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
