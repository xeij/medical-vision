import pandas as pd

from data.labels import LABELS
from data.vindr import load_vindr

VINDR_COLS = [
    "Aortic enlargement",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Pleural effusion",
    "Pneumothorax",
    "Pneumonia",
    "No finding",
]


def _write(meta, name, rows, with_rad):
    cols = ["image_id", *(["rad_id"] if with_rad else []), *VINDR_COLS]
    pd.DataFrame(rows, columns=cols).to_csv(meta / name, index=False)


def test_load_vindr_votes_train_and_keeps_official_test(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    _write(
        meta,
        "image_labels_train.csv",
        [
            ["t1", "R8", 0, 1, 0, 0, 0, 0, 0, 0],
            ["t1", "R9", 0, 1, 0, 0, 0, 0, 0, 0],
            ["t1", "R10", 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        with_rad=True,
    )
    _write(meta, "image_labels_test.csv", [["s1", 0, 0, 0, 0, 1, 0, 0, 0]], with_rad=False)

    df = load_vindr(meta)

    assert list(df.columns) == [
        "image_id",
        "patient_id",
        "view",
        "age",
        "sex",
        "official_test",
        *LABELS,
    ]
    assert df["patient_id"].tolist() == df["image_id"].tolist()  # no patient ids ship with VinDr
    assert df["official_test"].tolist() == [False, True]
    assert df.set_index("image_id").loc["t1", "atelectasis"] == 1
    assert df.set_index("image_id").loc["s1", "effusion"] == 1
    assert df["view"].tolist() == ["UNK", "UNK"]


def test_load_vindr_merges_dicom_header_info_when_present(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    _write(meta, "image_labels_train.csv", [["t1", "R8", 0, 0, 0, 0, 0, 0, 0, 1]], with_rad=True)
    _write(meta, "image_labels_test.csv", [["s1", 0, 0, 0, 0, 0, 0, 0, 1]], with_rad=False)
    pd.DataFrame(
        {
            "image_id": ["t1"],
            "width": [2500],
            "height": [3000],
            "mode": ["MONOCHROME2"],
            "view": ["PA"],
            "age": ["045Y"],
            "sex": ["F"],
        }
    ).to_csv(tmp_path / "image_info.csv", index=False)

    df = load_vindr(meta).set_index("image_id")

    assert df.loc["t1", "view"] == "PA" and df.loc["t1", "sex"] == "F" and df.loc["t1", "age"] == 45
    assert df.loc["s1", "view"] == "UNK"
