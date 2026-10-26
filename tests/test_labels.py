import pandas as pd

from data.labels import LABELS, nih_labels, vindr_labels


def test_shared_label_set():
    assert LABELS == (
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "effusion",
        "pneumothorax",
        "no_finding",
    )


def test_nih_pipe_string_maps_to_binary_columns():
    out = nih_labels(pd.Series(["Cardiomegaly|Effusion|Emphysema", "No Finding", "Infiltration"]))
    assert list(out.columns) == list(LABELS)
    assert out.iloc[0].tolist() == [0, 1, 0, 1, 0, 0]
    assert out.iloc[1].tolist() == [0, 0, 0, 0, 0, 1]
    assert out.iloc[2].tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # out-of-scope finding: neither normal nor positive


def _vindr_rows():
    cols = [
        "image_id",
        "rad_id",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Pleural effusion",
        "Pneumothorax",
        "Nodule/Mass",
        "No finding",
    ]
    rows = [
        ["a", "R1", 1, 0, 0, 1, 0, 0, 0],
        ["a", "R2", 1, 0, 0, 0, 0, 1, 0],
        ["a", "R3", 0, 0, 0, 0, 0, 0, 1],
        ["b", "R1", 0, 0, 0, 0, 0, 0, 1],  # test-style: one row per image
    ]
    return pd.DataFrame(rows, columns=cols)


def test_vindr_majority_vote_per_image():
    out = vindr_labels(_vindr_rows())
    assert list(out.columns) == ["image_id", *LABELS]
    a = out.set_index("image_id").loc["a"].tolist()
    assert a == [1, 0, 0, 0, 0, 0]  # atelectasis 2/3 -> 1, effusion 1/3 -> 0, no finding 1/3 -> 0
    assert out.set_index("image_id").loc["b"].tolist() == [0, 0, 0, 0, 0, 1]


def test_vindr_missing_source_column_raises():
    try:
        vindr_labels(_vindr_rows().drop(columns=["Pneumothorax"]))
    except KeyError as e:
        assert "Pneumothorax" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_vindr_positive_pathology_overrides_no_finding():
    df = _vindr_rows().iloc[[3]].copy()
    df["Pneumothorax"] = 1  # inconsistent rater: No finding + a lesion
    out = vindr_labels(df).set_index("image_id").loc["b"]
    assert out["pneumothorax"] == 1 and out["no_finding"] == 0


def test_iu_mesh_terms_map_before_modifiers_case_insensitive():
    from data.labels import iu_labels

    out = iu_labels(
        pd.Series(
            [
                "normal",
                "Cardiomegaly/borderline;Pulmonary Artery/enlarged",
                "Pulmonary Atelectasis/base/left;Pleural Effusion/right/small",
                "pneumothorax/left;Consolidation/lower lobe",
                "Opacity/lung/base",
                "No Indexing",
            ]
        )
    )
    assert list(out.columns) == list(LABELS)
    assert out.values.tolist() == [
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]


def test_iu_modifier_dependent_and_compound_terms():
    from data.labels import iu_labels

    out = iu_labels(
        pd.Series(
            ["Cardiac Shadow/enlarged/mild", "Cardiac Shadow/obscured", "Hydropneumothorax/right"]
        )
    )
    assert out["cardiomegaly"].tolist() == [1, 0, 0]
    assert out["pneumothorax"].tolist() == [0, 0, 1]
