import numpy as np
import pandas as pd

from data.splits import assign_splits


def _df(n_patients=100, images_per_patient=3, n_test_patients=20):
    pids = np.repeat(np.arange(n_patients), images_per_patient)
    df = pd.DataFrame(
        {"image_id": [f"img{i}" for i in range(len(pids))], "patient_id": pids.astype(str)}
    )
    df["official_test"] = df["patient_id"].astype(int) < n_test_patients
    return df


def test_official_test_is_kept_and_patients_never_straddle_splits():
    df = _df()
    split = assign_splits(df, val_fraction=0.1, seed=0)
    assert set(split.unique()) == {"train", "val", "test"}
    assert (split[df["official_test"]] == "test").all()
    assert (split[~df["official_test"]] != "test").all()
    by_patient = pd.Series(split.values, index=df["patient_id"].values).groupby(level=0).nunique()
    assert (by_patient == 1).all()


def test_val_fraction_is_over_patients_and_seed_is_deterministic():
    df = _df()
    s0 = assign_splits(df, val_fraction=0.1, seed=0)
    assert df.loc[s0 == "val", "patient_id"].nunique() == 8  # 10% of the 80 non-test patients
    assert s0.equals(assign_splits(df, val_fraction=0.1, seed=0))
    assert not s0.equals(assign_splits(df, val_fraction=0.1, seed=1))


def test_without_official_test_a_test_fraction_of_patients_is_held_out():
    df = _df(n_test_patients=0)
    split = assign_splits(df, val_fraction=0.1, seed=0, test_fraction=0.2)
    n = df.groupby(split)["patient_id"].nunique()
    assert n["test"] == 20 and n["val"] == 8 and n["train"] == 72  # 20% test, then 10% of the rest
    assert (df.groupby("patient_id").apply(lambda d: split[d.index].nunique()) == 1).all()


def test_test_fraction_is_ignored_when_an_official_test_exists():
    df = _df()
    assert assign_splits(df, 0.1, 0, test_fraction=0.5).equals(assign_splits(df, 0.1, 0))


def test_subsample_patients_keeps_patients_whole():
    from data.splits import subsample_patients

    df = pd.DataFrame({"patient_id": np.repeat([f"p{i}" for i in range(40)], [1, 2, 3, 4] * 10)})
    mask = subsample_patients(df, n_images=25, seed=0)
    assert mask.dtype == bool and len(mask) == len(df)
    per_patient = df.assign(m=mask).groupby("patient_id")["m"].agg(["min", "max"])
    assert (per_patient["min"] == per_patient["max"]).all()  # whole patients in or out
    n = mask.sum()
    assert n >= 25 and n < 25 + 4  # reaches the target within one patient's worth of images
    assert (mask == subsample_patients(df, 25, 0)).all()
    assert not (mask == subsample_patients(df, 25, 1)).all()
    assert subsample_patients(df, n_images=10_000, seed=0).all()
