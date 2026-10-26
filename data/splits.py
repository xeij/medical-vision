"""Patient-level split assignment. 'test' is the dataset's official test list when it has one,
otherwise a random `test_fraction` of patients; the remaining patients are split into train/val by
shuffling *patients* with a fixed seed."""

import numpy as np
import pandas as pd


def assign_splits(
    df: pd.DataFrame, val_fraction: float, seed: int, test_fraction: float = 0.0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    is_test = df["official_test"].to_numpy().copy()
    if not is_test.any() and test_fraction > 0:
        patients = np.sort(df["patient_id"].unique())
        test = set(rng.choice(patients, round(len(patients) * test_fraction), replace=False))
        is_test = df["patient_id"].isin(test).to_numpy()
    pool = np.sort(df.loc[~is_test, "patient_id"].unique())
    val = set(rng.choice(pool, round(len(pool) * val_fraction), replace=False))
    split = np.where(is_test, "test", np.where(df["patient_id"].isin(val), "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def subsample_patients(df: pd.DataFrame, n_images: int, seed: int) -> np.ndarray:
    """Boolean mask selecting whole patients, in a seeded random order, until at least
    `n_images` images are included. Used for the 1% / 10% / matched-size label regimes."""
    patients = np.sort(df["patient_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    counts = df["patient_id"].value_counts()
    n_keep = int(np.searchsorted(np.cumsum(counts[patients].to_numpy()), n_images) + 1)
    return df["patient_id"].isin(patients[:n_keep]).to_numpy()
