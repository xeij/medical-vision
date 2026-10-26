"""Label regimes shared by the frozen-head grid and the fine-tuned baseline: which rows of a
dataset's train/val split a (regime, seed) run may use."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.splits import subsample_patients

DATASETS = ("nih", "iu")
# fraction of ID training labels; "matched" = the other dataset's full training size (NIH only)
REGIMES = {"1pct": 0.01, "10pct": 0.1, "matched": None, "100pct": 1.0}
SEEDS = (0, 1, 2)


def config():
    return yaml.safe_load(open("configs/data.yaml"))


def metadata(cfg, dataset):
    return pd.read_parquet(Path(cfg["artifacts_dir"]) / "metadata" / f"{dataset}.parquet")


def image_paths(cfg, dataset, image_ids):
    d = Path(cfg["datasets_dir"]) / dataset / f"images_{cfg['image_size']}"
    return [d / f"{i}.png" for i in image_ids]


def regime_fraction(regime, train_ds, n_train):
    """Fraction of the train split for a regime, or None when the regime does not apply."""
    if regime != "matched":
        return REGIMES[regime]
    if train_ds != "nih":
        return None
    ood = DATASETS[1 - DATASETS.index(train_ds)]
    return n_train[ood] / n_train[train_ds]


def subset(meta, split, frac, seed):
    """Row indices of `split` for a regime: whole patients, seeded, until frac of the images."""
    rows = np.flatnonzero(meta["split"] == split)
    if frac >= 1:
        return rows
    return rows[subsample_patients(meta.iloc[rows], round(frac * len(rows)), seed)]
