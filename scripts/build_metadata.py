"""Build artifacts/metadata/<dataset>.parquet (shared schema + split column) and the committed
splits/<dataset>.csv for every dataset whose metadata files are on disk."""

from pathlib import Path

import yaml

from data.iu import load_iu
from data.nih import load_nih
from data.splits import assign_splits
from data.vindr import load_vindr

LOADERS = {"nih": load_nih, "iu": load_iu, "vindr": load_vindr}


def main():
    cfg = yaml.safe_load(open("configs/data.yaml"))
    out_dir = Path(cfg["artifacts_dir"]) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, load in LOADERS.items():
        meta_dir = Path(cfg["datasets_dir"]) / name / "meta"
        if not meta_dir.exists():
            print(f"{name}: {meta_dir} missing, skipped (run make data-{name})")
            continue
        df = load(meta_dir)
        df.insert(0, "dataset", name)
        df["split"] = assign_splits(df, cfg["val_fraction"], cfg["seed"], cfg["test_fraction"])
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        df[["image_id", "patient_id", "split"]].to_csv(
            Path(cfg["splits_dir"]) / f"{name}.csv", index=False
        )
        print(name, df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
