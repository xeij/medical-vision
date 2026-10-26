"""Phase 3 grid. For each (train dataset, encoder, label regime, seed): subsample the train and
val splits by patient, fit the linear probe and build the kNN index on cached embeddings, and
cache the scores of both heads on the in-distribution test split and on the other dataset's test
split (out-of-distribution). Resumable: a run whose predictions file exists is skipped.

    PYTHONPATH=. uv run python scripts/run_grid.py --train iu nih
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.labels import LABELS, PATHOLOGIES
from data.regimes import DATASETS, REGIMES, SEEDS, config, metadata, regime_fraction, subset
from encoders import ENCODERS
from heads.knn import KNN
from heads.probe import fit_probe

KS = (5, 20, 50)


def load(cfg, dataset, encoder):
    d = Path(cfg["artifacts_dir"]) / "embeddings" / dataset
    x = np.load(d / f"{encoder}.npy", mmap_mode="r")
    ids = pd.read_parquet(d / f"{encoder}.parquet")
    return x, ids.merge(metadata(cfg, dataset), on="image_id", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=list(DATASETS))
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="artifacts/results")
    args = ap.parse_args()
    cfg = config()
    preds_dir = Path(args.out) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = Path(args.out) / "runs.csv"
    n_train = {ds: int((metadata(cfg, ds)["split"] == "train").sum()) for ds in DATASETS}
    for train_ds in args.train:
        ood_ds = DATASETS[1 - DATASETS.index(train_ds)]
        for enc in args.encoders:
            x_all, meta = load(cfg, train_ds, enc)
            x_ood_all, meta_ood = load(cfg, ood_ds, enc)
            tests = {}
            for name, x, m in (("id", x_all, meta), ("ood", x_ood_all, meta_ood)):
                rows = np.flatnonzero(m["split"] == "test")
                tests[name] = (np.asarray(x[rows]), m.iloc[rows]["image_id"].to_numpy().astype(str))
            for regime in args.regimes:
                frac = regime_fraction(regime, train_ds, n_train)
                if frac is None:
                    continue
                for seed in args.seeds if frac < 1 else [0]:
                    run_id = f"{train_ds}_{enc}_{regime}_s{seed}"
                    if (preds_dir / f"{run_id}.npz").exists():
                        continue
                    t0 = time.time()
                    tr = subset(meta, "train", frac, seed)
                    va = subset(meta, "val", frac, seed)
                    x_tr, x_va = np.asarray(x_all[tr]), np.asarray(x_all[va])
                    y_tr = meta.iloc[tr][list(LABELS)].to_numpy(np.int8)
                    y_va = meta.iloc[va][list(LABELS)].to_numpy(np.int8)
                    probe = fit_probe(
                        x_tr, y_tr, x_va, y_va, pathology_cols=range(len(PATHOLOGIES))
                    )
                    knn = KNN(x_tr, y_tr)
                    arrays = {}
                    for name, (x_te, ids) in tests.items():
                        arrays[f"{name}_image_id"] = ids
                        arrays[f"{name}_probe"] = probe.predict(x_te)
                        for k, s in knn.scores(x_te, KS).items():
                            arrays[f"{name}_knn{k}"] = s.astype(np.float32)
                    tmp = preds_dir / f"{run_id}.tmp.npz"
                    np.savez(tmp, **arrays)
                    tmp.rename(preds_dir / f"{run_id}.npz")
                    row = {
                        "run_id": run_id,
                        "train_ds": train_ds,
                        "ood_ds": ood_ds,
                        "encoder": enc,
                        "regime": regime,
                        "seed": seed,
                        "n_train": len(tr),
                        "n_val": len(va),
                        "c": probe.c,
                        "c_fallback": probe.c_fallback,
                        **{f"k_eff{k}": knn.k_eff(k) for k in KS},
                        **{f"n_pos_{lab}": int(y_tr[:, j].sum()) for j, lab in enumerate(LABELS)},
                        "fit_seconds": round(time.time() - t0, 1),
                    }
                    pd.DataFrame([row]).to_csv(
                        runs_csv, mode="a", header=not runs_csv.exists(), index=False
                    )
                    print(f"{run_id}: n_train={len(tr)} C={probe.c} {row['fit_seconds']}s")


if __name__ == "__main__":
    main()
