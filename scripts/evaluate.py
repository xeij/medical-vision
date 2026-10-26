"""Phase 3 evaluation from cached predictions. For every (run, head): AUROC, AUPRC and ECE per
label and macro over the five pathologies, on the in-distribution (ID) and out-of-distribution
(OOD) test sets, with 95% patient-level bootstrap intervals (same resampled patients for every
run, so heads and encoders can be compared paired) and the degradation gap. The gap is signed so
positive always means worse OOD: ID - OOD for AUROC and AUPRC, OOD - ID for ECE.
Resumable: (run, head) pairs already in metrics.csv are skipped.

    PYTHONPATH=. uv run python scripts/evaluate.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.labels import LABELS, PATHOLOGIES
from eval.metrics import ece, macro, patient_bootstrap_weights, rank_metrics

HEADS = ("probe", "knn5", "knn20", "knn50", "ft")  # ft = fine-tuned CNN (Phase 4)
METRICS = ("auroc", "auprc", "ece")
SIGN = np.array([1, 1, -1])  # gap = SIGN * (id - ood)
ROWS = (*LABELS, "macro")
PATH = list(range(len(PATHOLOGIES)))


def evaluate_set(y, s, w):
    """-> point (labels+1, 3), boot (B, labels+1, 3); last row is the macro over pathologies."""
    point = np.full((len(ROWS), 3), np.nan)
    boot = np.full((len(w), len(ROWS), 3), np.nan)
    for j in range(y.shape[1]):
        point[j, :2] = rank_metrics(y[:, j], s[:, j])
        point[j, 2] = ece(y[:, j], s[:, j])
        boot[:, j, :2] = np.stack(rank_metrics(y[:, j], s[:, j], w), axis=1)
        boot[:, j, 2] = ece(y[:, j], s[:, j], w)
    point[-1] = macro(point[PATH])
    boot[:, -1] = macro(boot[:, PATH].transpose(1, 0, 2))
    return point, boot


def ci(boot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(boot, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/data.yaml"))
    results = Path(args.results)
    tests = {}
    for ds in ("nih", "iu"):
        m = pd.read_parquet(f"{cfg['artifacts_dir']}/metadata/{ds}.parquet")
        m = m[m["split"] == "test"]
        tests[ds] = {
            "ids": m["image_id"].to_numpy().astype(str),
            "y": m[list(LABELS)].to_numpy(np.int8),
            "w": patient_bootstrap_weights(m["patient_id"], args.n_boot, args.boot_seed),
        }
    cols = ["run_id", "train_ds", "ood_ds", "encoder", "regime", "seed"]
    runs = pd.concat(
        [
            pd.read_csv(f)[cols]
            for f in (results / "runs.csv", results / "runs_ft.csv")
            if f.exists()
        ]
    ).drop_duplicates("run_id", keep="last")
    metrics_csv = results / "metrics.csv"
    done = set()
    if metrics_csv.exists():
        done = set(
            map(tuple, pd.read_csv(metrics_csv)[["run_id", "head"]].drop_duplicates().values)
        )
    boot_dir = results / "boot"
    boot_dir.mkdir(exist_ok=True)
    for run in runs.itertuples():
        z = np.load(results / "preds" / f"{run.run_id}.npz")
        todo = [h for h in HEADS if f"id_{h}" in z.files and (run.run_id, h) not in done]
        if not todo:
            continue
        sets = {"id": run.train_ds, "ood": run.ood_ds}
        for name, ds in sets.items():
            assert (z[f"{name}_image_id"] == tests[ds]["ids"]).all(), run.run_id
        for head in todo:
            res = {
                n: evaluate_set(tests[d]["y"], z[f"{n}_{head}"], tests[d]["w"])
                for n, d in sets.items()
            }
            gap_point = SIGN * (res["id"][0] - res["ood"][0])
            gap_boot = SIGN * (res["id"][1] - res["ood"][1])
            np.savez(
                boot_dir / f"{run.run_id}_{head}.npz",
                id=res["id"][1].astype(np.float32),
                ood=res["ood"][1].astype(np.float32),
                gap=gap_boot.astype(np.float32),
                rows=np.array(ROWS),
                metrics=np.array(METRICS),
            )
            cis = {"id": ci(res["id"][1]), "ood": ci(res["ood"][1]), "gap": ci(gap_boot)}
            points = {"id": res["id"][0], "ood": res["ood"][0], "gap": gap_point}
            rows = []
            for r, label in enumerate(ROWS):
                for m, metric in enumerate(METRICS):
                    row = {
                        "run_id": run.run_id,
                        "train_ds": run.train_ds,
                        "ood_ds": run.ood_ds,
                        "encoder": run.encoder,
                        "regime": run.regime,
                        "seed": run.seed,
                        "head": head,
                        "label": label,
                        "metric": metric,
                    }
                    for k in ("id", "ood", "gap"):
                        row[k] = round(float(points[k][r, m]), 4)
                        row[f"{k}_lo"] = round(float(cis[k][0, r, m]), 4)
                        row[f"{k}_hi"] = round(float(cis[k][1, r, m]), 4)
                    for name, ds in sets.items():
                        row[f"n_pos_{name}"] = (
                            int(tests[ds]["y"][:, r].sum()) if r < len(LABELS) else ""
                        )
                    rows.append(row)
            pd.DataFrame(rows).to_csv(
                metrics_csv, mode="a", header=not metrics_csv.exists(), index=False
            )
            a = points["id"][-1, 0], points["ood"][-1, 0], points["gap"][-1, 0]
            print(f"{run.run_id} {head}: macro AUROC id={a[0]:.3f} ood={a[1]:.3f} gap={a[2]:+.3f}")


if __name__ == "__main__":
    main()
