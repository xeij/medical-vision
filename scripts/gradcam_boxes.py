"""Phase 4: does the model look where the radiologist drew the box? For every NIH test image
with an expert box for one of the shared labels, compute the class activation map of that label
and score its agreement with the box: pointing game (map maximum inside the box) and fraction of
map energy inside the box, each next to the chance level of a uniform map (box area fraction),
plus IoU of the map thresholded at half its maximum. Models: the fine-tuned ResNet-50 trained on
NIH (in-domain) and on IU (cross-hospital), and the frozen ImageNet ResNet-50 with the NIH- and
IU-trained linear probes. With a global-average-pool + linear head, Grad-CAM at the last block
equals the class activation map, so it is computed in closed form, ReLU(sum_c w_c f_c(x, y)),
with w = probe coefficient / scaler std for the probe. Patient-level bootstrap CIs; models are
compared on the same boxes. Models whose weights are missing are skipped and listed.

    PYTHONPATH=. uv run python scripts/gradcam_boxes.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import interpolate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from data.labels import LABELS, NIH_SOURCE  # noqa: E402
from data.regimes import config, image_paths, metadata, subset  # noqa: E402
from encoders.finetune import EVAL_TF, make_model  # noqa: E402
from eval.localization import area_fraction, energy_inside, iou_at, map_box, pointing_hit  # noqa: E402
from eval.metrics import patient_bootstrap_weights  # noqa: E402
from heads.probe import fit_probe  # noqa: E402

SIZE, SCALE, OFFSET = 224, 256 / 1024, 16  # NIH 1024-px frame -> Resize(256) + CenterCrop(224)
MODELS = {
    "ft_nih": "fine-tuned ResNet-50, trained on NIH (in-domain)",
    "ft_iu": "fine-tuned ResNet-50, trained on IU (cross-hospital)",
    "probe_nih": "frozen ImageNet ResNet-50 + NIH probe (in-domain)",
    "probe_iu": "frozen ImageNet ResNet-50 + IU probe (cross-hospital)",
}
PAIRS = [
    ("ft_nih", "probe_nih"),
    ("ft_iu", "probe_iu"),
    ("ft_iu", "ft_nih"),
    ("probe_iu", "probe_nih"),
]
LABEL_IDX = {lab: j for j, lab in enumerate(LABELS)}


def load_boxes(cfg):
    b = pd.read_csv(Path(cfg["datasets_dir"]) / "nih" / "meta" / "BBox_List_2017.csv")
    b = b.iloc[:, :6].set_axis(["image_id", "finding", "x", "y", "w", "h"], axis=1)
    b["image_id"] = b["image_id"].str.replace(".png", "", regex=False)
    b["label"] = b["finding"].map(NIH_SOURCE)
    b = b.dropna(subset=["label"])
    meta = metadata(cfg, "nih")[["image_id", "patient_id", "split"]]
    b = b.merge(meta, on="image_id", how="left")
    assert (b["split"] == "test").all(), "box images must be in the test split"
    b["box"] = [map_box((r.x, r.y, r.w, r.h), SCALE, OFFSET, SIZE) for r in b.itertuples()]
    dropped = int(b["box"].isna().sum())
    return b.dropna(subset=["box"]).reset_index(drop=True), dropped


def trunk_of(model):
    return nn.Sequential(*list(model.children())[:-2]).eval()


@torch.no_grad()
def feature_maps(trunk, paths, device, batch_size=32):
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([EVAL_TF(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]])
        out.append(trunk(x.to(device)).float().cpu())
    return torch.cat(out)  # (n, 2048, 7, 7)


def cams(fmaps, weights, label_idx):
    """fmaps (n, C, h, w), weights (L, C), label_idx (n,) -> (n, SIZE, SIZE) ReLU'd maps."""
    w = torch.as_tensor(weights, dtype=torch.float32)[label_idx]  # (n, C)
    cam = torch.relu(torch.einsum("nc,nchw->nhw", w, fmaps)).unsqueeze(1)
    return interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[:, 0].numpy()


def probe_weights(cfg, train_ds):
    d = Path(cfg["artifacts_dir"]) / "embeddings" / train_ds
    x = np.load(d / "resnet50.npy", mmap_mode="r")
    meta = pd.read_parquet(d / "resnet50.parquet").merge(metadata(cfg, train_ds), on="image_id")
    tr, va = subset(meta, "train", 1.0, 0), subset(meta, "val", 1.0, 0)
    y = meta[list(LABELS)].to_numpy(np.int8)
    probe = fit_probe(x[tr], y[tr], x[va], y[va], pathology_cols=range(5))
    rows = []
    for m in probe.models:
        rows.append(
            m.coef_[0] / probe.scaler.scale_ if hasattr(m, "coef_") else np.zeros(x.shape[1])
        )
    return np.stack(rows)


def summarise(df, w):
    """Weighted means over bootstrap resamples -> dict of (point, lo, hi) per metric."""
    out = {}
    for col in ("hit", "energy", "iou", "chance"):
        v = df[col].to_numpy()
        boot = (w @ v) / w.sum(1)
        out[col] = (float(v.mean()), *np.percentile(boot, [2.5, 97.5]))
    return out


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def example_figure(boxes, cam_by_model, paths, models, path):
    labels = [lab for lab in LABELS if lab in set(boxes["label"])]
    fig, axes = plt.subplots(
        len(models), len(labels), figsize=(2.6 * len(labels), 2.8 * len(models))
    )
    axes = np.atleast_2d(axes)
    ref = models[0]
    for c, lab in enumerate(labels):
        sel = boxes[boxes["label"] == lab]
        e = pd.Series(
            [energy_inside(cam_by_model[ref][i], boxes.at[i, "box"]) for i in sel.index], sel.index
        )
        i = int((e - e.median()).abs().idxmin())  # the median-agreement box, not a cherry-pick
        img = EVAL_TF(Image.open(paths[i]).convert("RGB"))
        gray = (img[0] * EVAL_TF.std[0] + EVAL_TF.mean[0]).clamp(0, 1).numpy()
        x0, y0, x1, y1 = boxes.at[i, "box"]
        for r, model in enumerate(models):
            ax = axes[r, c]
            cam = cam_by_model[model][i]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam / max(cam.max(), 1e-6), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="#eb6834", lw=1.8))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{lab}\n{boxes.at[i, 'image_id']}", fontsize=9)
            if c == 0:
                ax.set_ylabel(model.replace("_", " "), fontsize=9)
    fig.suptitle(
        "Class activation map (blue) vs expert box (orange), median-agreement example per label",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/results")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--paper-tables", default="paper/tables")
    args = ap.parse_args()
    cfg = config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    boxes, dropped = load_boxes(cfg)
    paths = image_paths(cfg, "nih", boxes["image_id"])
    label_idx = boxes["label"].map(LABEL_IDX).to_numpy()
    boxes["chance"] = [area_fraction(b, (SIZE, SIZE)) for b in boxes["box"]]

    cam_by_model, missing = {}, []
    trunk = trunk_of(make_model(len(LABELS))).to(device)  # ImageNet weights, fc unused
    fm_imagenet = feature_maps(trunk, paths, device)
    for name in ("probe_nih", "probe_iu"):
        cam_by_model[name] = cams(fm_imagenet, probe_weights(cfg, name.split("_")[1]), label_idx)
    for name in ("ft_nih", "ft_iu"):
        wfile = Path(args.results) / "weights" / f"{name.split('_')[1]}_resnet50_ft_100pct_s0.pt"
        if not wfile.exists():
            missing.append(name)
            continue
        model = make_model(len(LABELS))
        model.load_state_dict(torch.load(wfile, map_location="cpu"))
        fm = feature_maps(trunk_of(model).to(device), paths, device)
        cam_by_model[name] = cams(fm, model.fc.weight.detach().numpy(), label_idx)
    models = [m for m in MODELS if m in cam_by_model]

    rows = []
    for model in models:
        for i, b in enumerate(boxes.itertuples()):
            cam = cam_by_model[model][i]
            rows.append(
                {
                    "model": model,
                    "image_id": b.image_id,
                    "patient_id": b.patient_id,
                    "label": b.label,
                    "chance": b.chance,
                    "hit": pointing_hit(cam, b.box),
                    "energy": energy_inside(cam, b.box),
                    "iou": iou_at(cam, b.box),
                }
            )
    per_box = pd.DataFrame(rows)
    per_box.to_csv(Path(args.results) / "gradcam_boxes.csv", index=False)
    w = patient_bootstrap_weights(boxes["patient_id"], args.n_boot, 0)

    out = [
        "# Class activation maps vs expert bounding boxes (NIH test split)",
        "",
        f"{len(boxes)} boxes on {boxes['image_id'].nunique()} images for "
        f"{', '.join(sorted(set(boxes['label'])))} ({dropped} boxes dropped by the centre crop). "
        "Pointing game = map maximum inside the box; energy = fraction of map mass inside the box; "
        "chance = what a uniform map scores on both (box area fraction); IoU@0.5 = overlap of the "
        "box with the map thresholded at half its maximum. Brackets are 95% patient-level "
        "bootstrap intervals; all models are scored on the same boxes and resamples.",
        "",
    ]
    if missing:
        out += [f"Not available (weights missing): {', '.join(missing)}.", ""]
    stats = {}
    for model in models:
        out += [f"## {model}: {MODELS[model]}", ""]
        t = []
        for lab in ["all", *[lab for lab in LABELS if lab in set(boxes["label"])]]:
            sel = per_box[
                (per_box["model"] == model) & ((per_box["label"] == lab) if lab != "all" else True)
            ]
            idx = sel.index % len(boxes)
            s = stats[(model, lab)] = summarise(sel, w[:, idx])
            t.append(
                {
                    "label": lab,
                    "n": len(sel),
                    "pointing game": fmt(s["hit"]),
                    "energy inside": fmt(s["energy"]),
                    "chance": f"{s['chance'][0]:.3f}",
                    "IoU@0.5": f"{s['iou'][0]:.3f}",
                }
            )
        df = pd.DataFrame(t)
        out += ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
        out += [""]
    pairs = [(a, b) for a, b in PAIRS if a in models and b in models]
    if pairs:
        out += [
            "## Paired differences (all boxes)",
            "",
            "| contrast | pointing game | energy inside |",
            "|---|---|---|",
        ]
        for a, b in pairs:
            da = per_box[per_box["model"] == a].reset_index(drop=True)
            db = per_box[per_box["model"] == b].reset_index(drop=True)
            cells = []
            for col in ("hit", "energy"):
                d = (da[col] - db[col]).to_numpy()
                boot = (w @ d) / w.sum(1)
                cells.append(fmt((float(d.mean()), *np.percentile(boot, [2.5, 97.5]))))
            out += [f"| {a} - {b} | {cells[0]} | {cells[1]} |"]
        out += [""]
    labs = [lab for lab in LABELS if lab in set(boxes["label"])]
    short = {
        "ft_nih": "FT, NIH-trained",
        "ft_iu": "FT, IU-trained",
        "probe_nih": "frozen, NIH probe",
        "probe_iu": "frozen, IU probe",
    }

    def stacked(v):
        return (
            f"\\begin{{tabular}}[t]{{@{{}}c@{{}}}}{v[0]:.3f}"
            f"\\\\{{\\tiny[{v[1]:.3f}, {v[2]:.3f}]}}\\end{{tabular}}"
        )

    tex = [
        "\\begin{footnotesize}",
        "\\begin{tabular}{l" + "c" * (2 + len(labs)) + "}",
        "\\toprule",
        f"& & & \\multicolumn{{{len(labs)}}}{{c}}{{pointing game per label}} \\\\",
        f"\\cmidrule(lr){{4-{3 + len(labs)}}}",
        "model & pointing & energy & " + " & ".join(labs) + " \\\\",
        "\\midrule",
    ]
    for model in models:
        s = stats[(model, "all")]
        cells = [short.get(model, model), stacked(s["hit"]), stacked(s["energy"])]
        cells += [f"{stats[(model, lab)]['hit'][0]:.3f}" for lab in labs]
        tex.append(" & ".join(cells) + " \\\\")
    s = stats[(models[0], "all")]
    chance = [f"{s['chance'][0]:.3f}"] * 2
    chance += [f"{stats[(models[0], lab)]['chance'][0]:.3f}" for lab in labs]
    tex += ["\\midrule", "chance (uniform map) & " + " & ".join(chance) + " \\\\"]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{footnotesize}"]
    Path(args.paper_tables).mkdir(parents=True, exist_ok=True)
    Path(args.paper_tables, "gradcam.tex").write_text("\n".join(tex) + "\n")
    fig = Path(args.reports) / "figures" / "gradcam_examples.png"
    example_figure(boxes, cam_by_model, paths, models, fig)
    out += ["## Examples", "", "![examples](figures/gradcam_examples.png)", ""]
    Path(args.reports, "gradcam_boxes.md").write_text("\n".join(out))
    print("\n".join(out[: 6 + 0]))
    print(f"wrote {Path(args.reports) / 'gradcam_boxes.md'}; models: {models}; missing: {missing}")


if __name__ == "__main__":
    main()
