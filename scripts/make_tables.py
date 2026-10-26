"""Phase 3/4 tables and figures from artifacts/results/metrics.csv and the bootstrap replicates.
Writes reports/results.md, reports/results_macro.csv, reports/figures/*.png and the LaTeX tables
under paper/tables/. Point estimates
and 95% CIs are the seed-0 run; the spread over training subsets is reported separately as the
standard deviation of the point estimate over seeds (1%, 10% and matched regimes only). The
fine-tuned ResNet-50 (encoder "resnet50_ft", head "ft") is drawn as a baseline in every panel.

    PYTHONPATH=. uv run python scripts/make_tables.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.labels import LABELS  # noqa: E402

FT = "resnet50_ft"
HEADS = ["probe", "knn5", "knn20", "knn50", "ft"]
HEAD_LABEL = {
    "probe": "linear probe",
    "knn5": "kNN k=5",
    "knn20": "kNN k=20",
    "knn50": "kNN k=50",
    "ft": "fine-tuned ResNet-50",
}
HEAD_COLOR = {
    "probe": "#eb6834",
    "knn5": "#6da7ec",
    "knn20": "#2a78d6",
    "knn50": "#184f95",
    "ft": "#1baf7a",
}
ENCODERS = ["resnet50", "dinov2", "biomedclip", "rad_dino", FT]
FROZEN = ENCODERS[:4]
ENC_LABEL = {
    "resnet50": "ResNet-50 (ImageNet)",
    "dinov2": "DINOv2 ViT-B",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    FT: "ResNet-50 fine-tuned",
}
REGIMES = ["1pct", "10pct", "matched", "100pct"]
REGIME_LABEL = {"1pct": "1%", "10pct": "10%", "matched": "matched", "100pct": "100%"}
DIRECTIONS = [("nih", "iu"), ("iu", "nih")]
METRICS = ["auroc", "auprc", "ece"]
METRIC_LABEL = {"auroc": "AUROC", "auprc": "AUPRC", "ece": "ECE"}
KEY = ["train_ds", "encoder", "head", "regime", "metric"]


def md_table(df):
    df = df.fillna("")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt(p, lo, hi):
    return "n/a" if np.isnan(p) else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def combos(df):
    """(encoder, head) pairs present, frozen encoders x heads first, fine-tuned last."""
    present = set(zip(df["encoder"], df["head"], strict=True))
    out = [(e, h) for e in FROZEN for h in HEADS[:4] if (e, h) in present]
    return out + ([(FT, "ft")] if (FT, "ft") in present else [])


def regimes_of(df, train_ds):
    return [r for r in REGIMES if r in set(df[df["train_ds"] == train_ds]["regime"])]


def cell(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    return sel.iloc[0] if len(sel) else None


def macro_table(tidy, train_ds, metric, n_train):
    t = tidy[(tidy["train_ds"] == train_ds) & (tidy["metric"] == metric)]
    rows = []
    for enc, head in combos(t):
        for regime in regimes_of(t, train_ds):
            r = cell(t, encoder=enc, head=head, regime=regime)
            if r is None:
                continue
            rows.append(
                {
                    "encoder": ENC_LABEL[enc],
                    "head": HEAD_LABEL[head],
                    "regime": f"{REGIME_LABEL[regime]} (n={n_train[(train_ds, regime)]:,})",
                    "ID": fmt(r["id"], r["id_lo"], r["id_hi"]),
                    "OOD": fmt(r["ood"], r["ood_lo"], r["ood_hi"]),
                    "gap": fmt(r["gap"], r["gap_lo"], r["gap_hi"]),
                    "gap sd (seeds)": "" if r["n_seeds"] < 2 else f"{r['gap_sd']:.3f}",
                }
            )
    return md_table(pd.DataFrame(rows))


def label_table(m, train_ds):
    t = m[(m["train_ds"] == train_ds) & (m["regime"] == "100pct") & (m["metric"] == "auroc")]
    rows = []
    for enc, head in combos(t):
        row = {"encoder": ENC_LABEL[enc], "head": HEAD_LABEL[head]}
        for lab in LABELS:
            r = cell(t, encoder=enc, head=head, label=lab)
            row[lab] = f"{r['id']:.3f} / {r['ood']:.3f}"
        rows.append(row)
    return md_table(pd.DataFrame(rows))


def _boot(results, run_id, head, key):
    return np.load(results / "boot" / f"{run_id}_{head}.npz")[key][:, -1, 0]  # macro AUROC


def head_contrast(results, tidy, train_ds):
    """gap(probe) - gap(kNN) on the ID-vs-OOD macro AUROC gap, paired bootstrap."""
    t = tidy[(tidy["train_ds"] == train_ds) & (tidy["metric"] == "auroc")]
    rows = []
    for regime in regimes_of(t, train_ds):
        for enc in FROZEN:
            run_id = f"{train_ds}_{enc}_{regime}_s0"
            gp = _boot(results, run_id, "probe", "gap")
            sel = t[(t["encoder"] == enc) & (t["regime"] == regime)].set_index("head")["gap"]
            row = {"encoder": ENC_LABEL[enc], "regime": REGIME_LABEL[regime]}
            for head in HEADS[1:4]:
                gk = _boot(results, run_id, head, "gap")
                lo, hi = np.nanpercentile(gp - gk, [2.5, 97.5])
                row[f"probe - {HEAD_LABEL[head]}"] = fmt(sel["probe"] - sel[head], lo, hi)
            rows.append(row)
    return md_table(pd.DataFrame(rows))


def transfer_gaps(results, tidy, target_ds, source_ds):
    """Same-test-set gap on the target test split: in-domain ceiling (train target, 100% labels)
    minus the source-trained model at each regime. Values: (ceiling, transferred, gap, lo, hi,
    per-replicate differences), so heads can be contrasted on the same resamples."""
    t = tidy[tidy["metric"] == "auroc"]
    out = {}
    for regime in regimes_of(t, source_ds):
        for enc, head in combos(t):
            c = cell(t, train_ds=target_ds, encoder=enc, head=head, regime="100pct")
            s = cell(t, train_ds=source_ds, encoder=enc, head=head, regime=regime)
            if c is None or s is None:
                continue
            diff = _boot(results, f"{target_ds}_{enc}_100pct_s0", head, "id") - _boot(
                results, f"{source_ds}_{enc}_{regime}_s0", head, "ood"
            )
            lo, hi = np.nanpercentile(diff, [2.5, 97.5])
            out[(regime, enc, head)] = (c["id"], s["ood"], c["id"] - s["ood"], lo, hi, diff)
    return out


def transfer_table(gaps, regs, n_train, source_ds):
    rows = []
    for enc, head in combos(pd.DataFrame([{"encoder": e, "head": h} for _, e, h in gaps])):
        row = {"encoder": ENC_LABEL[enc], "head": HEAD_LABEL[head], "in-domain ceiling": ""}
        for regime in regs:
            if (regime, enc, head) not in gaps:
                continue
            c, tr, g, lo, hi, _ = gaps[(regime, enc, head)]
            row["in-domain ceiling"] = f"{c:.3f}"
            n = n_train[(source_ds, regime)]
            row[f"{REGIME_LABEL[regime]} (n={n:,})"] = f"{tr:.3f} (gap {fmt(g, lo, hi)})"
        rows.append(row)
    return md_table(pd.DataFrame(rows))


def transfer_contrast(gaps, regs):
    """Paired contrasts of the transfer gap: probe minus each kNN head per frozen encoder, and
    the fine-tuned ResNet-50 minus the probe and minus kNN k=50 (positive: the first loses more
    of its in-domain ceiling)."""
    rows = []
    for regime in regs:
        for enc in FROZEN:
            if (regime, enc, "probe") not in gaps:
                continue
            row = {"encoder": ENC_LABEL[enc], "regime": REGIME_LABEL[regime]}
            gp, dp = gaps[(regime, enc, "probe")][2], gaps[(regime, enc, "probe")][5]
            for head in HEADS[1:4]:
                gk, dk = gaps[(regime, enc, head)][2], gaps[(regime, enc, head)][5]
                lo, hi = np.nanpercentile(dp - dk, [2.5, 97.5])
                row[f"probe - {HEAD_LABEL[head]}"] = fmt(gp - gk, lo, hi)
            if (regime, FT, "ft") in gaps:
                gf, df_ = gaps[(regime, FT, "ft")][2], gaps[(regime, FT, "ft")][5]
                for head in ("probe", "knn50"):
                    gk, dk = gaps[(regime, enc, head)][2], gaps[(regime, enc, head)][5]
                    lo, hi = np.nanpercentile(df_ - dk, [2.5, 97.5])
                    row[f"fine-tuned - {HEAD_LABEL[head]}"] = fmt(gf - gk, lo, hi)
            rows.append(row)
    return md_table(pd.DataFrame(rows))


TEX_ENC = {
    "resnet50": "ResNet-50",
    "dinov2": "DINOv2",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    FT: "ResNet-50",
}
TEX_HEAD = {
    "probe": "probe",
    "knn5": "kNN$_{5}$",
    "knn20": "kNN$_{20}$",
    "knn50": "kNN$_{50}$",
    "ft": "fine-tuned",
}


TEX_ENC = {
    "resnet50": "ResNet-50",
    "dinov2": "DINOv2",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    FT: "ResNet-50",
}
TEX_HEAD = {
    "probe": "probe",
    "knn5": "kNN$_{5}$",
    "knn20": "kNN$_{20}$",
    "knn50": "kNN$_{50}$",
    "ft": "fine-tuned",
}


def tex(s):
    return str(s).replace("%", "\\%").replace("&", "\\&")


def stacked(top, lo, hi):
    """Two-line table cell: the value on top, its interval below in a smaller face."""
    if np.isnan(lo):
        return top
    return (
        "\\begin{tabular}[t]{@{}c@{}}" + top + f"\\\\{{\\tiny[{lo:.3f}, {hi:.3f}]}}\\end{{tabular}}"
    )


def latex_table(header, rows, align, path, pre=(), size="footnotesize"):
    lines = [
        f"\\begin{{{size}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        *pre,
        " & ".join(tex(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{size}}}"]
    path.write_text("\n".join(lines) + "\n")


def latex_transfer_table(gaps, regs, n_train, source_ds, path):
    """Paper table: transferred macro AUROC with the gap to the in-domain ceiling, for the probe,
    kNN k=50 and the fine-tuned network."""
    rows = []
    for enc, head in combos(pd.DataFrame([{"encoder": e, "head": h} for _, e, h in gaps])):
        if head not in ("probe", "knn50", "ft"):
            continue
        row = [TEX_ENC[enc], TEX_HEAD[head], f"{gaps[(regs[0], enc, head)][0]:.3f}"]
        for regime in regs:
            if (regime, enc, head) not in gaps:
                row.append("")
                continue
            _, tr, g, lo, hi, _ = gaps[(regime, enc, head)]
            row.append(stacked(f"{tr:.3f} ({g:+.3f})", lo, hi))
        rows.append(row)
    header = ["encoder", "head", "ceiling"]
    header += [f"{REGIME_LABEL[r]} ({n_train[(source_ds, r)]:,})" for r in regs]
    latex_table(header, rows, "ll" + "c" * (len(regs) + 1), path)


def latex_id_ood_table(tidy, path):
    """Appendix table: the gap as specified (ID vs OOD) at 100% labels, both directions."""
    t = tidy[(tidy["metric"] == "auroc") & (tidy["regime"] == "100pct")]
    rows = []
    for enc, head in combos(t):
        row = [TEX_ENC[enc], TEX_HEAD[head]]
        for train_ds, _ in DIRECTIONS:
            r = cell(t, train_ds=train_ds, encoder=enc, head=head)
            row += [f"{r['id']:.3f}", f"{r['ood']:.3f}"]
            row.append(stacked(f"{r['gap']:+.3f}", r["gap_lo"], r["gap_hi"]))
        rows.append(row)
    pre = [
        "& & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{trained on {ds.upper()}}}" for ds, _ in DIRECTIONS)
        + " \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
    ]
    header = ["encoder", "head"] + ["ID", "OOD", "ID $-$ OOD"] * 2
    latex_table(header, rows, "ll" + "ccc" * 2, path, pre)


def _panels(regs, n_train, ds, ylabel, title, series, path):
    """One panel per frozen encoder; `series(enc, head)` -> (point, lo, hi) arrays over regs or
    None. The fine-tuned baseline is drawn in every panel."""
    fig, axes = plt.subplots(1, len(FROZEN), figsize=(3.1 * len(FROZEN), 3.6), sharey=True)
    offsets = np.linspace(-0.2, 0.2, len(HEADS))
    for ax, enc in zip(axes, FROZEN, strict=True):
        ax.axhline(0, color="#9a9a96", lw=1, zorder=0)
        for off, head in zip(offsets, HEADS, strict=True):
            g = series(FT if head == "ft" else enc, head)
            if g is None:
                continue
            x = np.arange(len(regs)) + off
            ax.errorbar(
                x,
                g[0],
                yerr=[g[0] - g[1], g[2] - g[0]],
                color=HEAD_COLOR[head],
                lw=2,
                marker="s" if head == "ft" else "o",
                ms=5,
                capsize=2,
                label=HEAD_LABEL[head],
                ls="--" if head == "ft" else "-",
            )
        ax.set_title(ENC_LABEL[enc], fontsize=10)
        ax.set_xticks(range(len(regs)))
        ax.set_xticklabels([f"{REGIME_LABEL[r]}\nn={n_train[(ds, r)]:,}" for r in regs], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(ylabel)
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes), key=lambda t: len(t[0]))
    fig.legend(handles, labels, loc="lower center", ncol=len(HEADS), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gap_figure(tidy, train_ds, ood_ds, n_train, path):
    t = tidy[(tidy["train_ds"] == train_ds) & (tidy["metric"] == "auroc")]
    regs = regimes_of(t, train_ds)

    def series(enc, head):
        s = t[(t["encoder"] == enc) & (t["head"] == head)].set_index("regime").reindex(regs)
        return None if s["gap"].isna().all() else (s["gap"], s["gap_lo"], s["gap_hi"])

    _panels(
        regs,
        n_train,
        train_ds,
        "macro AUROC degradation\n(ID minus OOD, 95% CI)",
        f"Train on {train_ds.upper()}, held out {ood_ds.upper()}: macro AUROC gap by label regime",
        series,
        path,
    )


def transfer_figure(gaps, regs, n_train, target_ds, source_ds, path):
    def series(enc, head):
        if not any((r, enc, head) in gaps for r in regs):
            return None
        g = np.array([gaps.get((r, enc, head), (np.nan,) * 5)[2:5] for r in regs])
        return g[:, 0], g[:, 1], g[:, 2]

    _panels(
        regs,
        n_train,
        source_ds,
        "shortfall vs in-domain model\n(macro AUROC, same test set, 95% CI)",
        f"{target_ds.upper()} test set: in-domain ceiling minus {source_ds.upper()}-trained model, "
        "by source label regime",
        series,
        path,
    )


def dumbbell_figure(tidy, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    ylabels = []
    for ax, (train_ds, ood_ds) in zip(axes, DIRECTIONS, strict=True):
        t = tidy[(tidy["train_ds"] == train_ds) & (tidy["metric"] == "auroc")]
        t = t[t["regime"] == "100pct"]
        ylabels, y, prev = [], 0, None
        for enc, head in combos(t):
            r = cell(t, encoder=enc, head=head)
            if prev is not None and enc != prev:
                ax.axhline(y - 0.5, color="#e6e6e3", lw=0.8)
            prev = enc
            ax.plot([r["ood"], r["id"]], [y, y], color="#c3c2b7", lw=2, zorder=1)
            ax.plot([r["id_lo"], r["id_hi"]], [y, y], color="#184f95", lw=1, alpha=0.6, zorder=2)
            ax.plot([r["ood_lo"], r["ood_hi"]], [y, y], color="#6da7ec", lw=1, alpha=0.8, zorder=2)
            ax.plot(
                r["id"],
                y,
                "o",
                color="#184f95",
                ms=7,
                zorder=3,
                label="ID test" if y == 0 else None,
            )
            ax.plot(
                r["ood"],
                y,
                "o",
                color="#6da7ec",
                ms=7,
                zorder=3,
                label="OOD test" if y == 0 else None,
            )
            ylabels.append(f"{ENC_LABEL[enc]}  |  {HEAD_LABEL[head]}")
            y += 1
        ax.set_title(
            f"Train on {train_ds.upper()}, held out {ood_ds.upper()} (100% labels)", fontsize=10
        )
        ax.set_xlabel("macro AUROC (95% CI)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ylabels)))
    axes[0].set_yticklabels(ylabels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(
        fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/results")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--paper-tables", default="paper/tables")
    args = ap.parse_args()
    results, reports = Path(args.results), Path(args.reports)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    tables = Path(args.paper_tables)
    tables.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(results / "metrics.csv")
    runs = pd.concat(
        [pd.read_csv(f) for f in (results / "runs.csv", results / "runs_ft.csv") if f.exists()]
    )
    n_train = runs.groupby(["train_ds", "regime"])["n_train"].median().astype(int).to_dict()
    macro = m[m["label"] == "macro"]
    spread = (
        macro.groupby(KEY)
        .agg(
            gap_sd=("gap", "std"),
            id_sd=("id", "std"),
            ood_sd=("ood", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    tidy = macro[macro["seed"] == 0].merge(spread, on=KEY)
    tidy.to_csv(reports / "results_macro.csv", index=False)
    frozen_runs = runs[runs["encoder"] != FT]
    fb = frozen_runs[frozen_runs["c_fallback"].astype(bool)]
    ft_runs = runs[runs["encoder"] == FT]

    out = [
        "# Results: frozen encoders with linear probe or kNN retrieval vs a fine-tuned CNN, "
        "across hospitals",
        "",
        "ID = the training dataset's patient-level test split; OOD = the other dataset's test "
        "split. Macro = mean over the five pathologies (no_finding reported per label only). "
        "gap = ID - OOD for AUROC and AUPRC and OOD - ID for ECE, so positive is always "
        "degradation. Brackets are 95% patient-level bootstrap intervals (1,000 resamples, the "
        "same resampled patients for every run). Point estimates are the seed-0 training subset; "
        "'gap sd (seeds)' is the standard deviation of the point gap over three subsets. 'matched' "
        "trains NIH on as many images as IU's full training split. These are research "
        "measurements on public data; no clinical utility is claimed.",
        "",
        f"Frozen-head runs: {len(frozen_runs)}. Probe C grid {sorted(frozen_runs['c'].unique())}; "
        f"{len(fb)} runs fell back to the default C because validation had no positives "
        f"({', '.join(sorted(set(fb['regime'])))}). kNN k is clipped to the index size where the "
        f"training subset is smaller than k (min k_eff50 = {frozen_runs['k_eff50'].min()}).",
        "",
    ]
    if len(ft_runs):
        out += [
            f"Fine-tuned ResNet-50 runs: {len(ft_runs)} (ImageNet init, all layers, BCE, AdamW "
            f"lr {ft_runs['lr'].iloc[0]:g}, batch {int(ft_runs['batch_size'].iloc[0])}, early "
            "stopping on validation macro AUROC). Best epoch / epochs run per run: "
            + ", ".join(
                f"{r.run_id.replace('_resnet50_ft', '')}: {int(r.best_epoch)}/{int(r.epochs)}"
                for r in ft_runs.itertuples()
            )
            + ".",
            "",
        ]
    out += [
        "## Same-test-set transfer gap (primary)",
        "",
        "The ID-vs-OOD gap below compares one model on two different test sets, so it also "
        "measures how hard each benchmark is: the IU test split (manual MeSH labels, mostly "
        "normal studies) is easier than the NIH official test split (NLP-mined labels, skewed "
        "towards pathology), and most NIH-trained models score higher on IU than on NIH. The "
        "transfer gap fixes the test set instead: on the target's test split, the in-domain model "
        "(trained on the target, 100% labels) minus the source-trained model at each label "
        "regime. Positive means the transferred model falls short of the in-domain ceiling. "
        "Both models are evaluated on the same bootstrap resamples, so the CI is paired.",
        "",
    ]
    for target_ds, source_ds in DIRECTIONS:
        regs = regimes_of(tidy, source_ds)
        gaps = transfer_gaps(results, tidy, target_ds, source_ds)
        fig_name = f"transfer_auroc_target_{target_ds}.png"
        transfer_figure(gaps, regs, n_train, target_ds, source_ds, reports / "figures" / fig_name)
        latex_transfer_table(gaps, regs, n_train, source_ds, tables / f"transfer_{target_ds}.tex")
        out += [
            f"### Target {target_ds.upper()} test set, source {source_ds.upper()}",
            "",
            f"![transfer](figures/{fig_name})",
            "",
            "Macro AUROC of the transferred model on the target test set, with the gap to the "
            "in-domain ceiling in brackets.",
            "",
            transfer_table(gaps, regs, n_train, source_ds),
            "",
            "Paired contrasts of the transfer gap. 'probe - kNN' positive means the probe loses "
            "more of its in-domain ceiling than retrieval; 'fine-tuned - X' positive means the "
            "fine-tuned ResNet-50 loses more than frozen head X.",
            "",
            transfer_contrast(gaps, regs),
            "",
        ]
    out += ["## ID vs OOD side by side (as specified)", ""]
    for train_ds, ood_ds in DIRECTIONS:
        fig_name = f"gap_auroc_{train_ds}_to_{ood_ds}.png"
        gap_figure(tidy, train_ds, ood_ds, n_train, reports / "figures" / fig_name)
        out += [f"### Train on {train_ds.upper()}, held out {ood_ds.upper()}", ""]
        out += [f"![gap](figures/{fig_name})", ""]
        for metric in METRICS:
            out += [
                f"#### Macro {METRIC_LABEL[metric]}",
                "",
                macro_table(tidy, train_ds, metric, n_train),
                "",
            ]
        out += [
            "#### Head contrast: gap(probe) - gap(kNN), macro AUROC, paired bootstrap",
            "",
            "Positive means the linear probe degrades more than retrieval on the same resamples.",
            "",
            head_contrast(results, tidy, train_ds),
            "",
            "#### Per-label AUROC at 100% labels (ID / OOD)",
            "",
            label_table(m, train_ds),
            "",
        ]
    out += [
        "## ID vs OOD at 100% labels",
        "",
        "![id vs ood](figures/id_vs_ood_auroc_100pct.png)",
        "",
    ]
    dumbbell_figure(tidy, reports / "figures" / "id_vs_ood_auroc_100pct.png")
    latex_id_ood_table(tidy, tables / "id_ood.tex")
    (reports / "results.md").write_text("\n".join(out))
    n_fig = len(list((reports / "figures").glob("*.png")))
    print(f"wrote {reports / 'results.md'} and {n_fig} figures")


if __name__ == "__main__":
    main()
