"""Shared six-label space and per-dataset mappings. Rationale and ambiguities: LABELS.md."""

import pandas as pd

LABELS = ("atelectasis", "cardiomegaly", "consolidation", "effusion", "pneumothorax", "no_finding")
PATHOLOGIES = LABELS[:-1]

# source label name -> shared label (insertion order == LABELS order)
NIH_SOURCE = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Effusion": "effusion",
    "Pneumothorax": "pneumothorax",
    "No Finding": "no_finding",
}
VINDR_SOURCE = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Pleural effusion": "effusion",
    "Pneumothorax": "pneumothorax",
    "No finding": "no_finding",
}


def nih_labels(finding_labels: pd.Series) -> pd.DataFrame:
    """NIH 'Finding Labels' ('Cardiomegaly|Effusion', 'No Finding', ...) -> binary columns."""
    sets = finding_labels.str.split("|")
    cols = {ours: sets.map(lambda s, src=src: int(src in s)) for src, ours in NIH_SOURCE.items()}
    return pd.DataFrame(cols, index=finding_labels.index).astype("int8")


def vindr_labels(image_labels: pd.DataFrame) -> pd.DataFrame:
    """VinDr image_labels_{train,test}.csv (one row per image x radiologist) -> one row per
    image_id, majority vote per label (>50% of that image's raters). A positive pathology
    majority outranks a 'No finding' majority."""
    missing = [c for c in VINDR_SOURCE if c not in image_labels.columns]
    if missing:
        raise KeyError(f"VinDr label columns not found: {missing}")
    votes = image_labels.groupby("image_id")[list(VINDR_SOURCE)].mean()
    out = (votes > 0.5).astype("int8").rename(columns=VINDR_SOURCE)
    out.loc[out[list(PATHOLOGIES)].any(axis=1), "no_finding"] = 0
    return out.reset_index()


# IU (Open-i) manual MeSH major terms, lower-cased, modifiers after '/' dropped
IU_SOURCE = {
    "pulmonary atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "pleural effusion": "effusion",
    "pneumothorax": "pneumothorax",
    "normal": "no_finding",
}
# full-string prefixes (term + modifier) that also count as positives
IU_PREFIX = {
    "cardiac shadow/enlarged": "cardiomegaly",
    "hydropneumothorax": "pneumothorax",
    "hemopneumothorax": "pneumothorax",
}
IU_UNINDEXED = "no indexing"  # reports never manually coded: labels unknown, dropped by the loader


def iu_labels(mesh: pd.Series) -> pd.DataFrame:
    """IU 'MeSH' ('Cardiomegaly/mild;Pleural Effusion/left', 'normal', ...) -> binary columns."""
    terms = mesh.fillna("").str.lower().str.split(";").map(lambda ts: [t.strip() for t in ts])

    def has(ts, ours):
        heads = {t.split("/")[0] for t in ts}
        by_head = any(o == ours and src in heads for src, o in IU_SOURCE.items())
        by_prefix = any(
            o == ours and any(t.startswith(pre) for t in ts) for pre, o in IU_PREFIX.items()
        )
        return int(by_head or by_prefix)

    cols = {ours: terms.map(lambda ts, ours=ours: has(ts, ours)) for ours in LABELS}
    return pd.DataFrame(cols, index=mesh.index).astype("int8")
