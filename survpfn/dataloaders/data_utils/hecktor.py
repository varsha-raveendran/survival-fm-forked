"""
survpfn.dataloaders.hecktor — HECKTOR 2026 head-and-neck cancer dataset loader.

Source: /Users/koala/Code/datasets/hecktor/raw/HECKTOR_2026_training_data.csv
(same raw file used by the sibling `hnc-gnn-surv` project's Hydra pipeline).

Columns
-------
Age                    : numeric
Gender                 : binary (0/1)
Tobacco Consumption    : binary (0/1)
Alcohol Consumption    : binary (0/1)
Performance Status     : ordinal (0..4)
Treatment              : binary (0/1)
HPV Status             : binary (0/1)
CenterID               : categorical (one-hot)
T-stage                : ordinal (T0..T4 -> 0..4)
N-stage                : ordinal (N0..N3 -> 0..3)
RFS                    : time column
Relapse                : event (1=relapse, 0=censored)

Missingness
-----------
Several feature columns have substantial missingness among the 727
outcome-labelled patients (Tobacco ~35%, Alcohol ~36%, Performance Status
~39%, HPV Status ~24%). Unlike most other loaders in this module (which
just drop incomplete rows), a blanket dropna() here would shrink the
cohort from 727 to 429 rows. To keep the full labelled cohort and stay
roughly comparable with the median/mode imputation the sibling
`hnc-gnn-surv` project uses for its Cox/RSF baselines on this same data,
missing values are imputed with the column median (numeric/ordinal) or
mode (binary) computed over the whole labelled cohort, rather than
per-CV-fold. Simpler than `hnc-gnn-surv`'s leak-free per-fold imputation,
but consistent with how this benchmark harness loads a dataset once
(scaling/splitting happens afterwards, per fold, in scripts/benchmark.py).

Returns
-------
(df, "time", "event")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_PATH = "/Users/koala/Code/datasets/hecktor/raw/HECKTOR_2026_training_data.csv"

_RAW_TO_CANONICAL = {
    "CenterID": "center_id",
    "Age": "age",
    "Gender": "gender",
    "Tobacco Consumption": "tobacco",
    "Alcohol Consumption": "alcohol",
    "Performance Status": "performance_status",
    "Treatment": "treatment",
    "HPV Status": "hpv_status",
    "Relapse": "event",
    "RFS": "time",
    "T-stage": "t_stage",
    "N-stage": "n_stage",
}

_BINARY_COLS = ["gender", "tobacco", "alcohol", "hpv_status", "treatment"]
_ORDINAL_NUMERIC_COLS = ["age", "performance_status", "t_stage", "n_stage"]
_T_STAGE_MAP = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}
_N_STAGE_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}

_TIME_COL = "time"
_EVENT_COL = "event"


def load_hecktor(
    filepath: str | Path | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load and preprocess the HECKTOR 2026 head-and-neck cancer dataset.

    Parameters
    ----------
    filepath:
        Path to the raw CSV file. Defaults to the shared location used by
        the sibling `hnc-gnn-surv` project.

    Returns
    -------
    (df, "time", "event")
        df has numeric + ordinal + one-hot encoded features, with
        `time` (float, RFS in days) and `event` (0/1, relapse) appended.
        No StandardScaler is applied — callers must scale inside CV folds.
    """
    if filepath is None:
        filepath = _DEFAULT_PATH
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"HECKTOR CSV not found at '{filepath}'. "
            "Expected the HECKTOR 2026 training data used by the sibling "
            "hnc-gnn-surv project."
        )

    raw = pd.read_csv(filepath)
    raw = raw.drop(columns=["PatientID"])
    raw = raw.rename(columns=_RAW_TO_CANONICAL)

    # ── Drop rows with missing outcome ──────────────────────────────────────
    raw[_EVENT_COL] = pd.to_numeric(raw[_EVENT_COL], errors="coerce")
    raw[_TIME_COL] = pd.to_numeric(raw[_TIME_COL], errors="coerce")
    raw = raw.dropna(subset=[_EVENT_COL, _TIME_COL]).reset_index(drop=True)

    # ── Ordinal string -> int encoding ──────────────────────────────────────
    raw["t_stage"] = raw["t_stage"].map(_T_STAGE_MAP)
    raw["n_stage"] = raw["n_stage"].map(_N_STAGE_MAP)

    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in _ORDINAL_NUMERIC_COLS + _BINARY_COLS + ["center_id"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ── Impute missing feature values ───────────────────────────────────────
    for col in _ORDINAL_NUMERIC_COLS:
        raw[col] = raw[col].fillna(raw[col].median())
    for col in _BINARY_COLS:
        raw[col] = raw[col].fillna(raw[col].mode().iloc[0])

    # ── One-hot encode center_id ─────────────────────────────────────────────
    center_dummies = pd.get_dummies(
        raw["center_id"].astype(int).astype(str), prefix="center_id", drop_first=True
    ).astype(float)
    raw = pd.concat([raw.drop(columns=["center_id"]), center_dummies], axis=1)

    # ── Final dtype pass ─────────────────────────────────────────────────────
    feat_cols = [c for c in raw.columns if c not in {_TIME_COL, _EVENT_COL}]
    raw[feat_cols] = raw[feat_cols].astype(float)
    raw[_TIME_COL] = raw[_TIME_COL].astype(float)
    raw[_EVENT_COL] = raw[_EVENT_COL].astype(float)

    return raw, _TIME_COL, _EVENT_COL
