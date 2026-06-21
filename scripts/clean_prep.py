"""
Train-only cleaning + WQS data-prep.

Produces a model-ready analysis frame for each split. Every transform is FIT ON
TRAIN ONLY and applied unchanged to val/test, so no information leaks from the
held-out sets:

  * WQS quantile bands   -- cut-points from train quartiles
  * M_age standardisation -- mean/sd from train
  * category levels       -- ethnicity/marital/sex levels learned on train;
                             unseen levels in val/test fall back to reference/Other
  * medlev imputation     -- predictive model fit on train complete-cases

Outputs (data/prepped/):
  train.csv / val.csv / test.csv   model-ready frames
  transform_params.json            all fitted parameters (auditable)
  medlev_imputer.joblib            the fitted imputation model

Run AFTER scripts/split_data.py.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "config"))
import variables as v  # noqa: E402

SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "data" / "prepped"
OUT.mkdir(parents=True, exist_ok=True)

N_QUANTILES = 4          # quartile bands -> {0,1,2,3}; standard WQS choice
ETHNIC_MIN = 25          # train-count threshold below which a group -> 'Other'

# Domain-knowledge fold of duplicated/sub-group ethnicity labels (not data-driven).
ETHNIC_FOLD = {
    "Basare": "Konkomba/Basare",
    "Frafra/Kusasi": "Dagarti/Grushi/Frafra/Kusasi",
    "Chokosi": "Bimoba/Chokosi",
}
MEDLEV_ORDER = {"Primary": 1, "Middle/JHS": 2, "Technical/Commercial/SHS": 3}


def _strip(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _onehot(s: pd.Series, levels, prefix, reference) -> pd.DataFrame:
    """Drop-first one-hot using train-defined levels; unseen -> all zeros (=ref)."""
    cols = {}
    for lev in levels:
        if lev == reference:
            continue
        cols[f"{prefix}_{lev}"] = (s == lev).astype(int)
    return pd.DataFrame(cols, index=s.index)


def fit(train: pd.DataFrame) -> dict:
    p = {}

    # --- WQS quantile banding via train empirical CDF (rank-based) ---
    # Value-based cut-points fail here: 38-45% of prenatal/age1 values are a
    # single repeated number (likely upstream mean/median imputation). Rank-based
    # midpoint-CDF banding assigns tied masses deterministically and is the
    # standard WQS quantile-scoring approach.
    p["n_train"] = int(len(train))
    p["exposure_sorted"] = {col: np.sort(train[col].values).tolist() for col in v.EXPOSURES}
    p["tie_spikes"] = {}  # data-quality flag
    for col in v.EXPOSURES:
        vc = train[col].round(2).value_counts()
        top_val, top_n = vc.index[0], int(vc.iloc[0])
        p["tie_spikes"][col] = {"value": float(top_val), "n": top_n,
                                "frac": round(top_n / len(train), 3)}

    # --- M_age standardisation ---
    p["M_age_mean"] = float(train["M_age"].mean())
    p["M_age_sd"] = float(train["M_age"].std(ddof=0))

    # --- Child_sex levels ---
    sex = _strip(train["Child_sex"])
    p["sex_levels"] = sorted(sex.dropna().unique().tolist())
    p["sex_reference"] = "Male"  # reference

    # --- marital_status: collapse then learn levels ---
    mar = _strip(train["marital_status"]).map(
        lambda x: v.COLLAPSE["marital_status"].get(x, "Other")
    )
    p["marital_levels"] = sorted(mar.dropna().unique().tolist())
    p["marital_reference"] = "Married"

    # --- ethnicity: fold dup labels, bucket rare (train freq) ---
    eth = _strip(train["ethnicity"]).map(lambda x: ETHNIC_FOLD.get(x, x))
    counts = eth.value_counts()
    keep = counts[counts >= ETHNIC_MIN].index.tolist()
    p["ethnicity_keep"] = keep
    p["ethnicity_reference"] = counts.idxmax()  # largest group as reference

    # --- medlev imputation model (fit on train complete-cases) ---
    med = _strip(train["medlev"]).map(MEDLEV_ORDER)
    feat = _build_impute_features(train, p)
    obs = med.notna()
    if obs.sum() >= 30 and med[obs].nunique() > 1:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feat[obs.values], med[obs].astype(int))
        joblib.dump(clf, OUT / "medlev_imputer.joblib")
        p["medlev_imputer"] = "medlev_imputer.joblib"
    else:
        p["medlev_imputer"] = None
    p["medlev_mode"] = int(med.mode().iloc[0])  # fallback
    return p


def _build_impute_features(df: pd.DataFrame, p: dict) -> np.ndarray:
    """Predictors used to impute medlev: M_age(z), sex, marital, ethnicity."""
    parts = []
    parts.append(((df["M_age"] - p["M_age_mean"]) / p["M_age_sd"]).to_frame("M_age_z"))
    sex = _strip(df["Child_sex"])
    parts.append(_onehot(sex, p["sex_levels"], "sex", p["sex_reference"]))
    mar = _strip(df["marital_status"]).map(
        lambda x: v.COLLAPSE["marital_status"].get(x, "Other")
    )
    parts.append(_onehot(mar, p["marital_levels"], "marital", p["marital_reference"]))
    eth = _strip(df["ethnicity"]).map(lambda x: ETHNIC_FOLD.get(x, x))
    eth = eth.where(eth.isin(p["ethnicity_keep"]), "Other")
    levels = p["ethnicity_keep"] + (["Other"] if "Other" not in p["ethnicity_keep"] else [])
    parts.append(_onehot(eth, levels, "eth", p["ethnicity_reference"]))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X.values


def transform(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out[v.ID] = df[v.ID].values

    # Outcomes (native scale)
    for c in v.OUTCOMES + [v.OUTCOME_COMPOSITE]:
        out[c] = df[c].values

    # WQS bands (rank-based, train CDF) + raw exposures (kept for measurement-error work)
    band_map = {"PM25_prenatal": "q_prenatal", "PM25_age1": "q_age1", "PM2.5_age4": "q_age4"}
    n = p["n_train"]
    for col in v.EXPOSURES:
        srt = np.asarray(p["exposure_sorted"][col])
        x = df[col].values
        lo = np.searchsorted(srt, x, side="left")    # count of train < x
        hi = np.searchsorted(srt, x, side="right")   # count of train <= x
        cdf = (lo + hi) / 2.0 / n                     # midpoint rank -> [0,1)
        band = np.clip((cdf * N_QUANTILES).astype(int), 0, N_QUANTILES - 1)
        out[band_map[col]] = band
        out[f"raw_{band_map[col]}"] = x

    # M_age standardised
    out["M_age_z"] = (df["M_age"].values - p["M_age_mean"]) / p["M_age_sd"]

    # Child_sex
    sex = _strip(df["Child_sex"])
    out = out.join(_onehot(sex, p["sex_levels"], "sex", p["sex_reference"]))

    # marital
    mar = _strip(df["marital_status"]).map(
        lambda x: v.COLLAPSE["marital_status"].get(x, "Other")
    )
    out = out.join(_onehot(mar, p["marital_levels"], "marital", p["marital_reference"]))

    # ethnicity
    eth = _strip(df["ethnicity"]).map(lambda x: ETHNIC_FOLD.get(x, x))
    eth = eth.where(eth.isin(p["ethnicity_keep"]), "Other")
    levels = p["ethnicity_keep"] + (["Other"] if "Other" not in p["ethnicity_keep"] else [])
    out = out.join(_onehot(eth, levels, "eth", p["ethnicity_reference"]))

    # medlev: ordinal + missing flag + (preferred) NaN-preserving + imputed convenience col
    med = _strip(df["medlev"]).map(MEDLEV_ORDER)
    out["medlev_missing"] = med.isna().astype(int).values
    out["medlev_ord"] = med.values  # NaN preserved -> for in-model Bayesian imputation
    imp = med.copy()
    miss = imp.isna()
    if miss.any():
        if p.get("medlev_imputer"):
            clf = joblib.load(OUT / p["medlev_imputer"])
            feat = _build_impute_features(df, p)
            pred = clf.predict(feat[miss.values])
            imp.loc[miss] = pred
        else:
            imp.loc[miss] = p["medlev_mode"]
    out["medlev_ord_imp"] = imp.astype(int).values  # complete; for non-Bayesian use
    return out


def main():
    train = pd.read_csv(SPLITS / "train.csv")
    val = pd.read_csv(SPLITS / "val.csv")
    test = pd.read_csv(SPLITS / "test.csv")

    params = fit(train)
    with open(OUT / "transform_params.json", "w") as f:
        json.dump(params, f, indent=2)

    for name, df in (("train", train), ("val", val), ("test", test)):
        prepped = transform(df, params)
        prepped.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:5s}: {prepped.shape} -> data/prepped/{name}.csv")

    # ---- data-quality flag: repeated-value spikes in exposure ----
    print("\n[!] Exposure tie-spikes (likely upstream mean/median imputation):")
    for col, s in params["tie_spikes"].items():
        flag = "  <-- DOMINANT" if s["frac"] >= 0.20 else ""
        print(f"    {col}: value {s['value']} repeated {s['n']}x ({s['frac']:.0%}){flag}")

    # ---- sanity / no-leakage checks ----
    tr = transform(train, params)
    print("\nWQS band distribution (train) -- lumpy bands reflect the tie-spikes above:")
    for c in ["q_prenatal", "q_age1", "q_age4"]:
        print(f"  {c}: {tr[c].value_counts(normalize=True).sort_index().round(2).to_dict()}")
    print("\nmedlev after prep -- missing remaining in imputed col:",
          int(transform(val, params)["medlev_ord_imp"].isna().sum()))
    print("Design columns:", [c for c in tr.columns if c not in (
        [v.ID, v.OUTCOME_COMPOSITE] + v.OUTCOMES)])


if __name__ == "__main__":
    main()
