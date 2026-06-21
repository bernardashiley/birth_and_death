# birth_and_death

Bayesian Developmental Exposure Burden Index for Childhood PM2.5 Assessment.

Linking early-life PM2.5 exposure (prenatal, age 1, age 4) to KABC cognitive
domains in a cohort of 400 children, using a Bayesian Weighted Quantile Sum
(WQS) burden index with confounder adjustment and a pre-specified effect
modifier.

## Repository layout
```
Bernard_Proj.xlsx            raw data (400 x 34)
Dictioary_outcome VAr.docx   data dictionary (KABC domains + variables)
scripts/split_data.py        reproducible train/val/test split
data/splits/                 train.csv (279) / val.csv (61) / test.csv (60)
reports/exploration_summary.md   findings + data-quality issues
```

## Environment
Python 3.12 (no R toolchain on the host). Install:
```
python -m pip install pandas openpyxl python-docx scikit-learn numpy
```

## Data split
`python scripts/split_data.py` produces a stratified (by `Child_sex`),
reproducible (`seed=42`) 70/15/15 split into `data/splits/`. Partitions are
disjoint by `ID_Identifier_base` and exhaustive.

> **Note on n=400:** a held-out test set is small for a Bayesian model with
> small expected effects. For the actual WQS fit, prefer Bayesian
> cross-validation (LOO / WAIC) over the held-out test set; the explicit split
> is provided here as requested and is useful for a final, untouched sanity
> check. **Fit all data cleaning/imputation on TRAIN only**, then apply to
> VAL/TEST to avoid leakage.

## Key findings (see `reports/exploration_summary.md`)
- All three PM2.5 windows present and weakly correlated → WQS weights identifiable.
- Only **4 of 5** CHC domains present (Gc / `Know_Gc` is missing from the data).
- `medlev` (education) is **43.5% missing** — main risk to the SES effect-modifier analysis.
- `smokecur` is constant ("No") → the planned smoking confounder is unavailable.
- Several impossible values (heights ~193–196 cm, `gageanc=99`) need cleaning.
