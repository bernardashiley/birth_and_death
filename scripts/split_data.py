"""
Split the KABC / PM2.5 cohort into train / validation / test sets.

Why Python: no R toolchain is installed on this machine; Python 3.12 with
pandas + scikit-learn is available and is the efficient path for both the
exploration and a reproducible, stratified split.

Design choices
--------------
* Raw values are preserved. No imputation, no outlier removal here -- the split
  must not leak cleaning decisions across sets. Cleaning belongs downstream,
  fitted on TRAIN only and applied to VAL/TEST.
* 70 / 15 / 15 split (280 / 60 / 60). With n=400 this is small; see README for
  the recommendation to prefer Bayesian cross-validation (LOO/WAIC) over a
  held-out test set for the actual WQS model. The split is provided as asked
  and is reproducible.
* Stratified by Child_sex so each set keeps the ~50/50 sex balance. Sex is the
  most common effect modifier in this literature, so balance matters.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Bernard_Proj.xlsx"
OUT = ROOT / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of the full dataset

df = pd.read_excel(SRC)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

strat = df["Child_sex"]

# First peel off TEST, then split the remainder into TRAIN / VAL.
train_val, test = train_test_split(
    df, test_size=TEST_FRAC, random_state=SEED, stratify=strat
)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)  # val fraction of the train_val block
train, val = train_test_split(
    train_val,
    test_size=val_rel,
    random_state=SEED,
    stratify=train_val["Child_sex"],
)

for name, part in (("train", train), ("val", val), ("test", test)):
    path = OUT / f"{name}.csv"
    part.to_csv(path, index=False)
    sex = part["Child_sex"].value_counts(normalize=True).round(3).to_dict()
    print(f"{name:5s}: n={len(part):3d}  sex_balance={sex}  -> {path.relative_to(ROOT)}")

assert len(train) + len(val) + len(test) == len(df)
# No overlap between sets.
ids = "ID_Identifier_base"
assert set(train[ids]).isdisjoint(test[ids])
assert set(train[ids]).isdisjoint(val[ids])
assert set(val[ids]).isdisjoint(test[ids])
print("OK: partitions are disjoint and exhaustive.")
