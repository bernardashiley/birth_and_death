# Cleaning & WQS Data-Prep

`scripts/clean_prep.py` turns the three raw splits into model-ready frames in
`data/prepped/`. **Every transform is fit on TRAIN only** and applied unchanged
to val/test (no leakage). Fitted parameters are saved to
`data/prepped/transform_params.json` and the imputation model to
`data/prepped/medlev_imputer.joblib`.

## ⚠️ Headline data-quality finding: exposure tie-spikes
A single value dominates each early exposure window — and that value is
**almost exactly the median of the observed (non-modal) values**, the
fingerprint of **median-imputation of missing PM2.5**:

| Window | Repeated value | Share | Median of the rest |
|---|---|---|---|
| `PM25_prenatal` | 61.30 | **38%** | 61.15 |
| `PM25_age1` | 54.80 | **43%** | 54.92 |
| `PM2.5_age4` | 64.13 | 18% | — |

Note (2026-06): the data owner reports PM2.5 was tagged **per household**
(not an area average). That is good for the *genuine* readings — they have
real between-child variation and low measurement error, so the optional
measurement-error model layer is **not** needed for them. But per-household
tagging does **not** explain a value repeating to 2 d.p. for ~40% of
households; the median-match above shows those are filled-in missings.
Overlap is partial (91 children share the fill in two windows, 11 in all
three) → per-window filling. Awaiting confirmation + original values /
a measured-vs-filled flag.

Implications:
- A 38–45% point-mass **cannot be split into balanced quantile bands** — the
  spike lands in one band, so prenatal/age1 effectively have ~3 usable levels.
- WQS weights for prenatal and age1 will be **weakly identified** — nearly half
  those children share one exposure value, so they carry little information
  about the exposure–cognition slope.
- **Action for Nana Yaw:** confirm whether these repeats are imputed. If so,
  either obtain the un-imputed values, or model exposure measurement error /
  missingness explicitly (the Bayesian latent-exposure upgrade), rather than
  quantiling an imputed constant.

## Transforms applied
| Step | Method (fit on train) | Output columns |
|---|---|---|
| WQS banding | rank-based **midpoint empirical-CDF** into quartiles {0,1,2,3} — robust to the tie-spikes; bands 0 & 3 are clean 25% | `q_prenatal`, `q_age1`, `q_age4` |
| Raw exposure kept | unchanged (for later measurement-error modelling) | `raw_q_*` |
| `M_age` | standardised (train mean/sd) | `M_age_z` |
| `Child_sex` | one-hot, drop-first (ref = Male) | `sex_Female` |
| `marital_status` | strip + collapse → {Married(ref), Cohab, Other}; one-hot | `marital_Cohab`, `marital_Other` |
| `ethnicity` | fold duplicate labels, bucket train-freq <25 → Other, one-hot (ref = largest group) | `eth_*` (5 dummies) |
| `medlev` | ordinal {Primary=1, Middle/JHS=2, Technical/SHS=3}; **NaN preserved** + missing flag + predictive imputation | `medlev_ord`, `medlev_missing`, `medlev_ord_imp` |

## medlev (education) — the 43.5%-missing modifier
Two columns are provided so the modelling step can choose:
- **`medlev_ord`** — ordinal with **NaN preserved**. *Preferred*: let
  Stan/brms impute it **inside** the Bayesian model (joint imputation propagates
  the uncertainty honestly — exactly the Bayesian advantage).
- **`medlev_ord_imp`** — complete, point-imputed by a multinomial logistic model
  (fit on train complete-cases from M_age, sex, marital, ethnicity). Convenience
  only. **Caveat:** it never predicts the rare class 3 (Technical/SHS, n=15), so
  it compresses the top category — another reason to prefer in-model imputation.
- `medlev_missing` — carry as a covariate/sensitivity flag either way.

## Output frame (24 cols)
`ID`, 4 outcomes + `FCI_score`, 3 WQS bands + 3 raw exposures, `M_age_z`,
`sex_Female`, 2 marital dummies, 5 ethnicity dummies, 3 medlev columns.

## Reproduce
```
python scripts/split_data.py     # -> data/splits/
python scripts/clean_prep.py      # -> data/prepped/
```

## Next step
Feed `data/prepped/train.csv` to the Bayesian WQS model: simplex weights
`(w1,w2,w3)` over `q_prenatal/q_age1/q_age4`, multivariate-Normal over the 4
domains with shared weights + domain-specific effects, adjustment set
{`M_age_z`, `sex_Female`, marital, ethnicity, `medlev_ord`}, and the
`WQS × medlev_ord` interaction. Evaluate with LOO/WAIC rather than the held-out
test set given n.
