# Data Exploration Summary

**Dataset:** `Bernard_Proj.xlsx` — 400 children × 34 columns (one sheet).
**Goal context:** Bayesian Developmental Exposure Burden Index for childhood PM2.5
assessment, linking PM2.5 across life-stages to KABC cognitive domains.

## 1. What's actually in the file

### Exposure (PM2.5) — the three developmental windows are present
| Column | Window | mean | sd | min | max |
|---|---|---|---|---|---|
| `PM25_prenatal` | prenatal | 72.3 | 46.0 | 6.4 | 311 |
| `PM25_age1` | age 1 | 60.5 | 31.2 | 5.3 | 283 |
| `PM2.5_age4` | age 4 | 72.6 | 46.1 | 3.4 | 300 |
| `Early_life_PM25` | composite | 51.1 | 34.0 | 6.2 | 220 |

- The three windows the project needs (prenatal / age 1 / age 4) all exist.
- The three windows are **weakly correlated** (r ≈ 0.07–0.13), which is good —
  it means the WQS weights can actually be identified from the data.
- `Early_life_PM25` is **not** a clean mean of the three windows (corr 0.79 with
  prenatal, 0.58 with age1, 0.12 with age4). Its exact definition needs
  confirming from Nana Yaw before use; do not assume it is the WQS index.
- Values are high (50–70+ µg/m³) and look like **modelled/area estimates**, not
  personal monitors → supports the planned measurement-error / latent-exposure
  upgrade later.

### Outcome (KABC cognitive domains) — only 4 of the 5 CHC domains present
| Column | CHC domain | mean | sd |
|---|---|---|---|
| `seq_gsm` | Gsm — working memory | 69.4 | 9.7 |
| `Simul_Gv` | Gv — visual processing | 69.5 | 10.0 |
| `Learn_Glr` | Glr — long-term retrieval | 67.5 | 12.2 |
| `Plan_Gf` | Gf — fluid reasoning | 59.3 | 6.6 |
| `FCI_score` | Fluid–Crystallized composite | 54.6 | 6.4 |

- **Gc (Knowledge / `Know_Gc`) is in the data dictionary but NOT in the data.**
  The multivariate outcome is therefore **4 domains, not 5** unless Gc is
  supplied separately. Flag this with Nana Yaw — it changes the outcome vector.
- All outcomes are continuous standard scores → Normal likelihood is appropriate.
- These columns appear **duplicated** in the sheet (`seq_gsm.1`, `Simul_Gv.1`,
  `Learn_Glr.1`, `Plan_Gf.1` are byte-identical copies) — Excel artifact, ignore.

### Covariates (potential confounders / modifiers / mediators)
- **Confounder candidates:** `M_age` (14–46), `Child_sex` (202 M / 198 F),
  `gageanc`, `medlev` (education), `marital_status`, `ethnicity` (14 groups),
  `Occupation`, `bplace`, `b1placbir21` (delivery type), `numbaby`.
- **Mediator candidates (keep OUT of confounder set):** `birthwt_age1`,
  anthropometrics (`Ht_*`, `BMI_age1`, `Aual_age4`, `Aheadcap_age4`) that sit
  downstream of prenatal exposure.

## 2. Data-quality issues to resolve before modelling

| Issue | Detail | Action |
|---|---|---|
| `medlev` 43.5% missing | 174/400 blank — but this is the key SES/education confounder | Decide: model-based imputation (Bayesian, on TRAIN) or drop as confounder |
| `Occupation` 22.5% missing | 90/400 blank | Imputation or coarse recode |
| `smokecur` constant | every row = "No" — zero information | **Drop**; the planned smoking confounder is unavailable |
| Impossible `ht_age4` | 3 values ≈ 193–196 cm for 4-year-olds | Data-entry error (likely 93–96); flag/winsorize |
| `gageanc = 99` | 2 rows (5 rows > 45 weeks) | Treat 99 as missing code |
| `Aual_age4` = 65.5, `Aheadcap_age4` > 55 | single extreme outliers | Verify / winsorize |
| Three height-at-age-4 cols | `ht_age4` (0 missing), `Ht_age4` (20 missing), `ht_age4.1` (= `ht_age4`) | Keep one canonical column |
| Category whitespace | `' Married'`, `' Divorced'`, `' trader...'` have leading spaces | Strip before encoding |

## 3. Implications for the modelling plan
- n=400 supports: WQS burden index, main pollution effect, the multivariate
  (4-domain) outcome, and **one** pre-specified interaction (e.g. SES × burden).
- The heavy `medlev` missingness is the biggest threat to the
  environmental-justice (effect-modifier) analysis, since education/SES is the
  natural modifier. Resolve this first.
- All cleaning (imputation, outlier handling, category encoding) must be fit on
  **TRAIN only** and applied to VAL/TEST to avoid leakage.
