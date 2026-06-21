# Variable Triage

Every column assigned a causal role **before modelling**, using DAG logic
(confounder = causes both exposure and outcome → adjust; mediator = on the
exposure→outcome path → must NOT adjust; effect modifier = changes the strength
of the effect → interact). This is the single most defensible step against the
"34 columns, 400 children" researcher-degrees-of-freedom problem.

**Coefficient budget:** train n≈279 → keep estimated coefficients well under ~28
(≈1 per 10). The core set below spends ~10–12, leaving room for the WQS weights
and one interaction.

## Roles at a glance

| Role | Columns |
|---|---|
| **Outcome — primary (multivariate)** | `seq_gsm` (Gsm), `Simul_Gv` (Gv), `Learn_Glr` (Glr), `Plan_Gf` (Gf) |
| **Outcome — secondary (composite)** | `FCI_score` |
| **Exposure — WQS components** | `PM25_prenatal`, `PM25_age1`, `PM2.5_age4` |
| **Confounders — adjust** | `M_age`, `Child_sex`, `ethnicity`*, `marital_status`*, `medlev`* (SES) |
| **Effect modifier — pre-specified (1)** | `medlev` (education/SES) — fallback: constructed SES index |
| **Mediators — exclude from adjustment** | `birthwt_age1`, `BMI_age1`, `gageanc`, `b1placbir21`, `Ht_Age1`, `ht_age4`, `Aual_age4`, `Aheadcap_age4` |
| **Exclude entirely** | `ID_Identifier_base`, `smokecur`, `numbaby`, `Occupation`, `Ht_age7`, `Ht_age4`, `Early_life_PM25`, `*.1` duplicates |

\* requires recoding/imputation — see notes.

## Full table with reasoning

| # | Column | Role | Reasoning |
|---|---|---|---|
| 0 | `ID_Identifier_base` | exclude (keep for linkage) | identifier only |
| 1 | `M_age` | **confounder** | maternal age → both residence/exposure and cognition; complete |
| 2 | `Child_sex` | **confounder** (+ 2ndary modifier) | balanced 202/198; standard modifier in this literature → sensitivity-only interaction |
| 3 | `gageanc` | mediator/exclude | gestational-age-at-ANC; ambiguous, downstream-ish; has `99` error codes |
| 4 | `numbaby` | exclude | only 2 twins → near-zero variance; consider dropping the 2 twin rows |
| 5 | `bplace` | confounder (optional) | care-access/SES proxy; complete. Use only if SES signal is weak |
| 6 | `b1placbir21` | mediator/exclude | delivery type (CS) likely downstream of pregnancy complications |
| 7 | `smokecur` | **exclude** | constant ("No") — zero information; planned smoking confounder unavailable |
| 8 | `medlev` | **confounder + PRIMARY modifier** | education = the env-justice variable. **43.5% missing → impute on TRAIN** |
| 9 | `marital_status` | confounder | social-support/SES proxy. Collapse → {Married, Cohab, Other}; strip whitespace |
| 10 | `ethnicity` | confounder | proxy for location (→ exposure) & culture. **Collapse 14→~5**; messy overlapping labels; 3 missing |
| 11 | `Occupation` | exclude (optional sensitivity) | 22.5% missing AND redundant with other SES proxies |
| 12 | `FCI_score` | **outcome — composite (secondary)** | correlates 0.62–0.81 with the 4 domains → it IS their composite; keep OUT of the joint vector to avoid double-counting; report separately |
| 13 | `ht_age4` | mediator/exclude | postnatal anthropometric, downstream of exposure; **3 impossible values ~193–196 cm** |
| 14 | `seq_gsm` | **outcome (Gsm)** | continuous standard score |
| 15 | `Simul_Gv` | **outcome (Gv)** | continuous standard score |
| 16 | `Learn_Glr` | **outcome (Glr)** | continuous standard score |
| 17 | `Plan_Gf` | **outcome (Gf)** | continuous standard score |
| 18 | `Ht_Age1` | mediator/exclude | postnatal growth, downstream |
| 19 | `Ht_age4` | exclude | duplicate height-at-4 (20 missing); keep canonical `ht_age4` only |
| 20 | `Ht_age7` | **exclude** | measured at age 7, AFTER the age-4 outcome → temporally invalid as predictor |
| 21 | `BMI_age1` | mediator/exclude | postnatal nutrition status, downstream |
| 22 | `birthwt_age1` | **mediator — exclude** | classic PM2.5 → low birthweight → cognition path; adjusting erases real effect |
| 23 | `PM25_prenatal` | **exposure (window 1)** | WQS quantile component |
| 24 | `PM25_age1` | **exposure (window 2)** | WQS quantile component |
| 25 | `Early_life_PM25` | exclude | not a clean function of the 3 windows; definition unconfirmed; redundant |
| 26 | `ht_age4.1` | exclude | exact duplicate of `ht_age4` (Excel artifact) |
| 27 | `Aual_age4` | mediator/exclude | upper-arm length age 4, downstream; 1 extreme outlier (65.5) |
| 28 | `Aheadcap_age4` | mediator/exclude | head circumference age 4 — downstream; candidate **mediator** for a separate mediation analysis |
| 29 | `PM2.5_age4` | **exposure (window 3)** | WQS quantile component |
| 30–33 | `seq_gsm.1`, `Simul_Gv.1`, `Learn_Glr.1`, `Plan_Gf.1` | **exclude** | byte-identical duplicate columns (Excel artifact) |

## Resulting modelling set
- **Outcome vector (multivariate Normal):** `[seq_gsm, Simul_Gv, Learn_Glr, Plan_Gf]` — 4 domains (Gc absent). `FCI_score` reported as a secondary composite.
- **WQS index components:** quantiled `PM25_prenatal`, `PM25_age1`, `PM2.5_age4` with a shared simplex weight `(w1,w2,w3)`, domain-specific effects `β1^d`.
- **Adjustment set:** `M_age`, `Child_sex`, `ethnicity`(collapsed), `marital_status`(collapsed), `medlev`(imputed SES).
- **Interaction:** `WQS × medlev` (education/SES) — the environmental-justice term; one only, pre-specified.

## Open decisions for Nana Yaw
1. **Gc** — is the Knowledge/`Know_Gc` domain available separately? If yes, the outcome becomes the full 5-vector.
2. **`medlev` missingness (43.5%)** — confirm Bayesian imputation is acceptable, or fall back to `bplace`/a constructed SES index as the modifier.
3. **`gageanc` / `b1placbir21`** — confirm these are downstream (mediator) and not intended as confounders.
4. **`Early_life_PM25`** — confirm its definition; currently excluded as redundant.
