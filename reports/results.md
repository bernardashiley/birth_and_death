# Results — Bayesian WQS Burden Index

Full pipeline complete: **train → validate → test → diagnostics**. Fitted on
train (n=279), 4 chains × 1000 draws (4000 posterior samples). Numbers below come
from `data/model/evaluation.json` and the saved posterior; reproduce with
`wqs_model.py --fit` then `wqs_evaluate.py`.

> **Read this first.** The model is statistically sound and converged cleanly,
> but its **out-of-sample predictive skill is essentially nil** and the exposure
> data has a serious imputation artifact (38–45% of prenatal/age1 PM2.5 is one
> repeated value). Everything below should be read as *weak, suggestive
> associations that cannot be trusted until the exposure data is fixed* — not as
> findings.

## 1. Convergence diagnostics — all pass
| Diagnostic | Value | Verdict |
|---|---|---|
| max R̂ | 1.00 | converged |
| min ESS (bulk / tail) | 1222 / 1548 | ample |
| divergences | 0 | clean geometry |
| min BFMI | 0.78 | no energy pathology |
| PSIS-LOO Pareto-k | 99.6% ≤ 0.7 (max 0.71) | LOO reliable |

Sampling is trustworthy. Any weakness in the results is the **data and effect
sizes**, not the MCMC.

## 2. Out-of-sample performance — train ≈ val, near-zero skill
Expected log predictive density per child (higher = better):

| Split | n | elppd / child | RMSE raw (Gsm / Gv / Glr / Gf) |
|---|---|---|---|
| train | 279 | −5.20 | 9.6 / 9.9 / 11.9 / 6.2 |
| **val** | 61 | **−5.20** | 9.6 / 9.1 / 11.7 / 5.3 |
| **test** | 60 | **−5.39** | 9.2 / 10.0 / 12.1 / 8.3 |

- **No overfitting:** validation elppd equals train (−5.20); test is marginally
  worse (−5.39). The in-model SES imputation generalised to held-out children.
- **But near-zero predictive value:** RMSEs ≈ the raw outcome SDs (Gsm 9.7, Gv
  10.0, Glr 12.2, Gf 6.6) — i.e. the burden index + covariates predict barely
  better than guessing each child's mean. Gf on the test set (RMSE 8.3 > its SD
  6.6) even does worse than the mean, a small-n (60) reminder not to over-read
  any single domain. This is consistent with the tiny raw exposure–cognition
  correlations seen at exploration (|r| < 0.14).

## 3. Where the (weak) signal is
**Burden effect `b1` per domain** — effect of the weighted burden index on each
ability, in SD units (and raw KABC points), with posterior P(effect < 0):

| Domain | b1 (SD) | 94% HDI | raw pts | P(harmful) |
|---|---|---|---|---|
| Gsm (working memory) | +0.03 | [−0.11, +0.18] | +0.3 | 0.37 |
| Gv (visual) | +0.02 | [−0.13, +0.19] | +0.2 | 0.40 |
| Glr (retrieval) | −0.08 | [−0.22, +0.08] | −1.0 | 0.84 |
| **Gf (fluid reasoning)** | **−0.16** | **[−0.32, −0.01]** | **−1.0** | **0.97** |

- **Gf is the only domain whose 94% interval excludes zero** (P = 0.97 the burden
  is harmful), ≈ a 1-point drop across the burden range. Glr leans the same way
  (P = 0.84) but its interval crosses zero. Gsm and Gv show nothing.
- This *direction* — fluid reasoning most affected — is biologically plausible,
  but the effect is small and rests on the compromised exposure data, so treat it
  as a hypothesis to re-test, not a result.

**Window weights `w`** (the burden split): age4 0.53, age1 0.27, prenatal 0.20 —
**but the 94% HDIs span almost the entire simplex** (e.g. age4 [0.13, 0.85],
prenatal [0.00, 0.45]). The weights are **not identified**: the data cannot say
which window matters, exactly as predicted from the tie-spike artifact (prenatal
and age1 are ~40% a single imputed value). **Do not report a critical window.**

**Interaction `theta` (burden × SES):** all four domains lean negative (P(<0)
0.64–0.87) — i.e. a hint that burden bites harder at lower SES — but every
interval comfortably includes zero. **Inconclusive**, as anticipated for one
data-thin interaction at n=279. The environmental-justice question is
underpowered here, not answered.

**SES main effect `b_med`:** higher education leans toward better Glr (+0.20) and
Gf (+0.20), P>0 ≈ 0.82–0.83; intervals include zero.

**Residual domain correlations** (justifying the multivariate model): 0.22–0.49,
matching the raw data — the joint model correctly captured cross-domain
structure.

## 4. Bottom line
- **Methodologically:** the model does exactly what it should — converges
  cleanly, imputes SES honestly in-model, generalises without overfitting, and
  reports uncertainty faithfully. The machinery is sound and reusable.
- **Substantively:** the only association that rises above noise is **PM2.5
  burden → lower fluid reasoning (Gf), P ≈ 0.97**, and even that is small and
  sits on compromised exposure data. Window weights are unidentified; the SES
  interaction is inconclusive.
- **The binding constraint is the data, not the model.** The headline
  blocker is the exposure imputation (one repeated value for 38–45% of
  prenatal/age1 PM2.5). Until that is resolved, the burden index cannot do the
  job the title promises.

## 5. Recommended next steps (in priority order)
1. **Resolve the exposure tie-spikes with Nana Yaw** — the repeated value ≈ the
   median of the observed values (median-imputation of missing readings).
   Exposure is tagged per-household (good — the genuine values are reliable and
   need no measurement-error layer), but the ~40% median-filled values carry no
   information. Obtain the un-imputed PM2.5 or a measured-vs-filled flag per
   window. Nothing about the windows is interpretable until then.
2. **Confirm Gc** — recover the missing 5th CHC domain if it exists.
3. Re-fit once exposure is fixed; only then revisit the critical-window and
   environmental-justice questions.
4. Optionally add a sensitivity run with the point-imputed SES (`medlev_ord_imp`)
   to confirm the in-model imputation isn't driving anything.
