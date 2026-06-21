# Bayesian WQS Model (PyMC)

`scripts/wqs_model.py` — the headline burden-index model. Consumes
`data/prepped/train.csv`.

## Specification
Shared simplex weights, domain-specific effects, multivariate outcome.

```
BI_i = w1·q_prenatal_i + w2·q_age1_i + w3·q_age4_i        (w) ~ Dirichlet(1,1,1)

mu_{i,d} = b0_d
         + b1_d · BI_i                 # burden effect for domain d
         + theta_d · BI_i · medlev_c_i # environmental-justice interaction
         + (X·gamma)_{i,d}             # confounder adjustment

Y_{i,·} ~ MvNormal(mu_{i,·}, Sigma)    # Sigma via LKJ correlation × HalfNormal sds
```

| Parameter | Meaning | Prior |
|---|---|---|
| `w` (3) | window weights, sum to 1 | Dirichlet(1,1,1) — uniform over the simplex |
| `b1` (4) | burden effect per domain | Normal(0, 1) — centred at 0, must earn its effect |
| `theta` (4) | burden × SES interaction | Normal(0, 0.5) — **tighter**; one pre-specified, data-thin |
| `gamma` (P×4) | confounder coefficients | Normal(0, 1) |
| `chol`/`corr`/`sds` | 4×4 residual covariance | LKJCholeskyCov(eta=2), sds HalfNormal(1) |

- **Outcomes standardised** (train mean/sd) → effects read in SD units; priors stay generic. Back-transform stats saved with the run.
- **Adjustment set:** `M_age_z`, `sex_Female`, marital (2), ethnicity (5), and the centred `medlev` main effect.
- **Interaction:** `BI × medlev_c` (centred), so `b1` reads at mean SES and `theta` is the modification.

## Decisions a reviewer will ask about
- **Shared weights, domain-specific b1** — one universal index, but lets pollution hit some abilities harder than others.
- **MvNormal not 4 separate regressions** — borrows strength across the correlated domains (r = 0.25–0.51); more efficient at n=279.
- **b1 centred at 0, not directional** — classic WQS forces all weights same-sign and a directional effect; the Bayesian version stays unconstrained and lets the data speak. Switch `b1` to HalfNormal (or negative-mean) if a directional constraint is wanted.

## Known limitations (from the data, not the model)
1. **Gc absent** → 4-domain outcome, not 5.
2. **Exposure tie-spikes** → `w1` (prenatal) and `w2` (age1) are weakly identified (38–45% of those values are one repeated, likely imputed, number). Interpret window weights cautiously; the headline total-burden effect is more robust than the split.
3. **medlev** → scaffold uses the point-imputed column. **TODO (preferred):** impute `medlev_ord` (NaN-preserving) inside the model so SES uncertainty propagates into `theta` — the proper Bayesian handling of the 43.5% missingness.

## Evaluation
Use **LOO / WAIC** (`arviz.loo`) for model comparison, not the held-out test
set, given n. Posterior-predictive checks per domain. Report `w` with credible
intervals (the burden split), `b1` per domain (which abilities), and `theta`
(the justice signal) — all with full posteriors, not point estimates.

## Run
```
python scripts/wqs_model.py --smoke   # compile + prior-predictive + tiny sample
python scripts/wqs_model.py --fit      # full NUTS -> data/model/wqs_idata.nc
```
> Verified: `--smoke` compiles, prior-samples and NUTS-samples cleanly on this
> host (PyMC 6.0.1). No C compiler (`g++`) is installed, so PyTensor uses its
> NumPy backend — at this model size that is fine (the smoke sample took ~2s).
> For large production runs, installing `m2w64-toolchain` (conda) or using WSL
> would speed sampling further.
