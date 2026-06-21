"""
Bayesian Weighted Quantile Sum (WQS) burden-index model — PyMC scaffold.

Model (one shared burden index, domain-specific effects)
--------------------------------------------------------
Burden index for child i (shared across the 4 CHC domains):
    BI_i = w1*q_prenatal_i + w2*q_age1_i + w3*q_age4_i,   (w1,w2,w3) ~ Dirichlet(1,1,1)

Multivariate outcome (4 domains d = Gsm, Gv, Glr, Gf), standardised:
    mu_{i,d} = b0_d + b1_d * BI_i                      # burden effect (domain-specific)
             + theta_d * BI_i * medlev_c_i             # env-justice interaction
             + (X gamma)_{i,d}                         # confounder adjustment
    Y_{i,:} ~ MvNormal(mu_{i,:}, Sigma)                # Sigma = LKJ correlation x sds

Why these choices
-----------------
* Shared weights w, domain-specific b1_d  -> one universal index, but tells you
  which abilities pollution touches hardest (the useful question).
* MvNormal over the 4 domains borrows strength across the correlated outcomes
  (observed r = 0.25-0.51), tightening every estimate -- efficient at n=279.
* b1 centred at 0 -> the index must earn its effect from the data.
* theta (interaction) gets a TIGHTER prior: interactions are data-expensive and
  we pre-specified exactly one (medlev/SES). The prior keeps it stable when thin.

Caveats baked into the data (see reports/data_prep.md)
* Gc domain absent -> outcome is a 4-vector, not 5.
* 38-45% of prenatal/age1 exposure is one repeated (likely imputed) value, so
  w1/w2 are weakly identified -- read those weights cautiously.
* medlev: scaffold uses the point-imputed column; the TODO marks the preferred
  in-model imputation path.

Usage
-----
    python scripts/wqs_model.py --smoke      # compile + prior-predictive + tiny sample
    python scripts/wqs_model.py --fit        # full NUTS sample -> data/model/
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "data" / "prepped"
OUT = ROOT / "data" / "model"
OUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["seq_gsm", "Simul_Gv", "Learn_Glr", "Plan_Gf"]
DOMAIN_LABELS = ["Gsm", "Gv", "Glr", "Gf"]
BANDS = ["q_prenatal", "q_age1", "q_age4"]
WINDOW_LABELS = ["prenatal", "age1", "age4"]
CONFOUNDERS = [
    "M_age_z", "sex_Female", "marital_Cohab", "marital_Other",
    "eth_Konkomba/Basare", "eth_Gonja/Dagomba/Mamprusi", "eth_Mo",
    "eth_Akan", "eth_Other",
]
MEDLEV = "medlev_ord_imp"   # TODO: switch to in-model imputation of medlev_ord (NaN-preserving)


def load_train():
    """Return arrays + the train standardisation stats (for back-transform)."""
    df = pd.read_csv(PREP / "train.csv")
    q = df[BANDS].to_numpy(float)                       # (N, 3) bands 0..3

    y_raw = df[OUTCOMES].to_numpy(float)                # (N, 4)
    y_mean, y_sd = y_raw.mean(0), y_raw.std(0, ddof=0)
    y = (y_raw - y_mean) / y_sd                         # standardised outcomes

    medlev = df[MEDLEV].to_numpy(float)
    medlev_c = medlev - medlev.mean()                   # centre for clean main/interaction split

    # Confounder matrix; fold the centred medlev main effect in as a column.
    X = df[CONFOUNDERS].to_numpy(float)
    X = np.column_stack([X, medlev_c])                  # (N, P+1)
    conf_names = CONFOUNDERS + ["medlev_c"]

    return dict(q=q, y=y, medlev_c=medlev_c, X=X, conf_names=conf_names,
                y_mean=y_mean, y_sd=y_sd, N=len(df))


def build_model(data):
    import pymc as pm
    import pytensor.tensor as pt

    coords = {
        "domain": DOMAIN_LABELS,
        "window": WINDOW_LABELS,
        "conf": data["conf_names"],
        "obs": np.arange(data["N"]),
    }
    with pm.Model(coords=coords) as model:
        q = pm.Data("q", data["q"], dims=("obs", "window"))
        Xc = pm.Data("X", data["X"], dims=("obs", "conf"))
        mod = pm.Data("medlev_c", data["medlev_c"], dims="obs")
        Y = pm.Data("Y", data["y"], dims=("obs", "domain"))

        # --- WQS simplex weights (shared across domains) ---
        w = pm.Dirichlet("w", a=np.ones(3), dims="window")
        BI = pm.Deterministic("BI", pt.dot(q, w), dims="obs")   # burden index

        # --- domain-specific coefficients ---
        b0 = pm.Normal("b0", 0.0, 1.0, dims="domain")
        b1 = pm.Normal("b1", 0.0, 1.0, dims="domain")           # burden effect
        theta = pm.Normal("theta", 0.0, 0.5, dims="domain")     # interaction (tighter)
        gamma = pm.Normal("gamma", 0.0, 1.0, dims=("conf", "domain"))

        mu = (
            b0
            + BI[:, None] * b1
            + (BI * mod)[:, None] * theta
            + pt.dot(Xc, gamma)
        )

        # --- residual covariance across the 4 domains (LKJ) ---
        chol, corr, sds = pm.LKJCholeskyCov(
            "chol", n=len(DOMAIN_LABELS), eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True,
        )
        pm.Deterministic("corr", corr, dims=("domain", "domain"))

        pm.MvNormal("Y_obs", mu=mu, chol=chol, observed=Y, dims=("obs", "domain"))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="compile + prior-pred + tiny sample")
    ap.add_argument("--fit", action="store_true", help="full NUTS sample")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    import pymc as pm
    import arviz as az

    data = load_train()
    print(f"Train: N={data['N']}, {len(OUTCOMES)} domains, {data['X'].shape[1]} adjustment cols")
    model = build_model(data)

    if args.smoke:
        with model:
            pm.sample_prior_predictive(samples=50, random_seed=42)
            idata = pm.sample(draws=50, tune=50, chains=2, cores=1,
                              random_seed=42, progressbar=False)
        print("\nSMOKE OK — model compiles and samples.")
        print(az.summary(idata, var_names=["w", "b1", "theta"], round_to=3))
        return

    if args.fit:
        with model:
            idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                              target_accept=0.95, random_seed=42)
            idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))
        idata.to_netcdf(OUT / "wqs_idata.nc")
        print(f"\nSaved posterior -> {OUT / 'wqs_idata.nc'}")
        print(az.summary(idata, var_names=["w", "b1", "theta"], round_to=3))
        return

    print("Nothing to do. Pass --smoke or --fit.")


if __name__ == "__main__":
    main()
