"""
Full evaluation of the fitted WQS model: diagnostics + train/val/test performance.

Run AFTER `wqs_model.py --fit`. Produces data/model/evaluation.json and prints a
report. NOTHING here interprets the science -- it only computes the numbers the
write-up will quote, and only once train + val + test + diagnostics all succeed.

What it does
------------
1. Convergence diagnostics: max r_hat, min ESS, divergences, energy BFMI.
2. In-sample fit: LOO (elpd) from a manually-computed log-likelihood, and a
   posterior-predictive mean/sd check per domain.
3. Out-of-sample (val, test): expected log predictive density (elppd) and RMSE
   per domain. For held-out children the 43.5%-missing SES (medlev) is integrated
   out by drawing it from the posterior imputation submodel -- the same joint
   model used in training, so uncertainty is honoured out-of-sample too.

The multivariate-Normal log density is evaluated per posterior draw with the
covariance reconstructed as Sigma = diag(sds) corr diag(sds).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "model"
PREP = ROOT / "data" / "prepped"
OUTCOMES = ["seq_gsm", "Simul_Gv", "Learn_Glr", "Plan_Gf"]
DOMS = ["Gsm", "Gv", "Glr", "Gf"]
BANDS = ["q_prenatal", "q_age1", "q_age4"]
CONF = ["M_age_z", "sex_Female", "marital_Cohab", "marital_Other",
        "eth_Konkomba/Basare", "eth_Gonja/Dagomba/Mamprusi", "eth_Mo",
        "eth_Akan", "eth_Other"]
RNG = np.random.default_rng(7)


def load_idata():
    nc, pk = MODEL / "wqs_idata.nc", MODEL / "wqs_idata.pkl"
    if nc.exists():
        try:
            return az.from_netcdf(nc, engine="h5netcdf")
        except Exception:
            return az.from_netcdf(nc)
    import cloudpickle
    with open(pk, "rb") as fh:
        return cloudpickle.load(fh)


def stack(post, name):
    """Return posterior array with first axis = S = chain*draw."""
    a = post[name].values
    return a.reshape((-1,) + a.shape[2:])


def mvn_loglik(y, mu_s, Sigma_s):
    """Per-obs log MvNormal density for one draw. y,mu_s:(N,D) Sigma_s:(D,D)."""
    L = np.linalg.cholesky(Sigma_s)
    resid = (y - mu_s).T                       # (D, N)
    z = np.linalg.solve(L, resid)              # (D, N)
    quad = np.sum(z * z, axis=0)               # (N,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    D = y.shape[1]
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def split_arrays(name, y_mean, y_sd):
    df = pd.read_csv(PREP / f"{name}.csv")
    q = df[BANDS].to_numpy(float)
    X = df[CONF].to_numpy(float)
    medlev = df["medlev_ord"].to_numpy(float)              # NaN where missing
    y = (df[OUTCOMES].to_numpy(float) - y_mean) / y_sd     # standardised by TRAIN
    return q, X, medlev, y, df


def main():
    idata = load_idata()
    post = idata.posterior

    train = pd.read_csv(PREP / "train.csv")
    y_mean = train[OUTCOMES].mean().to_numpy()
    y_sd = train[OUTCOMES].std(ddof=0).to_numpy()
    medlev_mean = float(np.nanmean(train["medlev_ord"].to_numpy(float)))

    # posterior arrays
    b0, b1 = stack(post, "b0"), stack(post, "b1")
    theta, b_med = stack(post, "theta"), stack(post, "b_med")
    gamma = stack(post, "gamma")            # (S, conf, dom)
    w = stack(post, "w")                    # (S, 3)
    corr, sds = stack(post, "corr"), stack(post, "sds")
    a0, a_imp, a_sig = stack(post, "imp_a0"), stack(post, "imp_a"), stack(post, "imp_sigma")
    S = b0.shape[0]
    report = {}

    # ---------- 1. diagnostics ----------
    # Exclude 'corr' (constant unit diagonal -> NaN r_hat) from the summary.
    s = az.summary(idata, var_names=["w", "b1", "theta", "b_med", "gamma",
                                     "imp_a0", "imp_a", "imp_sigma", "sds"])
    # BFMI manually from energy (per chain): sum(diff^2)/sum((E-mean)^2).
    E = idata.sample_stats["energy"].values            # (chain, draw)
    bfmi = float(np.min([np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2)
                         for e in E]))
    report["diagnostics"] = {
        "n_samples": int(S),
        "max_rhat": round(float(np.nanmax(s["r_hat"])), 4),
        "min_ess_bulk": int(np.nanmin(s["ess_bulk"])),
        "min_ess_tail": int(np.nanmin(s["ess_tail"])),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "min_bfmi": round(bfmi, 3),
    }

    # ---------- shared predictor helper ----------
    def mu_and_ll(q, X, medlev, y, stored_BI=None, stored_medc=None):
        """Return (ll matrix S x N, mean mu N x D). Imputes medlev if not stored."""
        N = y.shape[0]
        Xg = np.einsum("nc,scd->snd", X, gamma)            # (S, N, D)
        if stored_BI is not None:
            BI = stored_BI                                  # (S, N)
        else:
            BI = np.einsum("nj,sj->sn", q, w)               # (S, N)
        if stored_medc is not None:
            medc = stored_medc                              # (S, N)
        else:
            mu_med = a0[:, None] + np.einsum("nc,sc->sn", X, a_imp)   # (S, N)
            draw = mu_med + a_sig[:, None] * RNG.standard_normal((S, N))
            obs = ~np.isnan(medlev)
            medc = np.where(obs[None, :], np.nan_to_num(medlev)[None, :], draw) - medlev_mean
        mu = (b0[:, None, :]
              + BI[:, :, None] * b1[:, None, :]
              + (BI * medc)[:, :, None] * theta[:, None, :]
              + medc[:, :, None] * b_med[:, None, :]
              + Xg)                                         # (S, N, D)
        ll = np.empty((S, N))
        for si in range(S):
            Sig = (sds[si][:, None] * corr[si] * sds[si][None, :])
            ll[si] = mvn_loglik(y, mu[si], Sig)
        return ll, mu.mean(0)

    def elppd_rmse(ll, mu_mean, y):
        from scipy.special import logsumexp
        lpd = logsumexp(ll, axis=0) - np.log(S)             # (N,)
        elppd = float(lpd.sum())
        rmse_sd = np.sqrt(((mu_mean - y) ** 2).mean(0))     # per domain, SD units
        rmse_raw = rmse_sd * y_sd
        return elppd, rmse_sd, rmse_raw

    # ---------- 2. in-sample (train): use stored BI & medlev_c ----------
    BI_tr = stack(post, "BI")                               # (S, N_train)
    medc_tr = stack(post, "medlev_c")                       # (S, N_train)
    q_tr, X_tr, med_tr, y_tr, _ = split_arrays("train", y_mean, y_sd)
    ll_tr, mu_tr = mu_and_ll(q_tr, X_tr, med_tr, y_tr, stored_BI=BI_tr, stored_medc=medc_tr)
    el, rsd, rraw = elppd_rmse(ll_tr, mu_tr, y_tr)

    report["train"] = {
        "n": int(y_tr.shape[0]), "elppd": round(el, 1),
        "rmse_raw": {d: round(float(r), 2) for d, r in zip(DOMS, rraw)},
    }
    # PSIS-LOO: attach the train log-likelihood as a group on the DataTree.
    try:
        import xarray as xr
        nc, nd = idata.posterior.sizes["chain"], idata.posterior.sizes["draw"]
        ll_ds = xr.Dataset({"Y_obs": (["chain", "draw", "obs_id"],
                                      ll_tr.reshape(nc, nd, -1))})
        idata["log_likelihood"] = ll_ds
        loo = az.loo(idata, var_name="Y_obs")
        report["train"]["loo_elpd"] = round(float(loo.elpd), 1)
        report["train"]["loo_p"] = round(float(loo.p), 1)
        report["train"]["loo_se"] = round(float(loo.se), 1)
        pk = np.asarray(loo.pareto_k.values)
        report["train"]["pareto_k_max"] = round(float(pk.max()), 2)
        report["train"]["pareto_k_pct_good"] = round(float((pk <= 0.7).mean() * 100), 1)
    except Exception as e:
        report["train"]["loo_error"] = str(e)

    # ---------- 3. out-of-sample: val, test ----------
    for name in ("val", "test"):
        q, X, med, y, _ = split_arrays(name, y_mean, y_sd)
        ll, mu = mu_and_ll(q, X, med, y)
        el, rsd, rraw = elppd_rmse(ll, mu, y)
        report[name] = {
            "n": int(y.shape[0]),
            "elppd": round(el, 1),
            "elppd_per_obs": round(el / y.shape[0], 3),
            "rmse_sd": {d: round(float(r), 3) for d, r in zip(DOMS, rsd)},
            "rmse_raw": {d: round(float(r), 2) for d, r in zip(DOMS, rraw)},
        }

    # train elppd per obs too, for comparison
    report["train"]["elppd_per_obs"] = round(report["train"]["elppd"] / report["train"]["n"], 3)

    # ---------- 4. posterior-predictive check (train) ----------
    has_ppc = hasattr(idata, "posterior_predictive")
    if has_ppc:
        yo = idata.observed_data["Y_obs"].values
        yr = idata.posterior_predictive["Y_obs"].values
        yr = yr.reshape(-1, yr.shape[-2], yr.shape[-1])
        report["ppc_train"] = {
            d: {"obs_mean": round(float(yo[:, j].mean()), 2),
                "rep_mean": round(float(yr[:, :, j].mean()), 2),
                "obs_sd": round(float(yo[:, j].std()), 2),
                "rep_sd": round(float(yr[:, :, j].std()), 2)}
            for j, d in enumerate(DOMS)
        }

    with open(MODEL / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {MODEL / 'evaluation.json'}")


if __name__ == "__main__":
    main()
