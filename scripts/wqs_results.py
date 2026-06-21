"""
Summarise the fitted WQS posterior for the write-up.

Reads data/model/wqs_idata.nc (produced by `wqs_model.py --fit`) and prints:
  * convergence (max r_hat, min ESS)
  * window weights w  (the burden split) with 94% HDI
  * burden effects b1 per domain, in SD units AND raw KABC-score points,
    plus P(effect < 0)  (posterior prob pollution is harmful)
  * interaction theta per domain (+ P<0) and SES main effect b_med
  * residual correlations across domains
  * LOO (predictive accuracy)
  * posterior-predictive sanity (observed vs replicated mean/sd per domain)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
IDATA = ROOT / "data" / "model" / "wqs_idata.nc"
PREP = ROOT / "data" / "prepped" / "train.csv"
OUTCOMES = ["seq_gsm", "Simul_Gv", "Learn_Glr", "Plan_Gf"]
DOMS = ["Gsm", "Gv", "Glr", "Gf"]


def main():
    idata = az.from_netcdf(IDATA, engine="h5netcdf")
    post = idata.posterior
    y_sd = pd.read_csv(PREP)[OUTCOMES].std(ddof=0).to_numpy()  # for back-transform

    print("=" * 64)
    s = az.summary(idata, var_names=["w", "b1", "theta", "b_med"])
    print("CONVERGENCE: max r_hat = %.3f | min ess_bulk = %d"
          % (s["r_hat"].max(), int(s["ess_bulk"].min())))
    div = int(idata.sample_stats["diverging"].sum())
    print(f"divergences = {div}")

    def hdi(name):
        return az.hdi(idata, var_names=[name])[name].values

    print("\n--- WINDOW WEIGHTS w (burden split; sum to 1) ---")
    wm = post["w"].mean(("chain", "draw")).values
    wh = hdi("w")
    for lab, m, h in zip(["prenatal", "age1", "age4"], wm, wh):
        print(f"  {lab:9s}: {m:5.2f}  [{h[0]:.2f}, {h[1]:.2f}]")

    print("\n--- BURDEN EFFECT b1 per domain ---")
    print("  (SD units; raw = SD x outcome sd; P<0 = prob harmful)")
    b1 = post["b1"]
    b1m = b1.mean(("chain", "draw")).values
    b1h = hdi("b1")
    pneg = (b1 < 0).mean(("chain", "draw")).values
    for d, m, h, p, sd in zip(DOMS, b1m, b1h, pneg, y_sd):
        print(f"  {d:4s}: {m:+.3f} SD [{h[0]:+.3f},{h[1]:+.3f}]  "
              f"= {m*sd:+.2f} pts  P(<0)={p:.2f}")

    print("\n--- INTERACTION theta (burden x SES) per domain ---")
    th = post["theta"]
    thm = th.mean(("chain", "draw")).values
    thh = hdi("theta")
    thp = (th < 0).mean(("chain", "draw")).values
    for d, m, h, p in zip(DOMS, thm, thh, thp):
        print(f"  {d:4s}: {m:+.3f} [{h[0]:+.3f},{h[1]:+.3f}]  P(<0)={p:.2f}")

    print("\n--- SES main effect b_med per domain ---")
    bm = post["b_med"].mean(("chain", "draw")).values
    bmh = hdi("b_med")
    for d, m, h in zip(DOMS, bm, bmh):
        print(f"  {d:4s}: {m:+.3f} [{h[0]:+.3f},{h[1]:+.3f}]")

    print("\n--- residual correlations across domains ---")
    corr = post["corr"].mean(("chain", "draw")).values
    print(pd.DataFrame(corr, index=DOMS, columns=DOMS).round(2).to_string())

    print("\n--- LOO ---")
    try:
        print(az.loo(idata))
    except Exception as e:
        print("LOO unavailable:", e)

    if "posterior_predictive" in idata.groups():
        print("\n--- posterior-predictive check (per domain) ---")
        yobs = idata.observed_data["Y_obs"].values
        yrep = idata.posterior_predictive["Y_obs"].values  # (c,d,obs,dom)
        yrep = yrep.reshape(-1, yrep.shape[-2], yrep.shape[-1])
        for j, d in enumerate(DOMS):
            print(f"  {d:4s}: obs mean/sd = {yobs[:,j].mean():+.2f}/{yobs[:,j].std():.2f}"
                  f"  | rep = {yrep[:,:,j].mean():+.2f}/{yrep[:,:,j].std():.2f}")


if __name__ == "__main__":
    main()
