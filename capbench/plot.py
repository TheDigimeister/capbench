"""Accuracy after SLO misses vs load level, one line per router: one row of
panels per scenario (hardware x arrivals), one column per budget level;
mean and 95% interval over seeds.

    python -m capbench.plot outputs/v1_s*.csv --out outputs/v1.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .suite import REFERENCE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    df = df[df["router"] != "predictor_unconstrained"]
    scen = df[["hardware", "arrivals"]].drop_duplicates().sort_values(["hardware", "arrivals"]).to_numpy()
    budgets = sorted(df["budget_frac"].unique())
    fig, axes = plt.subplots(len(scen), len(budgets), figsize=(6 * len(budgets), 4.2 * len(scen)),
                             sharey=True, squeeze=False)
    for i, (hw, arr) in enumerate(scen):
        for k, frac in enumerate(budgets):
            ax = axes[i][k]
            sub = df[(df["hardware"] == hw) & (df["arrivals"] == arr) & (df["budget_frac"] == frac)]
            for name, g in sub.groupby("router"):
                s = g.groupby("rho")["acc_slo"].agg(["mean", "std", "count"])
                ci = 1.96 * s["std"].fillna(0) / s["count"] ** 0.5
                style = dict(ls="--", lw=1, color="gray") if name in REFERENCE else dict(marker="o", lw=1.6)
                ax.errorbar(s.index, s["mean"], yerr=ci, label=name, capsize=3, **style)
            ax.set_title(f"{hw} / {arr} / budget {frac:g}x best-API spend", fontsize=10)
            ax.set_xlabel("rho (demand / self-hosted + API rate-limit capacity)")
            ax.grid(alpha=0.3)
        axes[i][0].set_ylabel("accuracy after SLO misses")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
