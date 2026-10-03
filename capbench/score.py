"""CapScore table from one or more suite result files (e.g. one per scenario).

    python -m capbench.score outputs/v1.csv
    python -m capbench.score outputs/v1_s*.csv --markdown
"""
import argparse

import pandas as pd

from .suite import REFERENCE, SUITES, cap_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    (name,) = df["suite"].unique()
    want = {(s["hardware"], s["arrivals"]) for s in SUITES[name]["scenarios"]}
    have = set(map(tuple, df[["hardware", "arrivals"]].drop_duplicates().to_numpy()))
    if have != want:
        raise SystemExit(f"suite {name} needs scenarios {sorted(want)}, files have {sorted(have)}")
    t = cap_score(df)
    if not args.markdown:
        print(t.round(4).to_string())
        return
    print("| router | CapScore | 95% CI | seeds |\n| --- | ---: | ---: | ---: |")
    for name, r in t.iterrows():
        label = f"*{name}* (reference)" if name in REFERENCE else f"`{name}`"
        print(f"| {label} | {r.cap_score:.3f} | ±{r.ci95:.3f} | {int(r.n_seeds)} |")


if __name__ == "__main__":
    main()
