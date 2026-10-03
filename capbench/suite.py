"""Versioned scenario suites. A suite fixes everything a score depends on:
data snapshot, split, model pool, rate-limit snapshot, hardware profiles,
arrival traces, load and budget levels, SLO, run length and seeds.

The primary score of a router on a suite is its CapScore: accuracy after SLO
misses (`acc_slo`), averaged with equal weight over every (scenario, rho,
budget) cell, then over seeds. Reference rows (oracle, fluid ceiling) are not
routers and get no CapScore.

Never edit a released suite; add a new version instead.
"""

SUITES = {
    "v1": {
        "released": "2026-10-03",
        "data": {
            # LLMRouterBench pre-collected results (MIT), huggingface.co/datasets/NPULH/LLMRouterBench
            "llmrouterbench": "fetch.LLMROUTERBENCH",   # HF revision + sha256 pinned there
            "traces": "traces.SOURCES",                  # URL + sha256 per trace
            "split_seed": 0,
            "train_ratio": 0.5,
        },
        "rate_limit_snapshot": "2026-09-27",   # provider pages fetched on this date (config.py)
        "tier": "t1",
        "load_basis": "total",
        "slo_ttft": 30.0,
        "duration_min": 30.0,
        "seeds": [0, 1, 2, 3, 4],
        "rhos": [0.3, 0.7, 1.0, 1.3],
        "budgets": [0.1, 0.5],
        "scenarios": [
            {"hardware": "h100", "arrivals": "poisson"},
            {"hardware": "h100", "arrivals": "burstgpt"},
            {"hardware": "spark", "arrivals": "poisson"},
            {"hardware": "spark", "arrivals": "burstgpt"},
        ],
    },
    # v2: only hardware whose simulator profile passed a live validation (docs/CALIBRATION.md).
    # The Spark profile is the contention model fitted on 2026-10-03; H100 returns once validated.
    "v2": {
        "released": "2026-10-03",
        "data": {
            "llmrouterbench": "fetch.LLMROUTERBENCH",
            "traces": "traces.SOURCES",
            "split_seed": 0,
            "train_ratio": 0.5,
        },
        "rate_limit_snapshot": "2026-09-27",
        "tier": "t1",
        "load_basis": "total",
        "slo_ttft": 30.0,
        "duration_min": 30.0,
        "seeds": [0, 1, 2, 3, 4],
        "rhos": [0.3, 0.7, 1.0, 1.3],
        "budgets": [0.1, 0.5],
        "scenarios": [
            {"hardware": "spark_v2", "arrivals": "poisson"},
            {"hardware": "spark_v2", "arrivals": "burstgpt"},
        ],
    },
}

REFERENCE = ("oracle_unconstrained", "predictor_unconstrained", "fluid_ceiling")  # bounds, not routers


def cap_score(df):
    """CapScore per router from a suite's result rows: mean acc_slo over the
    suite's cells per seed, then mean and 95% interval over seeds."""
    keys = ["hardware", "arrivals", "rho", "budget_frac"]
    per_seed = df.groupby(["router", "seed"])["acc_slo"].mean()
    cells = df.groupby(["router", "seed"]).size()
    full = cells.groupby(level=0).transform("max")
    if (cells < full).any():
        raise ValueError("incomplete suite run: some seeds miss cells")
    s = per_seed.groupby(level=0).agg(["mean", "std", "count"])
    s["ci95"] = 1.96 * s["std"].fillna(0) / s["count"] ** 0.5
    s["n_cells"] = df.groupby("router")[keys].apply(lambda g: len(g.drop_duplicates()))
    return s.drop(columns="std").rename(columns={"mean": "cap_score", "count": "n_seeds"}) \
            .sort_values("cap_score", ascending=False)
