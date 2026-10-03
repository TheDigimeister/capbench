"""Run routers through the simulator: either a released suite (the benchmark)
or an ad-hoc grid (exploration).

    python -m capbench.run --suite v1                                # all baselines, CapScore table
    python -m capbench.run --suite v1 --router my_pkg.routers:make   # your router, plus baselines
    python -m capbench.run --suite v1 --router my_pkg.routers:make --only-custom
    python -m capbench.run --seeds 0 --rhos 0.7 --n-arrivals 2000    # ad-hoc smoke run

A custom router is `module:attr`, where attr is called as attr(ctx) with a
RouterContext and returns a Router (a Router subclass works, since its
constructor then receives ctx).
"""
import argparse
import importlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B
from .config import API_LIMITS, HARDWARE, Scenario, make_pool
from .metrics import fluid_ceiling, score, unconstrained_oracle
from .predictor import shared_predictor
from .sim import Simulator
from .suite import SUITES, cap_score
from .workload import arrival_rate, budget_per_hour, load_bench, make_train_data, make_workload

OUT = Path(__file__).resolve().parent.parent / "outputs"


@dataclass(frozen=True)
class RouterContext:
    """What a router factory may use. P and the derived values are fit on the
    train split only; test-pool quality is never exposed."""
    pool: tuple          # model specs, pool order
    P: object            # Track B shared predictions (Qhat, Chat, That), indexed by Query.row
    lam: float           # price weight whose load-blind routing meets the budget (train split)
    tau: float           # Threshold baseline's cut-off
    rate: float          # mean arrival rate, requests/s
    budget: float        # $ per hour
    seed: int
    best_api: int        # most accurate API / self-hosted model on the train split
    best_self: int


BASELINES = {
    "random": lambda c: B.Random(c.seed),
    "always_best_api": lambda c: B.Fixed(c.best_api, "always_best_api"),
    "always_best_self": lambda c: B.Fixed(c.best_self, "always_best_self"),
    "threshold": lambda c: B.Threshold(c.P, c.best_api, c.best_self, c.tau),
    "static_qc": lambda c: B.StaticQC(c.P, c.lam),
    "static_qc_fallback": lambda c: B.StaticQCFallback(c.P, c.lam),
    "shortest_queue": lambda c: B.ShortestQueue(c.P),
    "capacity_greedy": lambda c: B.CapacityGreedy(c.P, c.lam),
    "batch_lp@2s": lambda c: B.BatchLP(c.P, c.lam, dt=2.0),
    "A:knn_capacity_greedy": lambda c: B.KNNCapacityGreedy(c.lam),
}


def load_factory(spec):
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--router expects module:attr, got {spec!r}")
    return getattr(importlib.import_module(module), attr)


def router_context(bench, P, rate, budget, seed):
    pool, tr = bench.pool, bench.train
    acc = bench.d["Q"][tr].mean(0)
    best_api = max((j for j, p in enumerate(pool) if p.kind == "api"), key=lambda j: acc[j])
    best_self = max((j for j, p in enumerate(pool) if p.kind == "self"), key=lambda j: acc[j])
    return RouterContext(pool, P, B.calibrate_lambda(P, pool, tr, rate, budget),
                         B.Threshold.calibrate(P, best_api, best_self, tr, rate, budget),
                         rate, budget, seed, best_api, best_self)


def run_grid(factories, *, seeds, rhos, budgets, tier, hardware, arrivals, load_basis, slo,
             duration_min=30.0, n_arrivals=None, api_limit_scale=1.0, references=True,
             log=lambda m: print(m, flush=True)):
    rows = []
    for seed in seeds:
        pool = make_pool(tier, hardware, api_limit_scale)
        bench = load_bench(pool)
        P = shared_predictor(bench, seed)
        train_data = make_train_data(bench)
        for rho in rhos:
            rate = arrival_rate(bench, rho, load_basis)
            n = n_arrivals or int(round(rate * duration_min * 60))
            w = make_workload(bench, rate, n, seed, arrivals)
            base = {"seed": seed, "rho": rho, "rate_per_s": rate, "n_arrivals": n,
                    "api_limit_scale": api_limit_scale, "tier": tier, "hardware": hardware,
                    "arrivals": arrivals, "load_basis": load_basis}
            for frac in budgets:
                sc = Scenario(rho=rho, load_basis=load_basis, arrivals=arrivals, budget_frac=frac,
                              slo_ttft=slo, n_arrivals=n, seed=seed)
                budget = budget_per_hour(bench, rate, frac)
                ctx = router_context(bench, P, rate, budget, seed)
                cfg = base | {"budget_frac": frac, "budget_per_hour": budget, "lambda": ctx.lam}
                start = len(rows)
                if references:
                    pred_acc = float(w.Q[np.arange(len(w.t)), P.Qhat[w.row].argmax(1)].mean())
                    rows.append(cfg | {"router": "oracle_unconstrained", "acc_slo": unconstrained_oracle(w)})
                    rows.append(cfg | {"router": "predictor_unconstrained", "acc_slo": pred_acc})
                    rows.append(cfg | {"router": "fluid_ceiling",
                                       "acc_slo": fluid_ceiling(w, bench.pool, budget, sc.slo_ttft,
                                                                sc.budget_window)})
                for make in factories:
                    router = make(ctx)
                    router.fit(train_data)
                    t0 = time.perf_counter()
                    r = Simulator(bench.pool, sc, budget).run(w, router)
                    rows.append(cfg | {"router": router.name, "sim_s": time.perf_counter() - t0}
                                | score(w, r, bench.pool))
                line = " ".join(f"{x['router']}={x['acc_slo']:.3f}" for x in rows[start:])
                log(f"{hardware}/{arrivals} seed={seed} rho={rho} budget={frac}: {line}")
    return rows


def run_suite(name, factories, references=True, scenarios=None):
    s = SUITES[name]
    rows = []
    for i, sc in enumerate(s["scenarios"]):
        if scenarios is not None and i not in scenarios:
            continue
        rows += run_grid(factories, seeds=s["seeds"], rhos=s["rhos"], budgets=s["budgets"], tier=s["tier"],
                         hardware=sc["hardware"], arrivals=sc["arrivals"], load_basis=s["load_basis"],
                         slo=s["slo_ttft"], duration_min=s["duration_min"], references=references)
    df = pd.DataFrame(rows)
    df.insert(0, "suite", name)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), help="run a released suite (ignores the grid options)")
    ap.add_argument("--router", action="append", default=[], help="custom router factory, module:attr (repeatable)")
    ap.add_argument("--only-custom", action="store_true", help="skip the built-in baselines")
    ap.add_argument("--scenario", type=int, nargs="+", help="suite scenario indices to run (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.3, 0.7, 1.0, 1.3])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.5], help="budget_frac values")
    ap.add_argument("--slo", type=float, default=Scenario.slo_ttft)
    ap.add_argument("--duration-min", type=float, default=30.0,
                    help="simulated minutes per run; arrivals = rate x duration (keeps budget windows per run fixed)")
    ap.add_argument("--n-arrivals", type=int, default=None, help="fixed arrival count instead of --duration-min")
    ap.add_argument("--arrivals", default="poisson", help="poisson | burstgpt | azure_code | azure_conv")
    ap.add_argument("--load-basis", default="total", choices=["total", "self"],
                    help="rho denominator: self-hosted + API rate limits, or self-hosted only")
    ap.add_argument("--tier", default="t1", choices=list(API_LIMITS), help="API rate-limit tier")
    ap.add_argument("--hardware", default="h100", choices=list(HARDWARE), help="self-hosted latency profile")
    ap.add_argument("--api-limit-scale", type=float, default=1.0, help="multiply every API rpm/tpm")
    ap.add_argument("--tag", default=None, help="output name (default: the suite name, or 'grid')")
    args = ap.parse_args()

    factories = ([] if args.only_custom else list(BASELINES.values())) + [load_factory(r) for r in args.router]
    if not factories:
        ap.error("no routers to run")
    OUT.mkdir(exist_ok=True)
    if args.suite:
        tag = args.tag or args.suite
        df = run_suite(args.suite, factories, references=not args.only_custom, scenarios=args.scenario)
        df.to_csv(OUT / f"{tag}.csv", index=False)
        print("wrote", f"outputs/{tag}.csv")
        if not args.scenario:
            print(cap_score(df).round(4).to_string())
        return
    rows = run_grid(factories, seeds=args.seeds, rhos=args.rhos, budgets=args.budgets, tier=args.tier,
                    hardware=args.hardware, arrivals=args.arrivals, load_basis=args.load_basis, slo=args.slo,
                    duration_min=args.duration_min, n_arrivals=args.n_arrivals,
                    api_limit_scale=args.api_limit_scale)
    tag = args.tag or "grid"
    pd.DataFrame(rows).to_csv(OUT / f"{tag}.csv", index=False)
    print("wrote", f"outputs/{tag}.csv")


if __name__ == "__main__":
    main()
