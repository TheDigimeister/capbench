"""Scores for one simulated run, plus offline reference values."""
import math

import numpy as np

from .baselines import api_first_token
from .lp import assign_lp
from .sim import HOUR


def score(w, r, pool):
    n = len(w.t)
    ok = r.outcome == "ok"
    j = np.maximum(r.model, 0)
    q = np.where(ok, w.Q[np.arange(n), j], 0.0)
    served = ~np.isnan(r.ttft)
    ttft = r.ttft[served]
    groups = {d: q[w.dataset == d].mean() for d in np.unique(w.dataset)}
    self_cost = sum(p.replicas * p.hourly_cost for p in pool if p.kind == "self") * r.horizon / HOUR
    out = {
        "acc_slo": float(q.mean()),
        "acc_slo_dataset_avg": float(np.mean(list(groups.values()))),
        "worst_group_acc": float(min(groups.values())),
        "slo_attainment": float(ok.mean()),
        "api_spend": r.spend,
        "api_spend_per_hour": r.spend / max(w.t[-1], 1e-9) * HOUR,
        "self_hosted_cost": self_cost,
        "ttft_p50": float(np.percentile(ttft, 50)) if len(ttft) else math.nan,
        "ttft_p95": float(np.percentile(ttft, 95)) if len(ttft) else math.nan,
        "ttft_p99": float(np.percentile(ttft, 99)) if len(ttft) else math.nan,
        "e2e_p50": float(np.nanpercentile(r.e2e, 50)) if served.any() else math.nan,
    }
    for k in ("ok", "late", "abandoned", "failed"):
        out[f"frac_{k}"] = float((r.outcome == k).mean())
    for k, v in r.n_refusals.items():
        out[f"refusals_{k}"] = v / n
    for jj, p in enumerate(pool):
        out[f"share_{p.name}"] = float((r.model == jj).mean())
    return out


def unconstrained_oracle(w):
    """Per-query best true quality with no capacity, budget or SLO."""
    return float(w.Q.max(1).mean())


def fluid_ceiling(w, pool, budget_per_hour, slo, window=HOUR, max_n=10_000, seed=0):
    """Approximate upper bound: an LP over the whole trace with true quality and
    true tokens. Self-hosted decode tokens are capped by peak batch throughput
    over the horizon, API calls by bucket size + refill, spend by the budget per
    started budget window; API cells whose median TTFT already misses the SLO are excluded.
    It ignores queueing and burstiness, so it is loose, and it lets decode run
    one mean service time past the last arrival. Runs longer than max_n arrivals
    are solved on a uniform subsample with every capacity scaled by the sample share."""
    n_all = len(w.t)
    H = float(w.t[-1])
    share = 1.0
    if n_all > max_n:
        keep = np.sort(np.random.default_rng([seed, 17]).choice(n_all, max_n, replace=False))
        share = max_n / n_all
        w = type(w)(**{k: getattr(w, k)[keep] for k in ("t", "row", "dataset", "prompt", "ptok", "ctok", "Q", "C")})
    n, m = w.Q.shape
    allowed = np.ones((n, m))
    tok_cap, req_cap = np.full(m, np.inf), np.full(m, np.inf)
    decode_cap = np.full(m, np.inf)
    for j, p in enumerate(pool):
        if p.kind == "self":
            tpot = p.saturated(w.ptok[:, j].mean(), w.ctok[:, j].mean())[1]
            tail = p.prefill(w.ptok[:, j].mean()) + w.ctok[:, j].mean() * tpot
            decode_cap[j] = share * p.replicas * (H + tail) * p.slots / tpot
        else:
            req_cap[j] = share * p.rpm * (1 + H / 60)
            tok_cap[j] = share * p.tpm * (1 + H / 60)
            allowed[:, j] = api_first_token(p, w.ctok[:, j]) <= slo
    api = np.array([p.kind == "api" for p in pool])
    cost = np.where(api, w.C, 0.0)
    budget = share * budget_per_hour * window / HOUR * math.ceil(H / window)
    obj, _ = assign_lp(w.Q, per_model=[(1.0, req_cap), (w.ptok + w.ctok, tok_cap), (w.ctok, decode_cap)],
                       global_=[(cost.ravel(), budget)], allowed=allowed)
    return obj / n
