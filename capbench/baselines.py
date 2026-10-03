"""Baseline routers (Track B: all share the same Predictions P, indexed by test-pool row).

Self-hosted models have zero marginal price to a router (their cost is sunk),
so every cost term below uses P.Chat for API models only.
"""
import math

import numpy as np

from .api import DROP, Defer, Router
from .lp import assign_lp


def marginal_cost(P, pool):
    api = np.array([p.kind == "api" for p in pool])
    return np.where(api, P.Chat, 0.0)


def calibrate_lambda(P, pool, rows, rate, budget, iters=40):
    """Smallest price weight lam whose load-blind choice argmax(Qhat - lam*Chat)
    spends at most `budget` per hour at arrival `rate` (train rows, in-sample)."""
    Q, C = P.Qhat[rows], marginal_cost(P, pool)[rows]

    def spend(lam):
        j = np.argmax(Q - lam * C, axis=1)
        return rate * 3600 * C[np.arange(len(rows)), j].mean()

    if spend(0.0) <= budget:
        return 0.0
    lo, hi = 0.0, 1.0
    while spend(hi) > budget:
        hi *= 4
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if spend(mid) <= budget else (mid, hi)
    return hi


def api_first_token(spec, tokens):
    """Median TTFT an API call would have, from public specs and a token forecast."""
    return spec.ttft_base + spec.hidden_reasoning * tokens * spec.tpot


class Random(Router):
    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng([seed, 3])

    def route(self, q, s):
        return int(self.rng.integers(len(s.models)))


class Fixed(Router):
    def __init__(self, j, name):
        self.j, self.name = j, name

    def route(self, q, s):
        return self.j


class Threshold(Router):
    """RouteLLM-style: strong model if its predicted gain over the weak model
    exceeds tau; tau set so the strong share fits the budget. Load-blind."""
    name = "threshold"

    def __init__(self, P, strong, weak, tau):
        self.P, self.strong, self.weak, self.tau = P, strong, weak, tau

    @staticmethod
    def calibrate(P, strong, weak, rows, rate, budget):
        gain = P.Qhat[rows, strong] - P.Qhat[rows, weak]
        per_call = P.Chat[rows, strong].mean()
        share = min(1.0, budget / max(rate * 3600 * per_call, 1e-12))
        return float(np.quantile(gain, 1 - share)) if share < 1 else -np.inf

    def route(self, q, s):
        gain = self.P.Qhat[q.row, self.strong] - self.P.Qhat[q.row, self.weak]
        return self.strong if gain > self.tau else self.weak


class StaticQC(Router):
    """argmax Qhat - lam * Chat: a quality/cost router that ignores load."""
    name = "static_qc"

    def __init__(self, P, lam):
        self.P, self.lam = P, lam

    def reset(self, pool, info):
        super().reset(pool, info)
        self.C = marginal_cost(self.P, pool)

    def route(self, q, s):
        return int(np.argmax(self.P.Qhat[q.row] - self.lam * self.C[q.row]))


class StaticQCFallback(StaticQC):
    """StaticQC with a fallback chain: after a refusal (429, full queue, budget)
    it retries with the next-best model by Qhat - lam * Chat. Uses refusal
    events only, never load state."""
    name = "static_qc_fallback"

    def reset(self, pool, info):
        super().reset(pool, info)
        self.refused = {}

    def on_event(self, e):
        if e.kind in ("429", "queue_full", "budget"):
            self.refused.setdefault(e.qid, set()).add(e.model)
        elif e.kind in ("done", "abandoned"):
            self.refused.pop(e.qid, None)

    def route(self, q, s):
        v = self.P.Qhat[q.row] - self.lam * self.C[q.row]
        for j in self.refused.get(q.qid, ()):
            v = v.copy() if v is self.P.Qhat[q.row] else v
            v[j] = -np.inf
        return int(np.argmax(v))


class ShortestQueue(Router):
    """Load only: least-loaded self-hosted model; once every self-hosted wait
    estimate exceeds half the SLO, overflow to the cheapest API model."""
    name = "shortest_queue"

    def __init__(self, P):
        self.P = P

    def reset(self, pool, info):
        super().reset(pool, info)
        self.self_ids = [j for j, p in enumerate(pool) if p.kind == "self"]
        self.api_ids = [j for j, p in enumerate(pool) if p.kind == "api"]
        self.cheap_api = min(self.api_ids, key=lambda j: self.P.Chat[:, j].mean())

    def route(self, q, s):
        j = min(self.self_ids, key=lambda k: (s.models[k].est_wait, -s.models[k].free_slots))
        if s.models[j].est_wait > 0.5 * self.info["slo_ttft"] or s.models[j].queue_room <= 0:
            return self.cheap_api
        return j


class CapacityGreedy(Router):
    """argmax Qhat - lam * Chat over models that can serve within the SLO now:
    self-hosted with queue room and est_wait under 80% of the SLO; API with
    rate headroom for the forecast tokens, budget left, and a forecast TTFT
    under the SLO. Falls back to the shortest self-hosted queue."""
    name = "capacity_greedy"
    margin = 0.8

    def __init__(self, P, lam):
        self.P, self.lam = P, lam

    def reset(self, pool, info):
        super().reset(pool, info)
        self.C = marginal_cost(self.P, pool)
        self.self_ids = [j for j, p in enumerate(pool) if p.kind == "self"]

    def predict(self, q):
        """(Qhat, marginal Chat, That) row vectors for q. Track B: shared predictions."""
        return self.P.Qhat[q.row], self.C[q.row], self.P.That[q.row]

    def feasible(self, q, s, pred=None):
        _, cost, that = pred or self.predict(q)
        slo = self.margin * (q.deadline - s.t)
        ok = np.zeros(len(self.pool), bool)
        for j, (p, ms) in enumerate(zip(self.pool, s.models)):
            if p.kind == "self":
                ok[j] = ms.queue_room > 0 and ms.est_wait + p.prefill(q.ptok) < slo
            else:
                ok[j] = (ms.rpm_headroom >= 1 and ms.tpm_headroom >= q.ptok + that[j]
                         and s.budget_remaining > cost[j]
                         and api_first_token(p, that[j]) < slo)
        return ok

    def route(self, q, s):
        pred = self.predict(q)
        ok = self.feasible(q, s, pred)
        if not ok.any():
            return min(self.self_ids, key=lambda k: s.models[k].est_wait)
        v = np.where(ok, pred[0] - self.lam * pred[1], -np.inf)
        return int(np.argmax(v))


class BatchLP(CapacityGreedy):
    """Buffer arrivals for up to `dt` seconds, then solve one assignment LP over
    the buffer with current slot/queue room, rate headroom and remaining budget;
    fractional or unplaced queries fall back to CapacityGreedy."""

    def __init__(self, P, lam, dt=2.0):
        super().__init__(P, lam)
        self.dt, self.name = dt, f"batch_lp@{dt:g}s"

    def reset(self, pool, info):
        super().reset(pool, info)
        self.buffer, self.decided, self.pending = [], {}, set()

    def route(self, q, s):
        if q.attempt > 0:
            return super().route(q, s)
        if q.qid in self.decided:
            return self.decided.pop(q.qid)
        if q.qid not in self.pending:
            self.pending.add(q.qid)
            self.buffer.append(q)
            return Defer((math.floor(s.t / self.dt) + 1) * self.dt)
        # q's tick has come: the first buffered query to be routed solves for all
        self.pending.discard(q.qid)
        if self.buffer:
            self.solve(s)
        return self.decided.pop(q.qid) if q.qid in self.decided else super().route(q, s)

    def solve(self, s):
        qs, self.buffer = self.buffer, []
        self.pending -= {q.qid for q in qs}
        rows = np.array([q.row for q in qs])
        ptok = np.array([q.ptok for q in qs])
        allowed = np.array([self.feasible(q, s) for q in qs], float)
        m = len(self.pool)
        req_cap, tok_cap = np.full(m, np.inf), np.full(m, np.inf)
        slo = self.info["slo_ttft"]
        for j, (p, ms) in enumerate(zip(self.pool, s.models)):
            if p.kind == "self":
                total = p.replicas * p.slots
                room = min(ms.queue_room, int(self.margin * slo * total / max(ms.svc_ewma, 1e-6)) - ms.queue_len)
                req_cap[j] = max(ms.free_slots, 0) + max(room, 0)
            else:
                req_cap[j], tok_cap[j] = ms.rpm_headroom, ms.tpm_headroom
        tokens = ptok[:, None] + self.P.That[rows]
        values = self.P.Qhat[rows] - self.lam * self.C[rows]
        values = values - values.min() + 1e-3  # placing beats leaving unplaced
        _, x = assign_lp(values, per_model=[(1.0, req_cap), (tokens, tok_cap)],
                         global_=[(self.C[rows].ravel(), max(s.budget_remaining, 0.0))],
                         allowed=allowed)
        for k, q in enumerate(qs):
            if x[k].max() > 0.5:
                self.decided[q.qid] = int(x[k].argmax())


class KNNCapacityGreedy(CapacityGreedy):
    """Track A example: its own predictor (TF-IDF cosine k-nearest neighbours over
    the train prompts) feeding CapacityGreedy's allocation. Sees prompt text only,
    never the test-pool row."""
    name = "A:knn_capacity_greedy"

    def __init__(self, lam, k=20):
        super().__init__(None, lam)
        self.k = k

    def fit(self, train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, max_features=50_000)
        self.X = self.vec.fit_transform([str(p) for p in train.prompt])
        self.Q, self.Cost, self.T = (np.asarray(x, float) for x in (train.Q, train.C, train.ctok))
        self.cache = {}

    def reset(self, pool, info):
        Router.reset(self, pool, info)
        self.api = np.array([p.kind == "api" for p in pool])
        self.self_ids = [j for j, p in enumerate(pool) if p.kind == "self"]

    def predict(self, q):
        if q.prompt not in self.cache:
            sim = (self.X @ self.vec.transform([str(q.prompt)]).T).toarray().ravel()
            near = np.argpartition(-sim, self.k)[:self.k]
            w = np.maximum(sim[near], 1e-6)
            avg = lambda A: (A[near] * w[:, None]).sum(0) / w.sum()
            self.cache[q.prompt] = (avg(self.Q), np.where(self.api, avg(self.Cost), 0.0), avg(self.T))
        return self.cache[q.prompt]
