"""Discrete-event simulator: self-hosted replicas with continuous batching,
API models behind token-bucket rate limits, and a spend cap over a trailing window.

Self-hosted decode is processor sharing: every active sequence on a replica
advances one token per tpot(n) seconds, n = occupied slots. Each replica keeps
a virtual clock V (tokens emitted per sequence so far); a sequence finishes
when V reaches its start mark plus its output length, so rate changes on
arrival/departure only require advancing V, not rescheduling every sequence.

Outcomes per arrival: "ok" (first token within SLO), "late" (served, TTFT
over SLO), "abandoned" (still queued or unrouted at its deadline), "failed"
(out of attempts or dropped). Only "ok" earns quality.
"""
import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .api import DROP, Defer, Event, ModelState, Query, SystemState

HOUR = 3600.0
EWMA = 0.1


@dataclass
class Workload:
    t: np.ndarray        # (n,) arrival times, s, sorted
    row: np.ndarray      # (n,) test-pool row
    dataset: np.ndarray  # (n,)
    prompt: np.ndarray   # (n,)
    ptok: np.ndarray     # (n, m) input tokens per model
    ctok: np.ndarray     # (n, m) actual output tokens (hidden from routers)
    Q: np.ndarray        # (n, m) true quality
    C: np.ndarray        # (n, m) true $ cost of an API call


@dataclass
class Result:
    model: np.ndarray    # (n,) serving model, -1 if none
    outcome: np.ndarray  # (n,) str
    ttft: np.ndarray     # (n,) s from arrival, nan if never started
    e2e: np.ndarray      # (n,) s from arrival to last token
    attempts: np.ndarray
    spend: float         # $ API spend
    n_refusals: dict
    horizon: float       # s, time of last event
    max_active: np.ndarray  # (m,) peak occupied slots per self-hosted model (api: 0)
    min_bucket: np.ndarray  # (m, 2) lowest rpm/tpm bucket level seen (api)


class _Replica:
    __slots__ = ("V", "last", "heap", "n", "version")

    def __init__(self):
        self.V, self.last, self.heap, self.n, self.version = 0.0, 0.0, [], 0, 0


class Simulator:
    def __init__(self, pool, scenario, budget_per_hour):
        self.pool, self.sc = pool, scenario
        self.window = float(scenario.budget_window)
        self.budget = float(budget_per_hour) * self.window / HOUR  # $ per trailing window

    # ------------------------------------------------------------ plumbing
    def _push(self, t, kind, *args):
        self._seq += 1
        heapq.heappush(self._ev, (t, self._seq, kind, args))

    def _advance(self, j, r, now):
        rep = self.reps[j][r]
        if rep.n:
            rep.V += (now - rep.last) / self.pool[j].tpot(rep.n)
        rep.last = now

    def _reschedule(self, j, r):
        rep = self.reps[j][r]
        rep.version += 1
        if rep.heap:
            dt = max(rep.heap[0][0] - rep.V, 0.0) * self.pool[j].tpot(rep.n)
            self._push(rep.last + dt, "decode", j, r, rep.version)

    def _bucket(self, j, now):
        spec, b = self.pool[j], self.buckets[j]
        dt = now - b[2]
        b[0] = min(spec.rpm, b[0] + dt * spec.rpm / 60.0)
        b[1] = min(spec.tpm, b[1] + dt * spec.tpm / 60.0)
        b[2] = now
        return b

    def _spent_in_window(self, now):
        while self.spend_log and self.spend_log[0][0] <= now - self.window:
            self.spend_window -= self.spend_log.popleft()[1]
        return self.spend_window

    def state(self, now):
        ms = []
        for j, spec in enumerate(self.pool):
            if spec.kind == "self":
                busy = sum(rep.n for rep in self.reps[j])
                total = spec.replicas * spec.slots
                free = total - busy
                ql = self.qlen[j]
                est = 0.0 if free > 0 else (ql + 1) * self.svc_ewma[j] / total
                ms.append(ModelState(spec.name, "self", free, ql, spec.max_queue - ql, est,
                                     self.svc_ewma[j], self.ttft_ewma[j], math.inf, math.inf))
            else:
                b = self._bucket(j, now)
                ms.append(ModelState(spec.name, "api", -1, 0, -1, 0.0, 0.0, self.ttft_ewma[j], b[0], b[1]))
        return SystemState(now, tuple(ms), self.budget - self._spent_in_window(now))

    # ------------------------------------------------------------ run
    def run(self, w: Workload, router):
        pool, sc = self.pool, self.sc
        n, m = w.Q.shape
        self._ev, self._seq = [], 0
        self.reps = [[_Replica() for _ in range(p.replicas)] if p.kind == "self" else [] for p in pool]
        self.queues = [deque() for _ in pool]
        self.qlen = [0] * m
        self.buckets = [[p.rpm, p.tpm, 0.0] if p.kind == "api" else None for p in pool]
        self.spend_log, self.spend_window, spend = deque(), 0.0, 0.0
        self.svc_ewma = [p.prefill(400) + 2000 * p.tpot(p.slots) if p.kind == "self" else 0.0 for p in pool]
        self.ttft_ewma = [0.0] * m
        max_active = np.zeros(m, int)
        min_bucket = np.full((m, 2), np.inf)

        model = np.full(n, -1)
        outcome = np.full(n, "failed", dtype=object)
        ttft, e2e = np.full(n, np.nan), np.full(n, np.nan)
        attempts = np.zeros(n, int)
        start_t = np.full(n, np.nan)
        queued = np.zeros(n, bool)
        refusals = {"429": 0, "queue_full": 0, "budget": 0}
        deadline = w.t + sc.slo_ttft
        rng = np.random.default_rng([sc.seed, 7])

        router.reset(pool, {"slo_ttft": sc.slo_ttft, "budget_per_window": self.budget, "budget_window": self.window,
                            "pool": pool, "horizon_hint": float(w.t[-1])})
        for i in range(n):
            self._push(w.t[i], "route", i)

        def query(i):
            return Query(i, int(w.row[i]), str(w.dataset[i]), w.prompt[i], float(w.ptok[i].mean()),
                         float(w.t[i]), float(deadline[i]), int(attempts[i]))

        def refuse(i, j, kind, now):
            refusals[kind] += 1
            router.on_event(Event(kind, i, j, now))
            attempts[i] += 1
            if attempts[i] < sc.max_attempts:
                self._push(now + sc.retry_delay, "route", i)
            else:
                outcome[i] = "failed"

        def start(j, i, now):
            """Occupy a slot on the least-loaded replica and begin prefill."""
            spec = pool[j]
            r = min(range(spec.replicas), key=lambda k: self.reps[j][k].n)
            self._advance(j, r, now)
            self.reps[j][r].n += 1
            max_active[j] = max(max_active[j], sum(rep.n for rep in self.reps[j]))
            self._reschedule(j, r)
            model[i], start_t[i] = j, now
            self._push(now + spec.prefill(w.ptok[i, j]), "prefill", j, r, i)

        def free_slot(j):
            return any(rep.n < pool[j].slots for rep in self.reps[j])

        def pump(j, now):
            q = self.queues[j]
            while q and free_slot(j):
                i = q.popleft()
                if not queued[i]:
                    continue  # abandoned while queued
                queued[i] = False
                self.qlen[j] -= 1
                start(j, i, now)

        now = 0.0
        while self._ev:
            now, _, kind, args = heapq.heappop(self._ev)
            if kind == "route":
                i = args[0]
                if now > deadline[i]:
                    outcome[i] = "abandoned"
                    router.on_event(Event("abandoned", i, -1, now))
                    continue
                d = router.route(query(i), self.state(now))
                if isinstance(d, Defer):
                    self._push(max(d.until, now), "route", i)
                    continue
                if d is DROP:
                    outcome[i] = "failed"
                    continue
                j = int(d)
                spec = pool[j]
                if spec.kind == "self":
                    if free_slot(j):
                        start(j, i, now)
                    elif self.qlen[j] < spec.max_queue:
                        self.queues[j].append(i)
                        self.qlen[j] += 1
                        queued[i] = True
                        model[i] = j
                        self._push(deadline[i], "abandon", j, i)
                    else:
                        refuse(i, j, "queue_full", now)
                else:
                    if self._spent_in_window(now) >= self.budget:
                        refuse(i, j, "budget", now)
                        continue
                    b = self._bucket(j, now)
                    tok = w.ptok[i, j] + w.ctok[i, j]
                    if b[0] < 1 or b[1] < tok:
                        refuse(i, j, "429", now)
                        continue
                    b[0] -= 1
                    b[1] -= tok
                    min_bucket[j] = np.minimum(min_bucket[j], b[:2])
                    cost = float(w.C[i, j])
                    spend += cost
                    self.spend_log.append((now, cost))
                    self.spend_window += cost
                    model[i], start_t[i] = j, now
                    first = spec.ttft_base * math.exp(spec.ttft_sigma * rng.standard_normal())
                    first += spec.hidden_reasoning * w.ctok[i, j] * spec.tpot
                    self._push(now + first, "first", j, i)
                    done = first + (1 - spec.hidden_reasoning) * w.ctok[i, j] * spec.tpot
                    self._push(now + max(done, first), "api_done", j, i)
            elif kind == "abandon":
                j, i = args
                if queued[i]:
                    queued[i] = False
                    self.qlen[j] -= 1
                    model[i] = -1
                    outcome[i] = "abandoned"
                    router.on_event(Event("abandoned", i, j, now))
            elif kind == "prefill":
                j, r, i = args
                self._advance(j, r, now)
                rep = self.reps[j][r]
                heapq.heappush(rep.heap, (rep.V + max(w.ctok[i, j], 1.0), i))
                self._reschedule(j, r)
                self._first(i, j, now, w, ttft, outcome, deadline)
            elif kind == "first":
                j, i = args
                self._first(i, j, now, w, ttft, outcome, deadline)
            elif kind == "decode":
                j, r, version = args
                rep = self.reps[j][r]
                if version != rep.version:
                    continue
                self._advance(j, r, now)
                while rep.heap and rep.heap[0][0] <= rep.V + 1e-6:
                    _, i = heapq.heappop(rep.heap)
                    rep.n -= 1
                    e2e[i] = now - w.t[i]
                    self.svc_ewma[j] += EWMA * ((now - start_t[i]) - self.svc_ewma[j])
                    router.on_event(Event("done", i, j, now, ttft[i]))
                self._reschedule(j, r)
                pump(j, now)
            elif kind == "api_done":
                j, i = args
                e2e[i] = now - w.t[i]
                router.on_event(Event("done", i, j, now, ttft[i]))

        return Result(model, outcome, ttft, e2e, attempts, spend, refusals, now, max_active, min_bucket)

    def _first(self, i, j, now, w, ttft, outcome, deadline):
        ttft[i] = now - w.t[i]
        self.ttft_ewma[j] += EWMA * (ttft[i] - self.ttft_ewma[j])
        outcome[i] = "ok" if now <= deadline[i] + 1e-9 else "late"
