import numpy as np
import pytest

from capbench.api import Router
from capbench.baselines import BatchLP, CapacityGreedy, Fixed, Random, ShortestQueue, StaticQC, StaticQCFallback
from capbench.config import API, Scenario, SelfHosted
from capbench.metrics import fluid_ceiling, score, unconstrained_oracle
from capbench.predictor import Predictions
from capbench.sim import Simulator, Workload

POOL = (SelfHosted("s0", replicas=2, slots=4, max_queue=8, t0=0.01, alpha=1.0),
        SelfHosted("s1", replicas=1, slots=4, max_queue=8, t0=0.02, alpha=0.5),
        API("a0", rpm=30, tpm=60_000, ttft_base=0.5),
        API("a1", rpm=10, tpm=20_000, ttft_base=1.0, hidden_reasoning=0.5))


def workload(n=400, rate=2.0, seed=0, m=4):
    rng = np.random.default_rng(seed)
    return Workload(t=np.cumsum(rng.exponential(1 / rate, n)), row=np.arange(n),
                    dataset=rng.choice(["x", "y", "z"], n), prompt=np.array(["p"] * n, object),
                    ptok=rng.uniform(100, 500, (n, m)), ctok=rng.uniform(50, 2000, (n, m)),
                    Q=(rng.random((n, m)) < [0.3, 0.4, 0.6, 0.8]).astype(float),
                    C=np.where(np.arange(m) >= 2, rng.uniform(1e-3, 1e-2, (n, m)), 0.0))


def preds(w, noise=0.2, seed=1):
    rng = np.random.default_rng(seed)
    return Predictions(np.clip(w.Q * 0.5 + 0.25 + noise * rng.standard_normal(w.Q.shape), 0, 1),
                       np.maximum(w.C, 1e-6), w.ctok * rng.uniform(0.7, 1.3, w.ctok.shape))


def routers(w, seed=0):
    P = preds(w)
    return [Random(seed), Fixed(0, "fixed0"), Fixed(3, "fixed3"), StaticQC(P, 10.0),
            ShortestQueue(P), CapacityGreedy(P, 10.0), BatchLP(P, 10.0, dt=2.0), StaticQCFallback(P, 10.0)]


class Recorder(Router):
    """Wraps a router and checks every observed state against hard limits."""

    def __init__(self, inner):
        self.inner, self.name, self.states = inner, inner.name, 0

    def reset(self, pool, info):
        super().reset(pool, info)
        self.inner.reset(pool, info)

    def route(self, q, s):
        self.states += 1
        for p, ms in zip(self.pool, s.models):
            if p.kind == "self":
                assert 0 <= ms.free_slots <= p.replicas * p.slots
                assert 0 <= ms.queue_len <= p.max_queue
            else:
                assert -1e-9 <= ms.rpm_headroom <= p.rpm + 1e-9
                assert -1e-9 <= ms.tpm_headroom <= p.tpm + 1e-9
        assert q.attempt < 3 and s.t <= q.deadline + 1e-9
        return self.inner.route(q, s)

    def on_event(self, e):
        self.inner.on_event(e)


@pytest.mark.parametrize("idx", range(8))
def test_limits_never_exceeded(idx):
    w = workload(rate=6.0)  # overloaded
    router = routers(w)[idx]
    sc = Scenario(slo_ttft=20.0)
    r = Simulator(POOL, sc, budget_per_hour=5.0).run(w, Recorder(router))
    for j, p in enumerate(POOL):
        if p.kind == "self":
            assert r.max_active[j] <= p.replicas * p.slots
        else:
            assert (r.min_bucket[j] >= -1e-6).all()
    # budget: total spend <= cap per started window + one call's overshoot per window
    win = sc.budget_window
    n_win = np.ceil(r.horizon / win)
    assert r.spend <= n_win * (5.0 * win / 3600 + w.C.max()) + 1e-9
    assert set(np.unique(r.outcome)) <= {"ok", "late", "abandoned", "failed"}
    ok = r.outcome == "ok"
    assert (r.ttft[ok] <= sc.slo_ttft + 1e-9).all()
    assert (r.model[ok] >= 0).all()


def test_single_request_timing_matches_formula():
    w = workload(n=1)
    p = POOL[1]
    r = Simulator(POOL, Scenario(), 1e9).run(w, Fixed(1, "f"))
    pre = p.prefill(w.ptok[0, 1])
    assert r.ttft[0] == pytest.approx(pre)
    assert r.e2e[0] == pytest.approx(pre + w.ctok[0, 1] * p.tpot(1))


def test_processor_sharing_two_requests():
    """Two simultaneous requests on one replica decode at tpot(2) until the shorter ends."""
    w = workload(n=2)
    w.t[:] = [0.0, 0.0]
    w.ptok[:] = 100.0
    w.ctok[:, 1] = [100.0, 300.0]
    p = POOL[1]
    r = Simulator(POOL, Scenario(), 1e9).run(w, Fixed(1, "f"))
    pre = p.prefill(100.0)
    t1 = pre + 100 * p.tpot(2)
    assert r.e2e[0] == pytest.approx(t1)
    assert r.e2e[1] == pytest.approx(t1 + 200 * p.tpot(1))


def test_queue_then_abandon():
    """A replica full of long jobs: extra requests queue, then abandon at the deadline."""
    pool = (SelfHosted("s", replicas=1, slots=1, max_queue=1, t0=1.0, alpha=0.0),)
    w = workload(n=3, m=1)
    w.t[:] = [0.0, 0.1, 0.2]
    w.ctok[:] = 100.0
    r = Simulator(pool, Scenario(slo_ttft=5.0, max_attempts=1), 1e9).run(w, Fixed(0, "f"))
    assert list(r.outcome) == ["ok", "abandoned", "failed"]  # 3rd: queue full, no retries


BIG = tuple(SelfHosted(p.name, replicas=1, slots=10_000, max_queue=10_000) if p.kind == "self"
            else API(p.name, rpm=1e9, tpm=1e12, ttft_base=0.01, ttft_sigma=0.0) for p in POOL)


def test_infinite_capacity_static_matches_offline_argmax():
    w = workload()
    P = preds(w)
    r = Simulator(BIG, Scenario(slo_ttft=1e6), 1e9).run(w, StaticQC(P, 0.0))
    assert (r.outcome == "ok").all()
    np.testing.assert_array_equal(r.model, P.Qhat.argmax(1))


def test_deterministic():
    w = workload(rate=6.0)
    a = score(w, Simulator(POOL, Scenario(seed=3), 5.0).run(w, routers(w, 3)[5]), POOL)
    b = score(w, Simulator(POOL, Scenario(seed=3), 5.0).run(w, routers(w, 3)[5]), POOL)
    assert a == b


def test_batch_lp_defers_at_most_one_tick():
    w = workload(rate=6.0)
    P = preds(w)
    r = Simulator(BIG, Scenario(slo_ttft=1e6), 1e9).run(w, BatchLP(P, 0.0, dt=2.0))
    assert (r.outcome == "ok").all()
    first = np.array([BIG[j].prefill(w.ptok[i, j]) if BIG[j].kind == "self" else 0.01 + BIG[j].hidden_reasoning
                      * w.ctok[i, j] * BIG[j].tpot for i, j in enumerate(r.model)])
    assert (r.ttft - first <= 2.0 + 1e-6).all()
    # with room for everyone, the LP places each query on its best predicted model
    np.testing.assert_array_equal(r.model, P.Qhat.argmax(1))


def test_ceilings_bound_every_router():
    w = workload(rate=6.0)
    sc = Scenario(slo_ttft=20.0)
    ceil = fluid_ceiling(w, POOL, 5.0, sc.slo_ttft)
    assert ceil <= unconstrained_oracle(w) + 1e-9
    for router in routers(w):
        acc = score(w, Simulator(POOL, sc, 5.0).run(w, router), POOL)["acc_slo"]
        assert acc <= ceil + 0.02, router.name  # fluid bound is approximate at the tail


def test_track_a_knn_router_respects_limits_and_learns():
    from capbench.api import TrainData
    from capbench.baselines import KNNCapacityGreedy
    rng = np.random.default_rng(5)
    n, m = 300, 4
    words = np.array(["alpha", "beta", "gamma"])
    kind = rng.integers(0, 3, n)
    prompts = np.array([f"{words[k]} question {i}" for i, k in enumerate(kind)], object)
    Q = np.zeros((n, m))
    Q[np.arange(n), kind] = 1.0  # prompt word decides which model is right
    train = TrainData(("s0", "s1", "a0", "a1"), prompts, np.array(["x"] * n), Q, np.full((n, m), 1e-3),
                      np.full((n, m), 200.0), np.full((n, m), 300.0))
    router = KNNCapacityGreedy(lam=0.0, k=5)
    router.fit(train)
    w = workload(n=200, rate=0.5)
    wk = rng.integers(0, 3, len(w.t))
    w.prompt = np.array([f"{words[k]} new {i}" for i, k in enumerate(wk)], object)
    w.Q = np.zeros_like(w.Q)
    w.Q[np.arange(len(w.t)), wk] = 1.0
    r = Simulator(BIG, Scenario(slo_ttft=1e6), 1e9).run(w, Recorder(router))
    assert (r.model == wk).mean() > 0.95


def test_serial_prefill_queues_ttft_and_slows_decode():
    from capbench.calibrate import simulate_closed_loop
    base = dict(slots=64, max_queue=10**6, a=0.05, b=2.5e-4, t0=0.07, alpha=0.0)
    legacy = SelfHosted("old", **base)
    contended = SelfHosted("new", serial_prefill=True, kappa=1e-3, beta=1.0, **base)
    ttft_old, tpot_old = simulate_closed_loop(legacy, 64, 1000, 100)
    ttft_new, tpot_new = simulate_closed_loop(contended, 64, 1000, 100)
    assert ttft_old == pytest.approx(0.05 + 0.25, rel=1e-6)          # every prefill runs in parallel
    assert ttft_new == pytest.approx(0.05 + 32 * 0.25, rel=0.05)     # FIFO: the median waits for ~32
    assert tpot_new > tpot_old
    # one request alone: identical except for the context term
    t1_old, _ = simulate_closed_loop(legacy, 1, 1000, 100)
    t1_new, _ = simulate_closed_loop(contended, 1, 1000, 100)
    assert t1_new == pytest.approx(t1_old, rel=1e-6)


def test_saturated_capacity_capped_by_prefill_server():
    s = SelfHosted("x", slots=64, a=0.0, b=1e-3, t0=1e-4, alpha=0.0, serial_prefill=True)
    lam, _ = s.saturated(ptok=1000, ctok=10)
    assert lam == pytest.approx(1.0, rel=1e-6)   # 1 / (b * ptok)
