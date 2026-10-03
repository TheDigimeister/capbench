import numpy as np
import pandas as pd
import pytest

from capbench import fetch, run, source, traces
from capbench.suite import SUITES, cap_score


def test_split_ignores_run_seed():
    rng = np.random.default_rng(0)
    n = 400
    ds = rng.choice(["a", "b"], n)
    prompt = np.array([f"{ds[i]}{i % 150}" for i in range(n)], object)
    idx = np.arange(n)
    tr, te = source.split_by_dataset_prompt(ds, idx, prompt, 0.5, seed=0)
    assert not set(prompt[tr]) & set(prompt[te])
    assert len(tr) + len(te) == n
    import inspect
    from capbench import workload
    assert inspect.signature(workload.load_bench).parameters["split_seed"].default == workload.SPLIT_SEED


def test_cap_score_averages_cells_then_seeds():
    rows = [{"router": r, "seed": s, "hardware": "h", "arrivals": a, "rho": 1.0, "budget_frac": 0.1,
             "acc_slo": v + 0.1 * s}
            for r, v in (("x", 0.5), ("y", 0.3)) for s in (0, 1) for a in ("p", "b")]
    t = cap_score(pd.DataFrame(rows))
    assert list(t.index) == ["x", "y"]
    assert t.loc["x", "cap_score"] == pytest.approx(0.55)
    assert t.loc["x", "n_cells"] == 2 and t.loc["x", "n_seeds"] == 2


def test_cap_score_rejects_incomplete_runs():
    rows = [{"router": "x", "seed": s, "hardware": "h", "arrivals": a, "rho": 1.0, "budget_frac": 0.1,
             "acc_slo": 0.5} for s, a in ((0, "p"), (0, "b"), (1, "p"))]
    with pytest.raises(ValueError):
        cap_score(pd.DataFrame(rows))


def test_suites_are_complete_and_pinned():
    for s in SUITES.values():
        assert s["seeds"] and s["rhos"] and s["budgets"] and s["scenarios"] and s["rate_limit_snapshot"]
    for _, url, digest in traces.SOURCES.values():
        assert len(digest) == 64 and "/master/" not in url
    assert len(fetch.LLMROUTERBENCH["sha256"]) == 64 and "/resolve/main/" not in fetch.LLMROUTERBENCH["url"]


def test_router_plugin_loading():
    make = run.load_factory("examples.my_router:make")
    assert callable(make)
    with pytest.raises(ValueError):
        run.load_factory("examples.my_router")
    assert set(run.BASELINES) >= {"capacity_greedy", "static_qc_fallback", "batch_lp@2s"}
