"""Fit a self-hosted latency profile against a live vLLM server, and check the
simulator against it (design doc: "Simulator validation").

Start vLLM on the target box first, e.g. on the DGX Spark:
    vllm serve Qwen/Qwen3-8B --max-num-seqs 64 --port 8000

    python -m capbench.calibrate fit --url http://<vllm-host>:8000 --model Qwen/Qwen3-8B
        -> sweeps concurrency, prints a HARDWARE entry {a, b, t0, alpha} for config.py
    python -m capbench.calibrate validate --url ... --model ... --profile spark --rho 0.7
        -> replays a 10-minute BurstGPT slice through vLLM and through the simulator;
           reports p50/p95/p99 TTFT for both (target: p95 within ~15%)

Needs the `calibrate` extra (httpx).
"""
import argparse
import asyncio
import json
import time

import numpy as np
from pathlib import Path

PROMPT_WORD = "benchmark "  # ~1 token per repeat for common tokenizers


async def _one(client, url, model, ptok, ctok, t_send, t0, out):
    await asyncio.sleep(max(0.0, t_send - (time.perf_counter() - t0)))
    body = {"model": model, "prompt": PROMPT_WORD * int(ptok), "max_tokens": int(ctok),
            "min_tokens": int(ctok), "ignore_eos": True, "stream": True, "temperature": 0.0}
    start = time.perf_counter()
    first = None
    n = 0
    async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            if json.loads(line[6:])["choices"][0].get("text"):
                n += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    out.append({"sched": t_send, "send": start - t0, "ttft": first - start, "e2e": end - start,
                "tpot": (end - first) / max(n - 1, 1), "ptok": ptok, "ctok": ctok})


async def _run(url, model, jobs):
    import httpx
    out = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        await asyncio.gather(*(_one(client, url, model, p, c, t, t0, out) for t, p, c in jobs))
    return out


# Closed-loop experiments for the contention model: (concurrency, prompt tokens, output tokens).
DESIGN = ((1, 256, 256), (16, 256, 256), (64, 256, 256), (1, 2048, 128), (16, 1024, 128),
          (32, 2048, 128), (64, 620, 116), (64, 2048, 128), (32, 1024, 256))
OUT = Path(__file__).resolve().parent.parent / "outputs" / "calibration"


def _save(rows, name):
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / name, index=False)
    print("wrote", f"outputs/calibration/{name}")


def measure(url, model, tag, design=DESIGN):
    """Single-request prefills, then each closed-loop experiment; raw rows saved per request."""
    rows = []
    for k, r in enumerate(asyncio.run(_run(url, model, [(i * 2.0, p, 1) for i, p in
                                                       enumerate([128, 512, 1024, 2048, 4096])]))):
        rows.append(r | {"exp": "prefill", "n": 1})
    for n, p, c in design:
        for r in asyncio.run(_run(url, model, [(0.0, p, c)] * n)):
            rows.append(r | {"exp": f"{n}x{p}/{c}", "n": n})
    _save(rows, f"measure_{tag}.csv")
    return rows


def simulate_closed_loop(spec, n, ptok, ctok):
    """The simulator's view of n simultaneous (ptok, ctok) requests on one replica."""
    from .api import Router
    from .config import Scenario
    from .sim import Simulator, Workload

    class One(Router):
        def route(self, q, s):
            return 0

    w = Workload(t=np.arange(n) * 1e-4, row=np.arange(n), dataset=np.array(["x"] * n),
                 prompt=np.array([""] * n, object), ptok=np.full((n, 1), float(ptok)),
                 ctok=np.full((n, 1), float(ctok)), Q=np.ones((n, 1)), C=np.zeros((n, 1)))
    r = Simulator((spec,), Scenario(slo_ttft=1e9), 1e9).run(w, One())
    return float(np.median(r.ttft)), float(np.median((r.e2e - r.ttft) / max(ctok - 1, 1)))


def fit_rows(rows, slots=64):
    """Fit {a, b, t0, alpha, kappa, beta} of the contention model to measured rows:
    least squares on log median TTFT and log median TPOT per experiment."""
    import pandas as pd
    from scipy.optimize import least_squares

    from .config import SelfHosted
    df = pd.DataFrame(rows)
    pre = df[df["exp"] == "prefill"]
    a0, b0 = np.linalg.lstsq(np.c_[np.ones(len(pre)), pre["ptok"]], pre["ttft"], rcond=None)[0]
    exps = []
    for name, g in df[df["exp"] != "prefill"].groupby("exp"):
        exps.append((int(g["n"].iloc[0]), float(g["ptok"].iloc[0]), float(g["ctok"].iloc[0]),
                     float(g["ttft"].median()), float(g["tpot"].median())))

    def spec(x):
        a, b, t0, alpha, kappa, beta = x
        return SelfHosted("fit", slots=slots, max_queue=10**6, a=a, b=b, t0=t0, alpha=alpha,
                          kappa=kappa, beta=beta, serial_prefill=True)

    def resid(x):
        s, out = spec(x), []
        for n, p, c, ttft, tpot in exps:
            st, sp = simulate_closed_loop(s, n, p, c)
            out += [np.log(st / ttft), np.log(sp / tpot)]
        return np.array(out)

    x0 = [max(a0, 0.01), max(b0, 1e-5), 0.07, 0.4, 0.0, 0.0]
    res = least_squares(resid, x0, bounds=([0, 1e-6, 1e-3, 0, 0, 0], [2, 1e-2, 1, 10, 1, 5]),
                        x_scale=[0.1, 1e-4, 0.01, 0.5, 0.01, 0.5])
    prof = dict(zip(["a", "b", "t0", "alpha", "kappa", "beta"], (float(f"{v:.4g}") for v in res.x)))
    prof["serial_prefill"] = True
    print("experiment            TTFT meas/sim (s)     TPOT meas/sim (ms)")
    s = spec(res.x)
    for n, p, c, ttft, tpot in exps:
        st, sp = simulate_closed_loop(s, n, p, c)
        print(f"{n:>3}x{int(p)}/{int(c):<10} {ttft:8.2f} {st:8.2f}      {1e3 * tpot:8.1f} {1e3 * sp:8.1f}")
    print("HARDWARE entry:", prof)
    return prof


def fit(url, model, slots=64, tag="fit"):
    return fit_rows(measure(url, model, tag), slots)


def validate(url, model, profile, rho=0.7, minutes=10, seed=0, slots=64):
    """Replay one BurstGPT window through vLLM and through one simulated replica."""
    from . import traces
    from .api import Router
    from .config import HARDWARE, Scenario, SelfHosted
    from .sim import Simulator, Workload

    spec = SelfHosted(model, replicas=1, slots=slots, max_queue=10**6, **HARDWARE[profile])
    rng = np.random.default_rng(seed)
    # token lengths from BurstGPT itself, capped to keep the run short
    import pandas as pd
    df = pd.read_csv(traces.RAW / traces.SOURCES["burstgpt"][0],
                     usecols=["Request tokens", "Response tokens"])
    ptok = np.clip(df["Request tokens"].to_numpy(), 16, 4096)
    ctok = np.clip(df["Response tokens"].to_numpy(), 16, 1024)
    rate = rho * spec.saturated(ptok.mean(), ctok.mean())[0]
    n = int(rate * minutes * 60)
    t = traces.arrivals("burstgpt", n, rate, seed)
    pick = rng.integers(0, len(ptok), n)
    real = asyncio.run(_run(url, model, [(ti, ptok[k], ctok[k]) for ti, k in zip(t, pick)]))

    class One(Router):
        def route(self, q, s):
            return 0

    w = Workload(t=t, row=np.arange(n), dataset=np.array(["x"] * n), prompt=np.array([""] * n, object),
                 ptok=ptok[pick][:, None].astype(float), ctok=ctok[pick][:, None].astype(float),
                 Q=np.ones((n, 1)), C=np.zeros((n, 1)))
    sim = Simulator((spec,), Scenario(slo_ttft=1e9), 1e9).run(w, One())
    q = [50, 95, 99]
    # TTFT from the scheduled arrival, so client-side send lag counts like queueing
    _save(real, f"validate_{profile}_rho{rho}.csv")
    real_ttft = np.percentile([r["send"] - r["sched"] + r["ttft"] for r in real], q)
    sim_ttft = np.percentile(sim.ttft, q)
    print(f"{n} requests at {rate:.2f}/s (rho={rho})")
    for k, rv, sv in zip(q, real_ttft, sim_ttft):
        print(f"TTFT p{k}: vLLM {rv:.3f} s   sim {sv:.3f} s   error {100 * (sv - rv) / rv:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "validate"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default="spark")
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--slots", type=int, default=64)
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.url, args.model, args.slots)
    else:
        validate(args.url, args.model, args.profile, args.rho, slots=args.slots)


if __name__ == "__main__":
    main()
