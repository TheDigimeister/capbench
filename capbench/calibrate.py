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


def fit(url, model, slots=64, levels=(1, 4, 16, 32, 48, 64), ctok=256):
    """Closed-loop concurrency sweep for decode; single requests for prefill."""
    pre = asyncio.run(_run(url, model, [(i * 2.0, p, 1) for i, p in enumerate([128, 512, 1024, 2048, 4096])]))
    P = np.array([[1.0, r["ptok"]] for r in pre])
    a, b = np.linalg.lstsq(P, np.array([r["ttft"] for r in pre]), rcond=None)[0]
    tp = []
    for n in levels:
        res = asyncio.run(_run(url, model, [(0.0, 256, ctok)] * n))
        tp.append((n, float(np.median([r["tpot"] for r in res]))))
    n_, y = np.array(tp).T
    t0 = y[0] if n_[0] == 1 else np.polyfit(n_, y, 1)[1]
    alpha = max(np.polyfit(n_ / slots, y / t0 - 1.0, 1)[0], 0.0)
    prof = {"a": round(float(max(a, 0.0)), 4), "b": float(f"{max(b, 1e-7):.3g}"),
            "t0": round(float(t0), 5), "alpha": round(float(alpha), 3)}
    print("tpot by concurrency:", [(int(n), round(t * 1e3, 2)) for n, t in tp], "ms")
    print("HARDWARE entry:", prof)
    return prof


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
    mean_svc = spec.prefill(ptok.mean()) + ctok.mean() * spec.tpot(slots)
    rate = rho * slots / mean_svc
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
