# capbench: a capacity-aware LLM routing benchmark

Most LLM routing benchmarks score a router as if every model were always available. **capbench** scores routers under serving limits instead. Self-hosted models have finite slots and queues, API models have rate limits, spend has a hard cap, and a request whose first token misses a 30 s TTFT SLO scores zero. Routers run in a discrete-event simulator over real per-query outcomes from [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench), with arrivals drawn from Poisson processes and the public [BurstGPT](https://github.com/HPMLL/BurstGPT) trace.

The primary score is **CapScore**: accuracy after SLO misses, averaged over a fixed, versioned suite of load, budget and hardware scenarios. See [docs/PROTOCOL.md](docs/PROTOCOL.md).

> **Status: v0.3, suite v2.** Suite v2 uses a self-hosted profile validated against live vLLM on a DGX Spark: TTFT p95 within 6.1% of a live replay ([docs/CALIBRATION.md](docs/CALIBRATION.md)). The H100 profile is not validated yet, so it appears only in suite v1. One API latency (gpt-5-chat) is still a placeholder.

## Leaderboard (suite v2)

| Router | Track | CapScore | 95% CI |
| --- | --- | ---: | ---: |
| *oracle, unconstrained (reference)* | – | 0.928 | ±0.001 |
| *fluid LP ceiling (reference)* | – | 0.879 | ±0.001 |
| *predictor argmax, unconstrained (reference)* | B | 0.813 | ±0.002 |
| `batch_lp@2s` | B | **0.678** | ±0.011 |
| `capacity_greedy` | B | 0.670 | ±0.011 |
| `A:knn_capacity_greedy` | A | 0.621 | ±0.008 |
| `static_qc_fallback` | B | 0.562 | ±0.018 |
| `static_qc` | B | 0.522 | ±0.021 |
| `shortest_queue` | B | 0.469 | ±0.015 |
| `always_best_api` | – | 0.250 | ±0.012 |
| `random` | – | 0.231 | ±0.004 |
| `threshold` | B | 0.105 | ±0.007 |
| `always_best_self` | – | 0.007 | ±0.000 |

CapScore is accuracy after SLO misses, averaged over the 16 cells of suite v2 (calibrated DGX Spark self-hosting × Poisson and BurstGPT arrivals × 4 load levels × 2 budgets), then over 5 seeds. Suite v1 results are kept in [docs/RESULTS.md](docs/RESULTS.md).

All rows are the built-in baselines (`capbench/baselines.py`). Plots for every cell are in `outputs/v2.png`, and the analysis is in [docs/RESULTS.md](docs/RESULTS.md). To add your router, see [docs/SUBMITTING.md](docs/SUBMITTING.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,embed]'
.venv/bin/python -m pytest -q                 # simulator and protocol tests (no data needed)
.venv/bin/python -m capbench.fetch            # LLMRouterBench results (1.3 GB, pinned) + arrival traces (53 MB)
.venv/bin/python -m capbench.run --suite v2   # all baselines; the first run embeds prompts on CPU (~25 min, once)
```

Run your own router: write `make(ctx) -> Router` (see [examples/my_router.py](examples/my_router.py)), then:

```bash
.venv/bin/python -m capbench.run --suite v2 --router examples.my_router:make --only-custom --tag v2_mine
.venv/bin/python -m capbench.score outputs/v2_mine.csv --markdown
```

`--scenario` and `--suite-seeds` run a subset of scenarios or seeds, so a suite can be split across processes. `capbench.score` and `capbench.plot` accept several CSVs. Without `--suite`, `capbench.run` runs an ad-hoc grid (`--tier`, `--hardware`, `--arrivals`, `--rhos`, `--budgets`, `--api-limit-scale`, …) for exploration.

## Layout

| File | Role |
| --- | --- |
| `capbench/api.py` | Router interface: `route(query, state) -> model index \| Defer \| DROP`, `on_event`, `fit(TrainData)` |
| `capbench/run.py` | Runner, `RouterContext`, baseline registry, `--router module:attr` plugins |
| `capbench/suite.py` | Versioned suites and CapScore |
| `capbench/sim.py` | Discrete-event simulator (processor-sharing decode, token buckets, windowed budget, SLO abandonment) |
| `capbench/config.py` | Model pool, API tiers and latency, hardware profiles, with citations |
| `capbench/baselines.py` | random, fixed, threshold, static_qc (+ fallback), shortest_queue, capacity_greedy, batch_lp, and the Track A kNN example |
| `capbench/workload.py`, `pool.py`, `source.py` | Data loading, the fixed train/test split, arrivals, and load and budget calibration |
| `capbench/predictor.py` | Track B shared predictor (KMeans on bge embeddings, fit on train) |
| `capbench/metrics.py`, `lp.py` | Scores, unconstrained oracle, fluid LP ceiling |
| `capbench/traces.py`, `fetch.py` | Pinned, checksummed downloads |
| `capbench/calibrate.py` | Fits a self-hosted profile from a live vLLM server and validates the simulator against it |

## Data and licences

Code: MIT ([LICENSE](LICENSE)). This repository contains no third-party data; `capbench.fetch` downloads it:

| Data | Licence | Use |
| --- | --- | --- |
| [LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) results ([HF](https://huggingface.co/datasets/NPULH/LLMRouterBench)) | MIT (per its README) | Per-query correctness, cost and tokens. Its underlying datasets have their own terms: GPQA, for example, asks that its examples not be posted online, so do not publish prompts from the pool |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) v2.0 | CC-BY-4.0 | Arrival timestamps |
| [Azure LLM inference traces 2023](https://github.com/Azure/AzurePublicDataset) | CC-BY-4.0 | Arrival timestamps (ad-hoc grids only) |

API rate limits and latencies are cited inline in `capbench/config.py`.

## Citation

If you use capbench, please cite this repository and LLMRouterBench.
