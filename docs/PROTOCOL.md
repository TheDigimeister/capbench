# Protocol

## Task

A router receives queries one at a time, in arrival order, and sends each one to a model in a mixed pool, defers it, or drops it. The pool has three self-hosted 8–9B models (slots, FIFO queues, continuous-batching slowdown) and five API models (per-minute request and token buckets, per-call prices). Every query must produce its first token within the TTFT SLO (30 s) and API spend must stay under a budget. A query that is dropped, refused on every attempt, abandoned in a queue, or answered after the SLO scores 0. Otherwise it scores the model's recorded correctness on that query (from LLMRouterBench).

## Score

**CapScore** is `acc_slo` (accuracy after SLO misses), averaged with equal weight over every (scenario, ρ, budget) cell of a suite, then over seeds. It is reported with a 95% interval over seeds. Every other column in the result CSVs (SLO attainment, TTFT percentiles, spend, per-dataset accuracy, model shares) is diagnostic only.

Three reference rows bracket the scores, and none of them is a router:

| Reference | What it is |
| --- | --- |
| `oracle_unconstrained` | The best true quality per query, with no capacity, budget or SLO |
| `predictor_unconstrained` | The Track B predictor's argmax, with no capacity or budget |
| `fluid_ceiling` | An LP over the whole trace with true quality and capacity, budget and SLO relaxations. It ignores queueing and bursts, so it is approximate rather than a strict bound |

## Suite v1 (released 2026-10-03)

Defined in `capbench/suite.py`. A released suite is never edited: changes go into a new version.

| Knob | Value |
| --- | --- |
| Quality data | LLMRouterBench `bench-release.tar.gz`, HF revision `0e5af1b8`, sha256-pinned (`capbench/fetch.py`). Pool: 3,184 queries × 8 models over AIME, GPQA, MMLU-Pro, LiveMathBench, LiveCodeBench and ArenaHard, joined on (dataset, prompt) |
| Split | Fixed 50/50 per-dataset split at the prompt level (`split_seed = 0`), so no prompt is in both halves. Run seeds never change it |
| API limits | Tier 1 (`t1`) snapshot from provider pages fetched 2026-09-27 |
| Scenarios | {H100, DGX Spark} self-hosted profile × {Poisson, BurstGPT} arrivals |
| Load ρ | 0.3, 0.7, 1.0, 1.3. ρ is the arrival rate ÷ (self-hosted full-batch throughput + API rate-limited throughput) |
| Budget | 0.1× and 0.5× the spend of sending everything to the most accurate API model. It is enforced as a hard cap over a trailing 10-minute window |
| Run length | 30 simulated minutes per cell |
| Seeds | 0–4. A seed sets arrival sampling, the trace window, the query draw and the predictor's KMeans initialisation |

## Tracks

- **Track B (shared predictor)**: routers get `ctx.P`, a cluster predictor of quality, cost and output tokens. It is fit on the train split and indexed by `Query.row`, so routers compete on allocation under load.
- **Track A (bring your own predictor)**: routers learn from `TrainData` in `fit()` and see only the prompt text at route time. They must not use `Query.row` or `ctx.P`; the scalar calibrations `ctx.lam` and `ctx.tau` (fit on train from the shared predictor) are allowed.

## Rules

1. Learn only from the train split, through `Router.fit(TrainData)` or `ctx`. The test half is public because all inputs are public, so this rule rests on the honour system. Tune hyperparameters on a split of the train half or on an ad-hoc grid, never on suite results.
2. Use only what `route()` and `on_event()` receive at run time: the query, the observable `SystemState`, and refusal and completion events. Do not read simulator internals.
3. Use the same configuration in every cell. A router may adapt online to load and budget signals, but must not branch on the scenario name or seed.
4. Report the full suite (all scenarios and seeds). `capbench.score` refuses incomplete runs.

## Simulator

`capbench/sim.py` is a discrete-event simulator. Self-hosted models use processor-sharing decode with `tpot(n) = t0·(1 + α·n/slots)`, prefill `a + b·ptok`, a model-level FIFO queue, and abandonment at the deadline. API models use token buckets for RPM and TPM, lognormal TTFT, and hidden reasoning tokens before the first visible token. Budget is enforced as a windowed spend refusal. Refusals (429, queue full, budget) give the router up to 3 attempts with a 0.2 s retry delay.

### Calibration status

| Parameter | Source | Status |
| --- | --- | --- |
| API rate limits, Tier 1 | OpenAI and Alibaba Model Studio docs; Gemini is third-party sourced | Sourced (2026-09-27). DeepSeek publishes no per-minute limit and is treated as unlimited overflow |
| API latency: gpt-5, gemini-2.5-pro, qwen3-235b | Artificial Analysis | Sourced |
| API latency: deepseek-v3-0324 | Artificial Analysis, DeepInfra (FP4) endpoint | Sourced (2026-10-03) |
| API latency: gpt-5-chat | None published | **Placeholder**: 0.6 s TTFT, 100 tok/s |
| API TTFT spread (lognormal σ = 0.5) | None published | **Assumption** |
| Self-hosted H100 profile | NVIDIA NIM Llama-3.1-8B benchmarks | Sourced. The fit reproduces ITL at concurrency 25 and 50 |
| Self-hosted Spark profile | LMSYS DGX Spark SGLang numbers | Sourced. A live vLLM fit (`spark_vllm`, 2026-10-03) failed validation; see CALIBRATION.md |
| Slots (64) and queue cap (256) per replica | Chosen | **Assumption** |
| Self-hosted $/replica-hour ($2) | None | **Assumption**. It is reported only; it never enters the budget or the score |
| Per-call API prices | LLMRouterBench recorded cost | Sourced |

**Validation against live vLLM failed** (2026-10-03, DGX Spark, Qwen3-8B). The simulator overstates self-hosted capacity by about 1.8× at realistic prompt lengths, because it does not model prefill contention or the effect of context length on decode. Suite v1 scores are therefore provisional. See [CALIBRATION.md](CALIBRATION.md); fixing this is the main item for suite v2. The models gpt-5, gpt-5-chat and deepseek-v3-0324 have since been retired or moved by their providers, so the numbers above describe the 2025–26 APIs that LLMRouterBench recorded.
