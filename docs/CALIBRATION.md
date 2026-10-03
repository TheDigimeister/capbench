# Simulator calibration log

## 2026-10-03 (second window): contention model. Validation passed.

**Model (v0.3, `SelfHosted.serial_prefill`):** each replica has one FIFO prefill server, busy for `b·ptok` per request. Prefills are batched, so `a` adds latency but does not occupy the server. Decode is processor sharing over decoding sequences only, with per-token time

`tpot = t0·(1 + α·n/slots) + κ·(context tokens held by decoding sequences)/1000`

multiplied by `(1 + β)` while the prefill server is busy. Profiles without `serial_prefill` behave exactly as in v0.1/v0.2, so suite v1 reproduces bit for bit.

**Fit** (`capbench.calibrate fit`, same server and flags as below): single-request prefills plus 9 closed-loop experiments. The simulator replays each experiment and least squares on log median TTFT and log median TPOT gives the parameters. Raw rows are in `outputs/calibration/measure_fit.csv`.

| Experiment (n × in/out) | TTFT vLLM | TTFT sim | TPOT vLLM | TPOT sim |
| --- | ---: | ---: | ---: | ---: |
| 1 × 256/256 | 0.11 s | 0.12 s | 72.3 ms | 68.7 ms |
| 16 × 256/256 | 0.76 s | 0.64 s | 71.9 ms | 74.3 ms |
| 64 × 256/256 | 2.36 s | 2.31 s | 95.1 ms | 92.6 ms |
| 1 × 2048/128 | 0.67 s | 0.60 s | 73.3 ms | 70.2 ms |
| 16 × 1024/128 | 2.34 s | 2.41 s | 86.5 ms | 94.2 ms |
| 32 × 1024/256 | 4.34 s | 4.64 s | 105.8 ms | 107.9 ms |
| 32 × 2048/128 | 8.63 s | 9.23 s | 165.8 ms | 169.7 ms |
| 64 × 620/116 | 5.09 s | 5.52 s | 131.7 ms | 136.1 ms |
| 64 × 2048/128 | 18.05 s | 18.14 s | 289.9 ms | 269.0 ms |

Fitted profile (`HARDWARE["spark_v2"]`): `a = 0.0475 s`, `b = 2.718e-4 s/token`, `t0 = 68.2 ms`, `α = 0`, `κ = 7.0e-4 s per 1k context tokens`, `β = 5.0`.

**Identification caveat:** β sits at the fit's upper bound. With the bound raised to 50, the fit reaches β ≈ 17 with the same error (RMS log error 0.066 vs 0.067), but a lower steady-state capacity (1.3 vs 2.0 req/s at BurstGPT's mean lengths). Closed-loop bursts do not separate the two. β = 5 passes the open-loop validation below; β ≈ 17 was not tested. A future sweep should include steady open-loop load at several rates.

**Validation** (`capbench.calibrate validate --profile spark_v2 --rho 0.7`): a 10-minute BurstGPT window at 1.42 req/s, 853 requests. Raw rows are in `outputs/calibration/validate_spark_v2_rho0.7.csv`.

| TTFT | vLLM | Simulator | Error |
| --- | ---: | ---: | ---: |
| p50 | 0.49 s | 0.30 s | −38% (−0.19 s) |
| p95 | 16.3 s | 15.3 s | **−6.1%** |
| p99 | 23.7 s | 21.6 s | −9.0% |

The p95 target (within 15%) is met. The simulator is slightly optimistic throughout, and most so at the median, where the absolute gap is small. Suite v2 uses this profile.

## 2026-10-03: DGX Spark, Qwen3-8B, vLLM. Validation failed.

**Setup:** NVIDIA DGX Spark (GB10), `nvcr.io/nvidia/vllm:26.07-py3`, `vllm serve Qwen/Qwen3-8B --max-num-seqs 64 --max-model-len 8192 --gpu-memory-utilization 0.6 --no-enable-prefix-caching`, BF16. The client ran over the network from a separate machine.

### Fit (first version of `capbench.calibrate fit`, v1 model)

| Concurrency | 1 | 4 | 16 | 32 | 48 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Median TPOT, ms (256 in / 256 out) | 71.3 | 67.7 | 71.5 | 85.5 | 90.0 | 95.4 |

Fitted profile (`HARDWARE["spark_vllm"]`): `a = 0.135 s`, `b = 2.63e-4 s/token`, `t0 = 71.3 ms`, `α = 0.40`. The published-numbers profile used by suite v1 (`spark`: LMSYS, Llama-3.1-8B FP8, SGLang) has `t0 = 49 ms` and `α = 1.55`.

### Validation (`capbench.calibrate validate --profile spark_vllm --rho 0.7`)

This replayed a 10-minute BurstGPT window: request and response lengths drawn from BurstGPT and clipped to 16–4096 in and 16–1024 out (means 621 / 117). The rate was ρ = 0.7 of the simulator's capacity for this profile, 3.75 req/s.

| TTFT | vLLM | Simulator | Error |
| --- | ---: | ---: | ---: |
| p95 | 399 s | 61 s | −85% |
| p99 | 436 s | 76 s | −83% |

The live server could not keep up, so the 10-minute replay took 18.7 minutes. A closed-loop check at the same mean lengths (64 concurrent requests, 620 in / 116 out) showed why:

| | vLLM | Simulator (`spark_vllm`) |
| --- | ---: | ---: |
| Throughput at 64 in flight | 2.9 req/s | 5.4 req/s |
| Median TPOT | 136 ms | 95 ms |
| Median TTFT | 5.3 s | 0.30 s |

### Diagnosis

The self-hosted model overstates capacity by about 1.8× at realistic prompt lengths, for two reasons:

1. **Prefill contention.** The simulator gives each request its own prefill time (`a + b·ptok`), independent of other requests. In vLLM, concurrent prefills share the GPU with each other and with decode (chunked prefill), so TTFT grows with the number of requests being admitted.
2. **Context-dependent decode.** TPOT was fitted with 256-token prompts. With longer contexts, each decode step reads more KV cache: 136 ms against 95 ms at 64 in flight.

The H100 profile was fitted the same way from published numbers and likely shares both biases.

### Consequences

- Suite v1 scores and rankings are **provisional**. At a given labelled ρ, the real self-hosted tier is more loaded than the simulator assumes. That favours routers that lean on self-hosted capacity.
- Suite v1 is left unchanged, as released suites are never edited. The measured profile is recorded as `spark_vllm` and is not used by any suite yet.
- Next: model prefill as shared GPU work, make TPOT depend on context, re-fit from these measurements, re-validate, then release suite v2.
