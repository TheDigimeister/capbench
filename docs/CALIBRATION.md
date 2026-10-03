# Simulator calibration log

## 2026-10-03: DGX Spark, Qwen3-8B, vLLM. Validation failed.

**Setup:** NVIDIA DGX Spark (GB10), `nvcr.io/nvidia/vllm:26.07-py3`, `vllm serve Qwen/Qwen3-8B --max-num-seqs 64 --max-model-len 8192 --gpu-memory-utilization 0.6 --no-enable-prefix-caching`, BF16. The client ran over the network from a separate machine.

### Fit (`capbench.calibrate fit`)

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
