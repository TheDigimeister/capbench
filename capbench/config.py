"""Model pool and scenario configuration.

Sourced values (pages fetched 2026-09-27) are cited inline; anything marked
PLACEHOLDER has no published figure yet. Per-call prices are not set here:
they come from LLMRouterBench's recorded cost of each call.
"""
from dataclasses import dataclass, field, replace

DATASETS = ["aime", "gpqa", "mmlupro", "livemathbench", "livecodebench", "arenahard"]


@dataclass(frozen=True)
class SelfHosted:
    name: str
    replicas: int = 1
    slots: int = 64            # concurrent sequences per replica (C_m)
    max_queue: int = 256       # model-level FIFO cap (Q_m)
    a: float = 0.02            # prefill fixed cost, s
    b: float = 1e-4            # prefill per input token, s
    t0: float = 0.012          # decode s/token with one active sequence
    alpha: float = 1.5         # tpot(n) = t0 * (1 + alpha * n / slots)
    hourly_cost: float = 2.0   # $/replica-hour, ASSUMPTION; sunk: reported, never budgeted
    kind: str = field(default="self", init=False)

    def tpot(self, n):
        return self.t0 * (1.0 + self.alpha * n / self.slots)

    def prefill(self, ptok):
        return self.a + self.b * ptok


@dataclass(frozen=True)
class API:
    name: str
    rpm: float = 120.0
    tpm: float = 400_000.0
    ttft_base: float = 0.8           # s, median visible-first-token latency
    ttft_sigma: float = 0.5          # lognormal shape, ASSUMPTION (no published spread)
    hidden_reasoning: float = 0.0    # share of output generated before the first visible token
    tpot: float = 0.01               # s/token
    kind: str = field(default="api", init=False)


# ---------------------------------------------------------------- hardware
# tpot(n) = t0 * (1 + alpha * n / 64) fitted to published 8B serving numbers.
HARDWARE = {
    # 1x H100, Llama-3.1-8B BF16, 1000 in / 1000 out: ITL 6.52 / 9.34 / 11.46 ms at
    # concurrency 1 / 25 / 50, TTFT 27 ms at concurrency 1 (NVIDIA NIM benchmarks,
    # docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html). Qwen3-8B on H100
    # with vLLM shows the same single-stream TPOT, 7.3 ms (docs.gpustack.ai).
    "h100": dict(a=0.01, b=2.5e-5, t0=0.0065, alpha=1.0),
    # DGX Spark (GB10), Llama-3.1-8B FP8, SGLang: prefill ~8,000 tok/s; decode
    # 20.5 tok/s at batch 1 and 368 tok/s total at batch 32 (lmsys.org blog, 2025-10-13).
    # Stands in for the vLLM calibration run on the Spark itself.
    "spark": dict(a=0.02, b=1.25e-4, t0=0.049, alpha=1.55),
    # DGX Spark, Qwen3-8B BF16, vLLM (nvcr.io/nvidia/vllm:26.07), max-num-seqs 64, no prefix
    # caching: measured with `capbench.calibrate fit` on 2026-10-03. Median TPOT 71.3 / 67.7 /
    # 71.5 / 85.5 / 90.0 / 95.4 ms at concurrency 1 / 4 / 16 / 32 / 48 / 64.
    "spark_vllm": dict(a=0.1347, b=2.63e-4, t0=0.0713, alpha=0.4),
}

# ---------------------------------------------------------------- API tiers
# (rpm, tpm) per model. None = no published per-minute limit: modelled as
# effectively unlimited (DeepSeek caps concurrent requests at 500-2,500 instead,
# api-docs.deepseek.com/quick_start/rate_limit).
API_LIMITS = {
    "t1": {  # lowest paid tier
        "gpt-5": (500, 500_000),               # OpenAI Tier 1, developers.openai.com/api/docs/models/gpt-5
        "gpt-5-chat": (500, 30_000),           # OpenAI Tier 1, .../models/gpt-5-chat-latest
        "gemini-2.5-pro": (150, 2_000_000),    # Gemini Tier 1 (third-party: aipromptshub.co, 2026-06-20)
        "qwen3-235b-a22b-2507": (600, 1_000_000),  # Alibaba Model Studio, all tiers (updated 2026-09-24)
        "deepseek-v3-0324": None,
    },
    "t3": {  # mid tier
        "gpt-5": (5_000, 2_000_000),           # OpenAI Tier 3
        "gpt-5-chat": (5_000, 800_000),        # OpenAI Tier 3
        "gemini-2.5-pro": (1_000, 8_000_000),  # Gemini Tier 2 RPM; TPM not published, Tier 3 value assumed
        "qwen3-235b-a22b-2507": (600, 1_000_000),
        "deepseek-v3-0324": None,
    },
}
# ASSUMPTION variant: DeepSeek-V3 is now served only through OpenRouter providers,
# whose limits are unpublished; give it Model Studio's qwen3-235b limits.
API_LIMITS["t1_ds_capped"] = {**API_LIMITS["t1"], "deepseek-v3-0324": (600, 1_000_000)}
UNLIMITED = (1e6, 1e10)

# ---------------------------------------------------------------- API latency
# Artificial Analysis first-party API measurements (artificialanalysis.ai/models/...):
# gpt-5 (medium) TTFT 31.1 s at 85.6 tok/s; gemini-2.5-pro TTFT 19.0 s at 139.1 tok/s
# (both hide reasoning before the first visible token); qwen3-235b-a22b-instruct-2507
# TTFT 2.36 s at 60.0 tok/s; deepseek-v3-0324 is benchmarked only through DeepInfra (FP4),
# TTFT 0.82 s at 78.2 tok/s (page fetched 2026-10-03). gpt-5-chat ("GPT-5 (ChatGPT)"):
# Artificial Analysis lists no speed data (PLACEHOLDER).
API_LATENCY = {
    "gpt-5": dict(ttft_base=1.0, tpot=1 / 85.6, hidden_reasoning=0.8),
    "gemini-2.5-pro": dict(ttft_base=1.0, tpot=1 / 139.1, hidden_reasoning=0.8),
    "qwen3-235b-a22b-2507": dict(ttft_base=2.36, tpot=1 / 60.0),
    "gpt-5-chat": dict(ttft_base=0.6, tpot=0.01),        # PLACEHOLDER
    "deepseek-v3-0324": dict(ttft_base=0.82, tpot=1 / 78.2),
}

SELF_HOSTED = (("Llama-3.1-8B-Instruct", 1), ("Qwen3-8B", 2), ("NVIDIA-Nemotron-Nano-9B-v2", 1))
API_MODELS = ("deepseek-v3-0324", "qwen3-235b-a22b-2507", "gpt-5-chat", "gpt-5", "gemini-2.5-pro")


def make_pool(tier="t1", hardware="h100", api_limit_scale=1.0):
    pool = [SelfHosted(name, replicas=r, **HARDWARE[hardware]) for name, r in SELF_HOSTED]
    for name in API_MODELS:
        rpm, tpm = API_LIMITS[tier][name] or UNLIMITED
        pool.append(API(name, rpm=rpm * api_limit_scale, tpm=tpm * api_limit_scale, **API_LATENCY[name]))
    return tuple(pool)


DEFAULT_POOL = make_pool()


@dataclass(frozen=True)
class Scenario:
    rho: float = 0.7                 # arrival rate / capacity (see load_basis)
    load_basis: str = "total"        # "total": self-hosted + API rate limits; "self": self-hosted only
    arrivals: str = "poisson"        # "poisson" | "burstgpt" | "azure_code" | "azure_conv"
    budget_frac: float = 0.1         # hourly budget as a share of always-best-API spend
    slo_ttft: float = 30.0           # s; a request whose first token is later scores 0
    max_attempts: int = 3            # routing attempts after 429 / queue-full / budget refusal
    retry_delay: float = 0.2         # s between a refusal and the next attempt
    budget_window: float = 600.0     # s; spend cap = hourly budget x window / 1 h, over a trailing window
    n_arrivals: int = 6000
    seed: int = 0

    def with_(self, **kw):
        return replace(self, **kw)
