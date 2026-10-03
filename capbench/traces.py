"""Arrival times from public LLM serving traces.

A trace supplies only timing. For n arrivals at mean rate `rate`, take n
consecutive trace requests from a seed-dependent offset, shift them to start
at 0 and rescale time linearly so the window's mean rate equals `rate`:
the burst shape (inter-arrival pattern) is kept, the level is set by rho.

Raw files live in traces/raw/ (gitignored); `fetch()` downloads them.
"""
import hashlib
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "traces" / "raw"

AZURE = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/790921015d50dd6aae7f7e47f39ba0e235ad6b08/data/"
SOURCES = {  # name: (file, pinned URL, sha256). All three traces are CC-BY-4.0.
    # BurstGPT v2.0 (Azure OpenAI GPT usage, 1-second timestamps), successful requests only
    "burstgpt": ("BurstGPT_without_fails_1.csv",
                 "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv",
                 "a4d068a7113ec0290e74063a1b3447dc6001a30e4298eb313581b71006dda1f4"),
    # Azure LLM inference traces 2023 (one hour each, sub-second timestamps)
    "azure_code": ("AzureLLMInferenceTrace_code.csv", AZURE + "AzureLLMInferenceTrace_code.csv",
                   "54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6"),
    "azure_conv": ("AzureLLMInferenceTrace_conv.csv", AZURE + "AzureLLMInferenceTrace_conv.csv",
                   "2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(names=tuple(SOURCES)):
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names:
        fname, url, digest = SOURCES[name]
        path = RAW / fname
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise RuntimeError(f"{path}: sha256 mismatch (expected {digest})")


@lru_cache(maxsize=None)
def timestamps(name):
    """Sorted arrival times in seconds from the trace start."""
    fname = SOURCES[name][0]
    path = RAW / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m capbench.traces` to download")
    if name == "burstgpt":
        t = pd.read_csv(path, usecols=["Timestamp"])["Timestamp"].to_numpy(float)
        # 1-second resolution: spread same-second requests evenly within the second
        t = np.sort(t)
        _, first, counts = np.unique(t, return_index=True, return_counts=True)
        rank = np.arange(len(t)) - np.repeat(first, counts)
        t = t + rank / np.repeat(counts, counts)
    else:
        ts = pd.to_datetime(pd.read_csv(path, usecols=["TIMESTAMP"])["TIMESTAMP"])
        t = np.sort((ts - ts.min()).dt.total_seconds().to_numpy())
    return t - t[0]


def arrivals(name, n, rate, seed):
    t = timestamps(name)
    if n > len(t) - 1:
        raise ValueError(f"trace {name} has {len(t)} requests; asked for {n}")
    rng = np.random.default_rng([seed, 13])
    start = int(rng.integers(0, len(t) - n))
    w = t[start:start + n + 1] - t[start]
    span = w[-1]  # n inter-arrival gaps
    if span <= 0:
        raise ValueError(f"degenerate window in trace {name}")
    return w[1:] * (n / rate) / span


def burstiness(name, n=6000, seeds=range(5)):
    """Coefficient of variation of inter-arrival gaps (Poisson = 1)."""
    cvs = []
    for s in seeds:
        g = np.diff(np.concatenate([[0.0], arrivals(name, n, 1.0, s)]))
        cvs.append(g.std() / g.mean())
    return float(np.mean(cvs))


if __name__ == "__main__":
    fetch()
    for name in SOURCES:
        t = timestamps(name)
        print(f"{name}: {len(t)} requests over {t[-1] / 3600:.1f} h, "
              f"inter-arrival CV (6000-request windows) = {burstiness(name):.2f}")
