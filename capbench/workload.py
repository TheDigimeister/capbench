"""Pool data, train/test split, arrival process and the load/budget calibration."""
from dataclasses import dataclass

import numpy as np

from . import pool as pool_data
from . import source, traces
from .config import DATASETS
from .sim import HOUR, Workload


@dataclass
class Bench:
    d: dict              # pool arrays (Q, C, Ptok, Ctok, dataset_id, prompt, ...)
    pool: tuple          # model specs, aligned with d["models"]
    train: np.ndarray
    test: np.ndarray
    ctok: np.ndarray     # imputed output tokens (zero-token failures filled)


SPLIT_SEED = 0      # the benchmark's fixed train/test split; run seeds never change it
TRAIN_RATIO = 0.5


def load_bench(pool, split_seed=SPLIT_SEED, train_ratio=TRAIN_RATIO):
    names = [p.name for p in pool]
    d = pool_data.load(names, DATASETS)
    ctok = source.impute_tokens(d["Ctok"], d["dataset_id"])
    train, test = source.split_by_dataset_prompt(d["dataset_id"], d["record_index"], d["prompt"],
                                                 train_ratio, split_seed)
    return Bench(d, tuple(pool), train, test, ctok)


def self_hosted_throughput(b: Bench):
    """Requests/s the self-hosted tier sustains at full batch, if each self-hosted
    model received its share of traffic in proportion to its own capacity."""
    ptok = b.d["Ptok"][b.train].mean()
    mu = 0.0
    for j, p in enumerate(b.pool):
        if p.kind == "self":
            mu += p.replicas * p.saturated(ptok, b.ctok[b.train, j].mean())[0]
    return mu


def api_throughput(b: Bench):
    """Requests/s the API tier sustains under its rate limits: per model, the
    tighter of RPM and TPM (at that model's mean tokens per request). Models
    without published limits (UNLIMITED) are left out: they are overflow, not
    capacity, and would otherwise swamp the denominator."""
    from .config import UNLIMITED
    tok = b.d["Ptok"][b.train] + b.ctok[b.train]
    mu = 0.0
    for j, p in enumerate(b.pool):
        if p.kind == "api" and p.rpm < UNLIMITED[0]:
            mu += min(p.rpm, p.tpm / tok[:, j].mean()) / 60.0
    return mu


def capacity(b: Bench, basis="total"):
    """Denominator of rho: "total" = self-hosted + API rate-limited throughput,
    "self" = self-hosted only (the design doc's original definition)."""
    if basis == "self":
        return self_hosted_throughput(b)
    if basis == "total":
        return self_hosted_throughput(b) + api_throughput(b)
    raise ValueError(basis)


def arrival_rate(b: Bench, rho, basis="total"):
    return rho * capacity(b, basis)


def budget_per_hour(b: Bench, rate, frac):
    """frac x the hourly spend of sending everything to the most accurate API model."""
    api = [j for j, p in enumerate(b.pool) if p.kind == "api"]
    best = max(api, key=lambda j: b.d["Q"][b.train, j].mean())
    return frac * rate * HOUR * b.d["C"][b.train, best].mean()


def make_workload(b: Bench, rate, n, seed, arrivals="poisson"):
    rng = np.random.default_rng([seed, 11])
    if arrivals == "poisson":
        t = np.cumsum(rng.exponential(1.0 / rate, n))
    else:
        t = traces.arrivals(arrivals, n, rate, seed)
    rows = rng.choice(b.test, size=n, replace=True)
    d = b.d
    return Workload(t=t, row=rows, dataset=d["dataset_id"][rows], prompt=d["prompt"][rows],
                    ptok=np.maximum(d["Ptok"][rows], 1.0), ctok=b.ctok[rows],
                    Q=d["Q"][rows].astype(float), C=d["C"][rows].astype(float))


def make_train_data(b: Bench):
    """Track A's view of the train split."""
    from .api import TrainData
    d, tr = b.d, b.train
    return TrainData(tuple(d["models"]), d["prompt"][tr], d["dataset_id"][tr], d["Q"][tr].astype(float),
                     d["C"][tr].astype(float), d["Ptok"][tr].astype(float), b.ctok[tr])
