"""Router-facing types. Routers see only what a real deployment could observe."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    qid: int          # arrival index in this run
    row: int          # row of the test pool (keys the Track B predictions)
    dataset: str
    prompt: str
    ptok: float       # input tokens (known at arrival)
    arrival: float    # s
    deadline: float   # arrival + TTFT SLO
    attempt: int      # 0 on first routing, +1 after each refusal


@dataclass(frozen=True)
class ModelState:
    name: str
    kind: str                 # "self" | "api"
    free_slots: int           # self-hosted: idle slots across replicas (api: -1)
    queue_len: int            # self-hosted queue length (api: 0)
    queue_room: int           # self-hosted: max_queue - queue_len (api: -1)
    est_wait: float           # s until a newly routed request would start (EWMA-based)
    svc_ewma: float           # s, recent mean slot-holding time per request (self; api: 0)
    ttft_ewma: float          # s, recent observed TTFT on this model
    rpm_headroom: float       # api: requests available now (self: inf)
    tpm_headroom: float       # api: tokens available now (self: inf)


@dataclass(frozen=True)
class SystemState:
    t: float
    models: tuple             # ModelState per pool model, pool order
    budget_remaining: float   # $ left in the trailing budget window


@dataclass(frozen=True)
class Defer:
    until: float


class Drop:
    pass


DROP = Drop()


@dataclass(frozen=True)
class Event:
    kind: str       # "done" | "429" | "queue_full" | "budget" | "abandoned"
    qid: int
    model: int      # pool index (-1 if none)
    t: float
    ttft: float = float("nan")


@dataclass(frozen=True)
class TrainData:
    """Track A: the train split, handed to Router.fit() before any run.
    Arrays are row-aligned; columns follow `models` (pool order)."""
    models: tuple
    prompt: object      # (n,) prompt text
    dataset: object     # (n,)
    Q: object           # (n, m) quality
    C: object           # (n, m) $ per call (self-hosted columns are sunk cost: ignore)
    ptok: object        # (n, m) input tokens
    ctok: object        # (n, m) output tokens


class Router:
    """route() returns a pool index, Defer(until) or DROP. A refusal triggers
    on_event() and, if attempts remain, another route() call with attempt+1."""
    name = "router"

    def fit(self, train: TrainData):
        """Track A only: learn from the train split. Track B routers get the shared
        predictions at construction instead and leave this a no-op."""

    def reset(self, pool, sim_info):
        """Called once per run. sim_info: dict of public facts (prices, slots, SLO)."""
        self.pool = pool
        self.info = sim_info

    def route(self, q: Query, s: SystemState):
        raise NotImplementedError

    def on_event(self, e: Event):
        pass
