"""Minimal custom router: run it with

    python -m capbench.run --suite v2 --router examples.my_router:make

`make(ctx)` receives a capbench.run.RouterContext and returns a Router.
This one sends a query to the model with the highest predicted quality among
those that can take it right now (a free self-hosted slot, or API headroom
and budget), and drops it otherwise.
"""
import numpy as np

from capbench.api import DROP, Router


class FirstFeasible(Router):
    name = "example:first_feasible"

    def __init__(self, ctx):
        self.Qhat = ctx.P.Qhat

    def route(self, q, s):
        for j in np.argsort(-self.Qhat[q.row]):
            m = s.models[j]
            if m.kind == "self" and m.free_slots > 0:
                return int(j)
            if m.kind == "api" and m.rpm_headroom >= 1 and s.budget_remaining > 0:
                return int(j)
        return DROP


def make(ctx):
    return FirstFeasible(ctx)
