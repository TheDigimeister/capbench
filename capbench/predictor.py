"""Track B's shared quality predictor: the vendored cluster predictor
(KMeans on bge embeddings, top-4 clusters), fit on the train split only.

Returns per test-pool row: Qhat (quality), Chat ($), That (output tokens).
"""
import hashlib

import numpy as np

from . import source


class Predictions:
    def __init__(self, Qhat, Chat, That):
        self.Qhat, self.Chat, self.That = Qhat, Chat, That


def shared_predictor(bench, seed):
    d = bench.d
    E = source.embeddings(d["prompt"])
    tag = "cap_" + hashlib.sha1(",".join(d["models"]).encode()).hexdigest()[:8]
    cp = source.ClusterPredictor(E, d["Q"], d["C"], bench.ctok, bench.train, seed, tag)
    rows = np.arange(len(d["prompt"]))
    _, Qhat, Chat, That = cp.predict(rows, alpha=1.0)
    return Predictions(Qhat, Chat, That)
