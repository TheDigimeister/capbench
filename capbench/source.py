"""LLMRouterBench result files, the leakage-safe split, token imputation and
the cluster quality predictor.

Vendored from an earlier `online_routing` prototype (data.py and stage1.py)
so this repository stands alone. Paths:
  CAPBENCH_DATA   directory holding bench-release/ (default: <repo>/data)
  CAPBENCH_CACHE  cache directory (default: <repo>/cache)
"""
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CAPBENCH_DATA", REPO / "data"))
BENCH = DATA / "bench-release"
CACHE = Path(os.environ.get("CAPBENCH_CACHE", REPO / "cache"))

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
N_CLUSTERS = 60
TOP_P = 4


# ------------------------------------------------------------ result files
def _match(value, include, exclude):
    if include is not None:
        return value in include
    if exclude is not None:
        return value not in exclude
    return True


def _timestamp_key(path):
    m = re.search(r"(\d{8})_(\d{6})\.json$", path.name)
    return int(m.group(1) + m.group(2)) if m else int(path.stat().st_mtime)


def prompt_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def latest_files(filters):
    """Latest result file per (dataset, split, model) passing the filters."""
    grouped = defaultdict(list)
    for path in sorted(BENCH.rglob("*.json")):
        with open(path, encoding="utf-8") as f:
            top = json.load(f)
        if filters.get("skip_demo", True) and top.get("demo", False):
            continue
        d, s, m = top.get("dataset_name"), top.get("split"), top.get("model_name")
        if not d or not m or s is None:
            continue
        if not (_match(d, filters.get("datasets"), filters.get("exclude_datasets"))
                and _match(m, filters.get("models"), filters.get("exclude_models"))
                and _match(s, filters.get("splits"), filters.get("exclude_splits"))):
            continue
        grouped[(d, s, m)].append(path)
    return sorted(max(files, key=_timestamp_key) for files in grouped.values())


# ------------------------------------------------------------ split, tokens
def split_by_dataset_prompt(dataset_id, record_index, prompt, train_ratio, seed, rows=None):
    """Shuffle distinct prompts within each dataset; no prompt is in both halves.
    Returns (train_rows, test_rows) as sorted arrays."""
    rows = np.arange(len(prompt)) if rows is None else np.asarray(rows)
    rng = random.Random(int(seed))
    train, test = [], []
    for d in dict.fromkeys(dataset_id[rows].tolist()):
        r = rows[dataset_id[rows] == d]
        first = {}
        for i in r:
            p, idx = prompt[i], record_index[i]
            first[p] = min(first.get(p, idx), idx)
        prompts = sorted(first, key=lambda p: first[p])
        pos = list(range(len(prompts)))
        rng.shuffle(pos)
        keep = {prompts[k] for k in pos[: int(len(prompts) * train_ratio)]}
        mask = np.array([prompt[i] in keep for i in r], dtype=bool)
        train.extend(r[mask].tolist())
        test.extend(r[~mask].tolist())
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


def impute_tokens(tokens, dataset_id):
    """Replace zero-token cells (failed calls) by the (dataset, model) median of non-zero cells."""
    t = tokens.astype(np.float64).copy()
    for d in np.unique(dataset_id):
        rows = dataset_id == d
        for j in range(t.shape[1]):
            col = t[rows, j]
            nz = col[col > 0]
            col[col <= 0] = np.median(nz) if len(nz) else np.median(t[t > 0])
            t[rows, j] = col
    return t


# ------------------------------------------------------------ predictor
def embeddings(prompts):
    h = hashlib.sha1(EMBED_MODEL.encode())
    for p in prompts:
        h.update(b"\0" + str(p).encode("utf-8", "replace"))
    path = CACHE / f"emb_{h.hexdigest()[:16]}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    E = enc.encode([str(p) for p in prompts], batch_size=32, show_progress_bar=True,
                   normalize_embeddings=True).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    np.save(path, E)
    return E


def _row_minmax(X):
    lo, hi = X.min(1, keepdims=True), X.max(1, keepdims=True)
    return (X - lo) / np.maximum(hi - lo, 1e-8)


class ClusterPredictor:
    """KMeans(k=60) on train embeddings; a query's forecast averages its top-4 clusters."""

    def __init__(self, E, Q, C, T, train_rows, seed, tag):
        from sklearn.cluster import KMeans
        split = hashlib.sha1(np.asarray(train_rows, np.int64).tobytes()).hexdigest()[:8]
        path = CACHE / f"kmeans_{tag}_train{split}_seed{seed}_k{N_CLUSTERS}.npz"
        if path.exists():
            z = np.load(path)
            labels, self.centers = z["labels"], z["centers"]
        else:
            km = KMeans(n_clusters=N_CLUSTERS, random_state=int(seed), n_init=10)
            labels = km.fit_predict(E[train_rows])
            self.centers = km.cluster_centers_.astype(np.float32)
            CACHE.mkdir(exist_ok=True)
            np.savez_compressed(path, labels=labels, centers=self.centers)
        m = Q.shape[1]
        self.q, self.c, self.t = (np.zeros((N_CLUSTERS, m)) for _ in range(3))
        Qtr, Ctr, Ttr = Q[train_rows], C[train_rows], T[train_rows]
        for k in range(N_CLUSTERS):
            mask = labels == k
            src = (Qtr[mask], Ctr[mask], Ttr[mask]) if mask.any() else (Qtr, Ctr, Ttr)
            self.q[k], self.c[k], self.t[k] = (x.mean(0) for x in src)
        self.q_norm, self.c_norm = _row_minmax(self.q), _row_minmax(self.c)
        self.E = E

    def predict(self, rows, alpha):
        """Return (S, Qhat, Chat, That) for the given rows."""
        from sklearn.metrics import pairwise_distances
        dist = pairwise_distances(self.E[rows], self.centers)
        near = np.argpartition(dist, TOP_P - 1, axis=1)[:, :TOP_P]
        score = alpha * self.q_norm + (1 - alpha) * (1 - self.c_norm)
        S = score[near].sum(1)
        return S, self.q[near].mean(1), np.maximum(self.c[near].mean(1), 1e-8), self.t[near].mean(1)
