"""Build a mixed self-hosted + API model pool from LLMRouterBench results.

Uses the vendored file discovery (latest file per dataset/split/model)
but joins models on (dataset_id, prompt) rather than (dataset_id, split,
record_index, prompt): small models were scored on differently named splits
of the same questions (e.g. mmlupro test_1000 vs test_3000).
"""
import hashlib
import json

import numpy as np
import pandas as pd

from .source import CACHE, latest_files, num, prompt_text

KEY = ["dataset_id", "prompt"]


def _rows(filters):
    for path in latest_files(filters):
        with open(path, encoding="utf-8") as f:
            top = json.load(f)
        for r in top.get("records", []):
            if "index" not in r:
                continue
            yield (top["dataset_name"], top["split"], top["model_name"], int(r["index"]),
                   prompt_text(r.get("prompt", "")), num(r.get("score")),
                   num(r.get("cost")), num(r.get("prompt_tokens")),
                   num(r.get("completion_tokens")))
        del top


def build(models, datasets, splits=None):
    filters = {"skip_demo": True, "models": list(models), "datasets": list(datasets), "splits": splits}
    df = pd.DataFrame(_rows(filters), columns=["dataset_id", "split", "model_name", "record_index",
                                               "prompt", "score", "cost", "ptok", "ctok"])
    missing = set(models) - set(df["model_name"])
    if missing:
        raise RuntimeError(f"no results for models: {sorted(missing)}")
    # one row per (dataset, prompt, model): the lowest record_index wins
    df = df.sort_values("record_index").drop_duplicates(KEY + ["model_name"], keep="first")
    mats = {c: df.pivot(index=KEY, columns="model_name", values=c).reindex(columns=list(models))
            for c in ("score", "cost", "ptok", "ctok")}
    complete = mats["score"].notna().all(axis=1)
    index = mats["score"][complete].sort_index().index
    first = df.drop_duplicates(KEY).set_index(KEY)["record_index"]
    meta = index.to_frame(index=False)
    return {
        "Q": mats["score"].loc[index].to_numpy(np.float32),
        "C": mats["cost"].loc[index].fillna(0.0).to_numpy(np.float32),
        "Ptok": mats["ptok"].loc[index].fillna(0.0).to_numpy(np.float32),
        "Ctok": mats["ctok"].loc[index].fillna(0.0).to_numpy(np.float32),
        "models": np.array(models),
        "dataset_id": meta["dataset_id"].to_numpy(str),
        "record_index": first.loc[index].to_numpy(np.int64),
        "prompt": meta["prompt"].to_numpy(object),
    }


def load(models, datasets, splits=None):
    """Cached build(); the cache key is the model/dataset/split selection."""
    key = hashlib.sha1(json.dumps([list(models), list(datasets), splits]).encode()).hexdigest()[:12]
    path = CACHE / f"pool_{key}.npz"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(path, **build(models, datasets, splits))
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
