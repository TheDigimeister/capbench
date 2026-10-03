"""Download every input the benchmark needs, pinned and checksummed.

    python -m capbench.fetch            # LLMRouterBench results (1.3 GB) + arrival traces (53 MB)

LLMRouterBench (github.com/ynulihao/LLMRouterBench, MIT per its README) is
extracted to $CAPBENCH_DATA/bench-release (default <repo>/data). Nothing from
it is redistributed in this repository: some of its datasets (e.g. GPQA) ask
that examples not be posted online.
"""
import tarfile
import urllib.request

from . import traces
from .source import BENCH, DATA

LLMROUTERBENCH = {
    "url": "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/"
           "0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz",
    "file": "bench-release.tar.gz",
    "sha256": "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1",
}


def fetch_llmrouterbench():
    if BENCH.is_dir():
        print("found", BENCH)
        return
    DATA.mkdir(parents=True, exist_ok=True)
    tar = DATA / LLMROUTERBENCH["file"]
    if not tar.exists():
        print("downloading", LLMROUTERBENCH["url"])
        urllib.request.urlretrieve(LLMROUTERBENCH["url"], tar)
    if traces.sha256(tar) != LLMROUTERBENCH["sha256"]:
        raise RuntimeError(f"{tar}: sha256 mismatch (expected {LLMROUTERBENCH['sha256']})")
    with tarfile.open(tar) as t:
        t.extractall(DATA, filter="data")
    if not BENCH.is_dir():
        raise RuntimeError(f"{tar} did not contain bench-release/")
    print("extracted", BENCH)


def main():
    fetch_llmrouterbench()
    traces.fetch()
    print("traces in", traces.RAW)


if __name__ == "__main__":
    main()
