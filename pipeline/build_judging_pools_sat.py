"""
Builds the LLM judging pools for the Scrape-and-Tag corpus (pipeline/data_sat/,
from import_scrape_and_tag.py) -- the (gig, provider) pairs that get graded 0-3.

Same TREC-style pooling as the synthetic ground truth (see llm_judgments.py /
judging_pools_v2.json): judge the union of what several retrievers put near the
top, plus a random sample, rather than the full gigs x providers cross-product.
A reranker only ever sees Stage 1's candidates, so that's where labels matter;
random pairs are almost all obviously irrelevant and teach it little.

Per gig, the pool is the union of
    RRF (k=60) top --rrf-top        the candidates the reranker will actually re-order
    refined BM25 top --method-top   lexical matches, incl. keyword-overlap near-misses
    dense (mxbai, full dim) top --method-top   semantic matches
    --random providers outside the above   so the labels aren't only retrieval-shaped

Writes to pipeline/data_sat/:
    judging_pools.json   {hire_id: [provider_id, ...]}
    pool_sources.json    {hire_id: {provider_id: ["rrf", "bm25", ...]}}  -- audit trail

Embeddings are cached under pipeline/cache/ (git-ignored), so a rerun with other
pool sizes doesn't re-embed.

Run: python pipeline/build_judging_pools_sat.py [--rrf-top 20 --method-top 10 --random 5]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from corpus import hirer_text, provider_text
from retrieval_bm25 import BM25Retriever
from retrieval_dense import BASE_MODEL_NAME, encode_docs, encode_queries, rank_from_raw
from retrieval_rrf import rrf_fuse

BASE = Path(__file__).parent
DATA_DIR = BASE / "data_sat"
CACHE_DIR = BASE / "cache"


def cached(name: str, build):
    path = CACHE_DIR / name
    if path.exists():
        return np.load(path)
    arr = build()
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-top", type=int, default=20)
    ap.add_argument("--method-top", type=int, default=10)
    ap.add_argument("--random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    hirers = json.loads((DATA_DIR / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((DATA_DIR / "providers.json").read_text(encoding="utf-8"))
    pids = [p["provider_id"] for p in providers]
    print(f"{len(hirers)} gigs x {len(providers)} providers")

    bm25 = BM25Retriever(providers, refined=True)
    print(f"embedding with {BASE_MODEL_NAME} (CPU is slow the first time; cached after)...", flush=True)
    doc_raw = cached("sat_docs_mxbai.npy", lambda: encode_docs([provider_text(p) for p in providers]))
    q_raw = cached("sat_queries_mxbai.npy", lambda: encode_queries([hirer_text(h) for h in hirers]))

    rng = random.Random(args.seed)
    pools, sources = {}, {}
    for i, h in enumerate(hirers):
        sparse = bm25.rank(hirer_text(h), query_title=h["hire_title"])
        dense = rank_from_raw(doc_raw, q_raw[i], pids)
        fused = rrf_fuse(sparse, dense, k=60)
        src = {}
        for tag, ranked, n in [("rrf", fused, args.rrf_top), ("bm25", sparse, args.method_top),
                               ("dense", dense, args.method_top)]:
            for pid, _ in ranked[:n]:
                src.setdefault(pid, []).append(tag)
        outside = [p for p in pids if p not in src]
        for pid in rng.sample(outside, args.random):
            src[pid] = ["random"]
        hid = str(h["hire_id"])
        pools[hid] = sorted(src)
        sources[hid] = {str(p): s for p, s in sorted(src.items())}

    (DATA_DIR / "judging_pools.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")
    (DATA_DIR / "pool_sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")

    sizes = [len(v) for v in pools.values()]
    print(f"pool size per gig: min {min(sizes)}, mean {np.mean(sizes):.1f}, max {max(sizes)}")
    print(f"total pairs to judge: {sum(sizes)}")
    print(f"written to {DATA_DIR}")


if __name__ == "__main__":
    main()
