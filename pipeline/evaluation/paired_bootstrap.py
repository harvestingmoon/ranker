"""
Evaluate every ranked-list file in results_<tag>/ on the labelled queries, then
run a paired bootstrap between them.

Discovers systems from the directory rather than hardcoding paths, so it works for
any corpus (data_pilot, data_all, ...) as soon as the run produces files.

Usage: python -m pipeline.evaluation.paired_bootstrap --tag data_all [--baseline rrf_k60.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.common.evaluate import evaluate_all_hirers
from pipeline.common.paths import PIPELINE

BASE = PIPELINE
METRICS = ("precision@5", "recall@10", "ndcg@10", "mrr")


def load_systems(results_dir: Path):
    """Discover every ranked-list file in the results folder as a competing system.

    Discovery rather than a hardcoded list, so a new run's JSON is compared
    automatically. Files that are not {query_id: [ranked ids]} are skipped, which
    keeps stray artifacts out of the comparison.
    """
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        # a ranked-list file: {query_id: [int, int, ...]}
        if not isinstance(d, dict) or not d:
            continue
        first = next(iter(d.values()))
        if isinstance(first, list) and first and isinstance(first[0], int):
            out[p.name] = d
    return out


def main():
    """CLI: score every system in a results folder, then paired-bootstrap each one
    against --baseline.

    Resampling is over queries, not over pairs, so the CI respects the fact that a
    gig's candidates are not independent evidence. A delta whose 95% CI crosses 0
    is noise at this corpus size, not a win.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data_all")
    ap.add_argument("--baseline", default="rrf_k60.json")
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()

    results_dir = BASE / f"results_{args.tag}"
    gt = json.loads((BASE / args.tag / "ground_truth_llm.json").read_text())
    systems = load_systems(results_dir)
    if not systems:
        raise SystemExit(f"no ranked-list files in {results_dir}")
    if args.baseline not in systems:
        raise SystemExit(f"baseline {args.baseline} not found; have {list(systems)}")

    # score everything on every query, then restrict to queries that have labels
    scored = {n: evaluate_all_hirers(r, gt) for n, r in systems.items()}
    labelled = [h for h in gt]

    print(f"corpus {args.tag}: {len(gt)} labelled queries, "
          f"{sum(len(v) for v in gt.values())} judged pairs\n")
    print(f"{'system':<26}" + "".join(f"{m:>11}" for m in METRICS))
    for n in systems:
        print(f"{n:<26}" + "".join(f"{scored[n][m]:>11.4f}" for m in METRICS))

    # per-query vectors for the paired test
    from pipeline.common.evaluate import ndcg_at_k, precision_at_k, reciprocal_rank
    per = {n: {h: {
        "precision@5": precision_at_k(r.get(h, []), gt.get(h, {}), 5),
        "ndcg@10": ndcg_at_k(r.get(h, []), gt.get(h, {}), 10),
        "mrr": reciprocal_rank(r.get(h, []), gt.get(h, {})),
    } for h in labelled} for n, r in systems.items()}

    rng = np.random.default_rng(20261011)
    print(f"\npaired bootstrap vs {args.baseline}, {args.resamples:,} resamples over queries:")
    for n in systems:
        if n == args.baseline:
            continue
        print(f"\n  {n}  vs  {args.baseline}")
        for m in METRICS:
            if m not in per[n][labelled[0]]:
                continue
            a = np.array([per[args.baseline][h][m] for h in labelled])
            b = np.array([per[n][h][m] for h in labelled])
            d = b - a
            boot = d[rng.integers(0, len(d), size=(args.resamples, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            flag = "crosses 0" if lo <= 0 <= hi else "EXCLUDES 0"
            print(f"    {m:<12} d={d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {flag}")


if __name__ == "__main__":
    main()
