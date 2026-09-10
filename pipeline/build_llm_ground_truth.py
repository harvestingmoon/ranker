"""
Builds data/ground_truth_llm.json -- the LLM-judged ground truth, replacing
the taxonomy-formula ground_truth.json as the pipeline's primary evaluation
target.

Source: llm_judgments.py (LLM_JUDGMENTS + LLM_JUDGMENTS_V2_DELTA), 3,769
individually-read relevance judgments over a widened pool (data/
judging_pools_v2.json) built from 8 diverse retrieval configs plus a random
sample per hirer, specifically to reduce pooling bias (see the round-3
conversation for why: judging only the original 3-final-config pool would
have biased the ground truth toward exactly the methods it then scores).

Grades are the 0-3 ordinal scale (0=irrelevant, 1=weak/tangential,
2=good match, 3=excellent match), rescaled linearly to 0-100 to match
evaluate.py's existing scale and RELEVANCE_THRESHOLD convention:
    score = round(grade * 100 / 3)   ->   0, 33, 67, 100

Any (hirer, provider) pair NOT in the widened pool has no judgment at all
-- it is neither known-relevant nor known-irrelevant, it is simply
unjudged. Following standard TREC pooling convention, unjudged pairs are
treated as not relevant (absent from the ground truth dict) for scoring
purposes. This is still an approximation, not a claim that every unjudged
pair is truly irrelevant -- see the round-3 conversation for the tradeoff.

Run: python3 pipeline/build_llm_ground_truth.py
"""
import json
from pathlib import Path

from llm_judgments import LLM_JUDGMENTS, LLM_JUDGMENTS_V2_DELTA

DATA_DIR = Path(__file__).parent / "data"


def main():
    merged = {}
    all_hirers = set(LLM_JUDGMENTS) | set(LLM_JUDGMENTS_V2_DELTA)
    for hid in all_hirers:
        row = dict(LLM_JUDGMENTS.get(hid, {}))
        row.update(LLM_JUDGMENTS_V2_DELTA.get(hid, {}))
        merged[hid] = row

    ground_truth = {}
    for hid, row in merged.items():
        rescaled = {pid: round(grade * 100 / 3) for pid, grade in row.items() if grade > 0}
        ground_truth[hid] = rescaled

    out_path = DATA_DIR / "ground_truth_llm.json"
    out_path.write_text(json.dumps(ground_truth, indent=2))

    n_pairs = sum(len(v) for v in ground_truth.values())
    print(f"hirers: {len(ground_truth)}  relevant pairs (score>0): {n_pairs}  "
          f"avg score: {sum(s for row in ground_truth.values() for s in row.values()) / n_pairs:.1f}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
