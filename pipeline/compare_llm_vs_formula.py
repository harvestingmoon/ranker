"""
Compares the taxonomy-formula ground_truth.json against the LLM-judged
relevance in llm_judgments.py, and re-evaluates BM25/Dense/RRF against both,
to see whether the pipeline ranking conclusions hold when judged against
genuine content-level relevance rather than a taxonomy-ID formula.

LLM_JUDGMENTS uses a 0-3 ordinal scale; ground_truth.json uses a 0-100
continuous scale. To compare fairly, evaluate.py's RELEVANCE_THRESHOLD (40)
is mapped onto the 0-3 scale as >=2 ("good match" or better) counting as
relevant for Precision/Recall/MRR, matching the same "genuinely relevant,
not just tangential" bar.

Run: python3 pipeline/compare_llm_vs_formula.py
"""
import json
from pathlib import Path

from evaluate import evaluate_all_hirers, dcg_at_k, ndcg_at_k
from llm_judgments import LLM_JUDGMENTS

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def load_results(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())


# --- Build an LLM-judged ground truth in the same shape as ground_truth.json,
# rescaled 0-3 -> 0-100 linearly so evaluate.py's RELEVANCE_THRESHOLD=40
# lands at grade 2 ("good match") -- i.e. grade*100/3, so grade 2 -> 66.7,
# grade 1 -> 33.3, grade 3 -> 100.
def build_llm_ground_truth():
    gt = {}
    for hid, row in LLM_JUDGMENTS.items():
        gt[hid] = {pid: round(grade * 100 / 3) for pid, grade in row.items() if grade > 0}
    return gt


def agreement_stats(formula_gt, llm_gt, pools):
    """For every judged pair, compare the formula's implied relevance
    (>=40, i.e. grade>0 in original 0/35/70+adjustments scheme) against the
    LLM's judgment (grade>=2 counts relevant)."""
    both_relevant = formula_only = llm_only = neither = 0
    for hid, pool in pools.items():
        f_row = formula_gt.get(hid, {})
        l_row = LLM_JUDGMENTS.get(hid, {})
        for pid in pool:
            pid_s = str(pid)
            f_relevant = f_row.get(pid_s, 0) >= 40
            l_relevant = l_row.get(pid_s, 0) >= 2
            if f_relevant and l_relevant:
                both_relevant += 1
            elif f_relevant and not l_relevant:
                formula_only += 1
            elif l_relevant and not f_relevant:
                llm_only += 1
            else:
                neither += 1
    total = both_relevant + formula_only + llm_only + neither
    print(f"Agreement on {total} judged pairs:")
    print(f"  both say relevant:        {both_relevant} ({100*both_relevant/total:.1f}%)")
    print(f"  formula says yes, LLM no: {formula_only} ({100*formula_only/total:.1f}%)")
    print(f"  LLM says yes, formula no: {llm_only} ({100*llm_only/total:.1f}%)")
    print(f"  both say not relevant:    {neither} ({100*neither/total:.1f}%)")
    agree = both_relevant + neither
    print(f"  raw agreement rate: {100*agree/total:.1f}%")


def main():
    formula_gt = load_json("ground_truth.json")
    llm_gt = build_llm_ground_truth()
    pools = json.load(open(DATA_DIR / "judging_pools.json"))

    print("=" * 70)
    print("PART 1: Formula vs. LLM judgment agreement")
    print("=" * 70)
    agreement_stats(formula_gt, llm_gt, pools)

    print()
    print("=" * 70)
    print("PART 2: Re-evaluating BM25/Dense/RRF against LLM judgments")
    print("(restricted to the pooled hirers/candidates only)")
    print("=" * 70)

    approaches = {
        "BM25 (refined)": load_results("bm25_refined"),
        "Dense (1024-dim)": load_results("dense_full"),
        "RRF (sparse=0.7, dense=1.3)": load_results("rrf_w_sw0.7_dw1.3"),
    }

    for gt_name, gt in [("Formula ground truth", formula_gt), ("LLM-judged ground truth", llm_gt)]:
        print(f"\n--- against {gt_name} ---")
        for name, res in approaches.items():
            # restrict each hirer's ranked list to just the pooled candidates,
            # re-ranked in the same relative order, so both ground truths are
            # evaluated on an identical candidate universe
            restricted = {}
            for hid, ranked in res.items():
                if hid not in pools:
                    continue
                pool_set = set(pools[hid])
                restricted[hid] = [pid for pid in ranked if pid in pool_set]
            m = evaluate_all_hirers(restricted, gt, ks=(5, 10))
            print(f"  {name:<28} P@5={m['precision@5']:.3f}  R@5={m['recall@5']:.3f}  "
                  f"NDCG@5={m['ndcg@5']:.3f}  P@10={m['precision@10']:.3f}  "
                  f"R@10={m['recall@10']:.3f}  NDCG@10={m['ndcg@10']:.3f}  MRR={m['mrr']:.3f}")


if __name__ == "__main__":
    main()
