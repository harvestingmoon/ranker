"""
IR evaluation metrics: Precision@K, Recall@K, NDCG@K (graded relevance), and
MRR (Mean Reciprocal Rank), against the ground_truth relevance scores
produced by generate_synthetic_data.py.

ground_truth format (v2): {hire_id_str: {provider_id_str: score}}, score is
a continuous 1-100 integer. A pair absent from the dict is score 0
(irrelevant).

NDCG uses the score directly as a *linear* gain (`dcg`), not the classic
`2^rel - 1` exponential gain -- that formula is meant for small integer
grades (0-3ish) and would blow up (2^100) on our continuous 0-100 scale.
Since NDCG is a ratio of DCG/IDCG, any consistent monotonic rescaling of the
gain cancels out in the ratio, so a linear gain is both correct and simpler
here.

Precision/Recall/MRR need a binary relevant/not-relevant decision, so a
continuous score is thresholded at RELEVANCE_THRESHOLD.
"""
import math

RELEVANCE_THRESHOLD = 40  # score >= this counts as "relevant" for P/R/MRR


def _relevance(ground_truth_row: dict, provider_id: int) -> int:
    return ground_truth_row.get(str(provider_id), 0)


def _is_relevant(ground_truth_row: dict, provider_id: int) -> bool:
    return _relevance(ground_truth_row, provider_id) >= RELEVANCE_THRESHOLD


def precision_at_k(ranked_ids: list[int], ground_truth_row: dict, k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for pid in top_k if _is_relevant(ground_truth_row, pid))
    return relevant / len(top_k)


def recall_at_k(ranked_ids: list[int], ground_truth_row: dict, k: int) -> float:
    total_relevant = sum(1 for pid_str in ground_truth_row if ground_truth_row[pid_str] >= RELEVANCE_THRESHOLD)
    if total_relevant == 0:
        return None  # undefined -- no relevant items exist for this query
    top_k = ranked_ids[:k]
    relevant_found = sum(1 for pid in top_k if _is_relevant(ground_truth_row, pid))
    return relevant_found / total_relevant


def dcg_at_k(ranked_ids: list[int], ground_truth_row: dict, k: int) -> float:
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        rel = _relevance(ground_truth_row, pid)
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i+2: rank is 1-indexed, i is 0-indexed
    return dcg


def ndcg_at_k(ranked_ids: list[int], ground_truth_row: dict, k: int) -> float:
    ideal_order = sorted(ground_truth_row.values(), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_order[:k]))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, ground_truth_row, k) / idcg


def reciprocal_rank(ranked_ids: list[int], ground_truth_row: dict) -> float:
    for i, pid in enumerate(ranked_ids):
        if _is_relevant(ground_truth_row, pid):
            return 1.0 / (i + 1)
    return 0.0


def evaluate_all_hirers(
    results_by_hirer: dict[str, list[int]],
    ground_truth: dict[str, dict],
    ks: tuple[int, ...] = (5, 10),
) -> dict:
    """results_by_hirer: {hire_id_str: [ranked provider_id, ...]}
    Returns averaged metrics across all hirers."""
    metrics = {f"precision@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    metrics.update({f"ndcg@{k}": [] for k in ks})
    metrics["mrr"] = []

    for hid, ranked_ids in results_by_hirer.items():
        gt_row = ground_truth.get(hid, {})
        for k in ks:
            metrics[f"precision@{k}"].append(precision_at_k(ranked_ids, gt_row, k))
            r = recall_at_k(ranked_ids, gt_row, k)
            if r is not None:
                metrics[f"recall@{k}"].append(r)
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_ids, gt_row, k))
        metrics["mrr"].append(reciprocal_rank(ranked_ids, gt_row))

    return {name: (sum(vals) / len(vals) if vals else None) for name, vals in metrics.items()}
