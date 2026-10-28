"""
Lightweight tier, Stage 2: LambdaMART re-ranker over the top-50 RRF candidates.

Reads `features_data/<corpus>/candidates_top50.csv` (retrieval + structured
features + labels, from features.py) and `tag_features.csv` (from
tag_features.py), and trains XGBoost `rank:ndcg` on the 17 features below.

- Labels are the LLM grades 0-3, trained with LINEAR gain (ndcg_exp_gain=False)
  so the model optimises the same NDCG that evaluate.py measures.
- Score features are z-scored within each gig (see normalize_by_query).
- GroupKFold by gig: every gig is scored by a model that never saw it, so the
  reported metrics and the written ranked lists are out-of-fold (OOF).
- A final model is then fit on all gigs and saved for serving.

    python -m pipeline.lightweight.rerank_ltr --data-dir data/data_set_v5_tax --tag v5tax_tags

Writes results_data/<corpus>/<tag>.json (OOF ranked lists) and
models_data/<corpus>/<tag>_xgb.json (serving model, read by serve.py).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from pipeline.common.evaluate import evaluate_all_hirers
from pipeline.features.tag_features import TAG_FEATURES
from pipeline.common.paths import PIPELINE

BASE = PIPELINE
DATA_DIR = BASE / "data"
FEAT_DIR = BASE / "features_data"
RESULTS_DIR = BASE / "results_data"
MODELS_DIR = BASE / "models_data"


def set_dataset(data_dir: str):
    """Point every path at one corpus: data/<name> -> {features,results,models}_data/<name>."""
    global DATA_DIR, FEAT_DIR, RESULTS_DIR, MODELS_DIR
    name = Path(data_dir).name
    DATA_DIR = BASE / data_dir
    FEAT_DIR = BASE / "features_data" / name
    RESULTS_DIR = BASE / "results_data" / name
    MODELS_DIR = BASE / "models_data" / name


def as_id(value):
    """Ids are kept as they appear in the corpus: ints when numeric, else strings."""
    s = str(value).strip()
    return int(s) if s.isdigit() else s


# Shared by the CV models and the final model so the two can never drift apart.
# `ndcg_exp_gain=False` = linear gain, matching evaluate.py (XGBoost's default
# exponential gain would treat grade 3 as 7x grade 1 instead of 3x).
MODEL_PARAMS = dict(
    objective="rank:ndcg", n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
    ndcg_exp_gain=False,
)
DEFAULT_FEATURES = ["bm25_score", "bm25_rank", "dense_cosine", "dense_rank",
                    "rrf_score", "rrf_rank", "budget_fit", "seniority_fit", "avail_immediacy"]
# Features whose raw scale drifts per gig, so they are z-scored within each gig.
# The *_rank features are already monotone within a gig, so they are left alone.
SCORE_FEATURES = ["bm25_score", "dense_cosine", "rrf_score",
                  "budget_fit", "seniority_fit", "avail_immediacy"]


def normalize_by_query(rows, keys, how="zscore"):
    """Z-score feature(s) WITHIN each gig (population std).

    bm25_score depends on query length and term rarity, dense_cosine on the
    query's embedding neighbourhood, rrf_score on how many lists agreed. A tree
    splits on absolute values, so the same threshold would mean something
    different for every gig; per-gig z-scores make a split mean "how does this
    candidate compare with its siblings for the same gig".
    """
    if how == "none":
        return
    if how != "zscore":
        raise ValueError(f"unsupported normalisation: {how}")
    if isinstance(keys, str):
        keys = [keys]
    by_query = {}
    for r in rows:
        by_query.setdefault(r["hire_id"], []).append(r)
    for group in by_query.values():
        for key in keys:
            vals = [r[key] for r in group if r[key] == r[key]]  # skip NaN
            if len(vals) < 2:
                continue
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
            for r in group:
                if r[key] == r[key]:
                    r[key] = (r[key] - mu) / sd


def train_labels(rows):
    """Training target: the raw 0-3 grade."""
    return np.array([r["label"] for r in rows], dtype=np.float32)


def load_json(p):
    return json.loads(Path(p).read_text())


def load_candidates(top_k: int = 50, tag_features: bool = True):
    """Candidate rows with their feature values as floats; returns (rows, feature names)."""
    path = FEAT_DIR / f"candidates_top{top_k}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python -m pipeline.features.features` first")
    rows = list(csv.DictReader(path.open()))

    features = list(DEFAULT_FEATURES)
    if tag_features:
        # Built in a separate process: tag_features.py loads torch, which
        # segfaults alongside xgboost on macOS.
        tf_path = FEAT_DIR / "tag_features.csv"
        if not tf_path.exists():
            raise SystemExit(f"{tf_path} not found -- build it with "
                             f"`python -m pipeline.features.tag_features --data-dir {DATA_DIR.relative_to(BASE)}`")
        tf = {(r["hire_id"], r["provider_id"]): r for r in csv.DictReader(tf_path.open())}
        missing = [r for r in rows if (r["hire_id"], r["provider_id"]) not in tf]
        if missing:
            raise SystemExit(f"{tf_path.name} is stale: {len(missing)} candidate rows have no tag "
                             f"features -- rebuild it after features.py")
        for r in rows:
            src = tf[(r["hire_id"], r["provider_id"])]
            for f in TAG_FEATURES:
                r[f] = src[f]
        features.extend(TAG_FEATURES)

    for r in rows:
        for f in features:
            r[f] = float(r[f]) if r.get(f) not in ("", None) else float("nan")
        r["label"] = int(r["label"]) if r["label"] not in ("", None) else None
        r["hire_id"] = as_id(r["hire_id"])
        r["provider_id"] = as_id(r["provider_id"])
    return rows, features


def full_lists(ranked_head, rrf_full):
    """Pad each top-50 list with the rest of the RRF ranking (only MRR can read the tail)."""
    out = {}
    for hid, head in ranked_head.items():
        seen = set(head)
        out[str(hid)] = head + [pid for pid in rrf_full.get(str(hid), []) if pid not in seen]
    return out


def report(name, ranked_by_query, gt, rrf_full):
    m = evaluate_all_hirers(full_lists(ranked_by_query, rrf_full), gt, ks=(5, 10))
    print(f"  {name:22s} P@5={m['precision@5']:.3f} R@5={m['recall@5']:.3f} NDCG@5={m['ndcg@5']:.3f} "
          f"P@10={m['precision@10']:.3f} R@10={m['recall@10']:.3f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.3f}")
    return m


def train_oof(rows, features, folds, seed, gt, rrf_full):
    """Return out-of-fold LambdaMART scores keyed by (hire_id, provider_id)."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    queries = sorted(by_query)
    oof, importances, fold_metrics = {}, [], []

    for fold, (tr_idx, te_idx) in enumerate(GroupKFold(n_splits=folds).split(queries, groups=queries), 1):
        tr_q = sorted(queries[i] for i in tr_idx)
        te_q = sorted(queries[i] for i in te_idx)

        tr_rows = [r for q in tr_q for r in by_query[q] if r["label"] is not None]
        X = np.array([[r[f] for f in features] for r in tr_rows], dtype=np.float32)
        groups = [sum(1 for r in tr_rows if r["hire_id"] == q) for q in tr_q]
        model = xgb.XGBRanker(**MODEL_PARAMS, random_state=seed, n_jobs=4)
        model.fit(X, train_labels(tr_rows), group=groups)

        te_rows = [r for q in te_q for r in by_query[q]]
        Xt = np.array([[r[f] for f in features] for r in te_rows], dtype=np.float32)
        preds = model.predict(Xt)
        for r, p in zip(te_rows, preds):
            oof[(r["hire_id"], r["provider_id"])] = float(p)

        importances.append(model.feature_importances_)
        fold_rank = {}
        for r, p in zip(te_rows, preds):
            fold_rank.setdefault(r["hire_id"], []).append((r["provider_id"], float(p)))
        ranked = {q: [pid for pid, _ in sorted(v, key=lambda x: -x[1])] for q, v in fold_rank.items()}
        fold_metrics.append(evaluate_all_hirers(full_lists(ranked, rrf_full), gt, ks=(10,))["ndcg@10"])
        print(f"  fold {fold}: {len(tr_q)} train / {len(te_q)} held-out gigs  NDCG@10={fold_metrics[-1]:.4f}")

    print("\n  mean feature importance:")
    for f, v in sorted(zip(features, np.mean(importances, axis=0)), key=lambda x: -x[1]):
        print(f"    {f:18s} {v:.4f}")
    print(f"  fold NDCG@10: mean={np.mean(fold_metrics):.4f} std={np.std(fold_metrics):.4f}")
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--tag", default="v5tax_tags", help="output name for the OOF lists and the model")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    set_dataset(args.data_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # The ranking tail beyond the top 50 only pads each list (see full_lists); the
    # ~31 MB dump is git-ignored, so training also runs without it.
    rrf_path = RESULTS_DIR / "rrf_k60.json"
    rrf_full = load_json(rrf_path) if rrf_path.exists() else {}
    gt = load_json(DATA_DIR / "ground_truth_llm.json")

    rows, features = load_candidates()
    n_labelled = sum(1 for r in rows if r["label"] is not None)
    if not n_labelled:
        raise SystemExit(f"FAIL: no candidate row in {FEAT_DIR.name}/candidates_top50.csv carries a "
                         f"label -- the grades are keyed on different ids than the corpus")
    normalize_by_query(rows, SCORE_FEATURES)
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    print(f"loaded {len(rows)} candidate rows / {len(by_query)} gigs ({n_labelled} labelled)")
    print(f"features ({len(features)}): {features}\n")

    print("baseline (same candidate pool):")
    base_rrf = {q: [r["provider_id"] for r in sorted(rs, key=lambda r: r["rrf_rank"])]
                for q, rs in by_query.items()}
    m_rrf = report("RRF", base_rrf, gt, rrf_full)

    print("\nLambdaMART (GroupKFold by gig, out-of-fold):")
    oof = train_oof(rows, features, args.folds, args.seed, gt, rrf_full)
    ltr_rank = {q: [r["provider_id"] for r in sorted(rs, key=lambda r: -oof[(q, r["provider_id"])])]
                for q, rs in by_query.items()}
    m_ltr = report("LambdaMART (OOF)", ltr_rank, gt, rrf_full)

    oof_path = RESULTS_DIR / f"{args.tag}.json"
    oof_path.write_text(json.dumps({str(q): [int(pid) for pid in rs] for q, rs in ltr_rank.items()}, indent=2))
    print(f"wrote out-of-fold ranked lists to {oof_path}")

    # Serving model: the same configuration fit on every labelled gig.
    all_rows = sorted([r for r in rows if r["label"] is not None], key=lambda r: r["hire_id"])
    X = np.array([[r[f] for f in features] for r in all_rows], dtype=np.float32)
    all_q = sorted({r["hire_id"] for r in all_rows})
    groups = [sum(1 for r in all_rows if r["hire_id"] == q) for q in all_q]
    final = xgb.XGBRanker(**MODEL_PARAMS, random_state=args.seed, n_jobs=4)
    final.fit(X, train_labels(all_rows), group=groups)
    model_path = MODELS_DIR / f"{args.tag}_xgb.json"
    final.save_model(str(model_path))
    print(f"saved full-data model to {model_path}")

    print("\n--- summary ---")
    print(f"  RRF baseline        NDCG@10={m_rrf['ndcg@10']:.4f}  P@5={m_rrf['precision@5']:.3f}  MRR={m_rrf['mrr']:.3f}")
    print(f"  + LambdaMART (OOF)  NDCG@10={m_ltr['ndcg@10']:.4f}  P@5={m_ltr['precision@5']:.3f}  MRR={m_ltr['mrr']:.3f}")
    print(f"\n{len(by_query)} gigs -- treat any NDCG@10 delta under ~0.007 as noise.")


if __name__ == "__main__":
    main()
