"""
Pro tier: the 37-feature vector the set-wise ranker (pro_ranker.py) reads.

    17  base           the Lightweight features (rerank_ltr.load_candidates),
                       score features z-scored within each gig
     6  taxonomy graph partial credit through the sf_v1 tree (graph_features)
     3  interactions   tag_level crossed with rank / coverage
     2  provider prior fold-aware mean grade of the provider on OTHER gigs
     3  cross-encoder  out-of-fold score from each fine-tuned CE (merge_ce_scores)
     6  CE consensus   per-gig rank of each CE score, top-10/top-20 votes, mean rank

Plus the continuous labels (load_soft) the pairwise loss is weighted by.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.lightweight import rerank_ltr as R

GRAPH_FEATURES = ["skill_sib_cov", "skill_cat_cov", "graph_dist_mean", "graph_dist_min",
                  "spec_cat_cov", "skill_jaccard"]
INTERACTION_FEATURES = ["tl_x_rank", "speccov_x_tl", "skillcov_x_tl"]
PRIOR_FEATURES = ["prov_prior", "prov_ct"]


# --------------------------------------------------------------------------
# taxonomy graph
# --------------------------------------------------------------------------
def load_index(data_dir: Path):
    """Each record's (categories, specialisations, skills), plus skill -> spec/category maps."""
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))

    sk2spec, sk2cat = defaultdict(set), defaultdict(set)

    def axes(rec):
        cats, specs, skills = set(), set(), set()
        for c in (rec.get("tags") or {}).get("categories") or []:
            cats.add(c["name"])
            for s in c.get("specialisations") or []:
                specs.add((c["name"], s["name"]))
                for k in s.get("skills") or []:
                    skills.add(k)
                    sk2spec[k].add((c["name"], s["name"]))
                    sk2cat[k].add(c["name"])
        return cats, specs, skills

    gig = {r["hire_id"]: axes(r) for r in hirers}
    prov = {r["provider_id"]: axes(r) for r in providers}
    return gig, prov, sk2spec, sk2cat


def graph_features(g, p, sk2spec, sk2cat):
    """Partial-credit coverage of the gig's skills by the provider.

    A gig skill the provider does not hold exactly can still be reached through
    the tree: a sibling skill under the same specialisation, or any skill under
    the same category.
    """
    gc, gs, gk = g
    pc, ps, pk = p
    shared = gk & pk

    # best hop distance per gig skill: 0 exact, 1 sibling (same spec), 2 same cat, 3 none
    dists = []
    for k in gk:
        if k in pk:
            dists.append(0)
        elif any(sk2spec[k] & sk2spec[k2] for k2 in pk):
            dists.append(1)
        elif any(sk2cat[k] & sk2cat[k2] for k2 in pk):
            dists.append(2)
        else:
            dists.append(3)
    if dists:
        d = np.asarray(dists)
        sib_cov = float((d <= 1).mean())
        cat_cov = float((d <= 2).mean())
        dist_mean = float(d.mean() / 3.0)
        dist_min = float(d.min() / 3.0)
    else:
        sib_cov = cat_cov = dist_mean = dist_min = 0.0

    # share of the gig's (cat, spec) pairs whose CATEGORY the provider covers
    spec_cat_cov = float(sum(1 for (c, _s) in gs if c in pc) / len(gs)) if gs else 0.0
    jacc = float(len(shared) / len(gk | pk)) if (gk | pk) else 0.0

    return {
        "skill_sib_cov": sib_cov,
        "skill_cat_cov": cat_cov,
        "graph_dist_mean": dist_mean,
        "graph_dist_min": dist_min,
        "spec_cat_cov": spec_cat_cov,
        "skill_jaccard": jacc,
    }


# --------------------------------------------------------------------------
# feature rows
# --------------------------------------------------------------------------
def build_rows(data_dir):
    """Base + graph + interaction rows. The prior columns are filled per fold (prior_by_fold)."""
    R.set_dataset(data_dir)
    rows, base = R.load_candidates(50, tag_features=True)
    R.normalize_by_query(rows, R.SCORE_FEATURES, "zscore")

    gig, prov, sk2spec, sk2cat = load_index(Path(R.DATA_DIR))
    for r in rows:
        r.update(graph_features(gig[r["hire_id"]], prov[r["provider_id"]], sk2spec, sk2cat))
        tl = r["tag_level"]
        r["tl_x_rank"] = tl * (50.0 - r["rrf_rank"]) / 49.0
        r["speccov_x_tl"] = r["spec_cov"] * tl
        r["skillcov_x_tl"] = r["skill_cov"] * tl

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    feats = list(base) + GRAPH_FEATURES + INTERACTION_FEATURES + PRIOR_FEATURES
    return rows, by_query, feats


def prior_by_fold(rows, tr_q, alpha=5.0):
    """Provider prior from the TRAINING gigs only: alpha-smoothed mean grade of the
    provider, leaving out the row's own gig. Rebuilt every fold, so no label leaks."""
    tot_s, tot_c = defaultdict(float), defaultdict(int)
    pq_s, pq_c = defaultdict(float), defaultdict(int)
    gsum = gcnt = 0
    for r in rows:
        if r["label"] is None or r["hire_id"] not in tr_q:
            continue
        p, q, y = r["provider_id"], r["hire_id"], r["label"]
        tot_s[p] += y; tot_c[p] += 1; pq_s[(p, q)] += y; pq_c[(p, q)] += 1
        gsum += y; gcnt += 1
    gm = gsum / gcnt if gcnt else 0.0
    for r in rows:
        p, q = r["provider_id"], r["hire_id"]
        s, c = tot_s[p], tot_c[p]
        if q in tr_q:
            s -= pq_s[(p, q)]; c -= pq_c[(p, q)]
        r["prov_prior"] = (s + alpha * gm) / (c + alpha)
        r["prov_ct"] = float(np.log1p(c))


# --------------------------------------------------------------------------
# cross-encoder inputs
# --------------------------------------------------------------------------
def merge_ce_scores(rows, ce_dir: Path, ce_tags):
    """Add ce_score, ce_score2, ... from features_data/<corpus>/ce_scores_<tag>.csv.
    A pair with no CE score gets 0.0. Returns the new column names."""
    cols = []
    for i, tag in enumerate(ce_tags):
        col = "ce_score" if i == 0 else f"ce_score{i + 1}"
        path = ce_dir / f"ce_scores_{tag}.csv"
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(path.open())}
        miss = 0
        for r in rows:
            v = ce.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None or v != v:
                v = 0.0; miss += 1
            r[col] = v
        cols.append(col)
        print(f"{col} <- {tag} ({miss} missing -> 0.0)")
    return cols


def _rank_desc(vals):
    """1-based descending rank; ties share the best rank."""
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    out = [0] * len(vals)
    prev, prev_rank = None, 0
    for pos, i in enumerate(order, 1):
        if vals[i] != prev:
            prev, prev_rank = vals[i], pos
        out[i] = prev_rank
    return out


def add_ce_rank_features(by_query, cols):
    """Per-gig rank of each CE score + consensus votes. Ranks are relative to the
    shortlist, which is how a reranker is actually consumed."""
    names = [f"rank_{c}" for c in cols] + ["ce_top10_votes", "ce_top20_votes", "ce_mean_rank"]
    for rs in by_query.values():
        ranks = {c: _rank_desc([r[c] for r in rs]) for c in cols}
        for c in cols:
            for r, v in zip(rs, ranks[c]):
                r[f"rank_{c}"] = float(v)
        for i, r in enumerate(rs):
            rk = [ranks[c][i] for c in cols]
            r["ce_top10_votes"] = float(sum(x <= 10 for x in rk))
            r["ce_top20_votes"] = float(sum(x <= 20 for x in rk))
            r["ce_mean_rank"] = float(np.mean(rk))
    return names


# --------------------------------------------------------------------------
# continuous labels
# --------------------------------------------------------------------------
def load_soft(data_root: Path):
    """The grader's continuous score in [0, 1] per (hire_id, provider_id), including top-up grades."""
    soft = {}
    scores = json.loads((data_root / "llm_scores_v5.json").read_text())
    for h, v in scores.items():
        for p, s in v.items():
            soft[(int(h), int(p))] = float(s)
    topup = data_root / "judgments_topup_tags.jsonl"
    if topup.exists():
        for line in topup.read_text().splitlines():
            d = json.loads(line)
            soft[(int(d["hire_id"]), int(d["provider_id"]))] = float(d["score"])
    return soft
