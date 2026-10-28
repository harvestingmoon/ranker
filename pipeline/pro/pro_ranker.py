"""
Pro tier, Stage 2: set-wise transformer over each gig's 50 candidates.

A per-row model (the Lightweight LambdaMART) scores every candidate on its own,
but NDCG@10 is a list metric. Here each candidate is a 37-feature vector
(pro_features.py) and self-attention runs across the whole shortlist, so the
model can learn relative judgements ("the best fit for THIS gig") and decide,
per gig, how much to trust each cross-encoder.

    candidate features [50 x 37] -> Linear(37->64) -> GELU -> dropout
        -> 2 x TransformerEncoderLayer(d=64, 4 heads, norm-first) -> Linear(64->1)

Loss: pairwise logistic weighted by |gain_i - gain_j|, on the grader's continuous
0-100 score; unlabelled candidates carry no weight. Out-of-fold by gig
(GroupKFold), several seeds averaged per fold.

Needs the three cross-encoder OOF score files from rerank_crossencoder.py.
The defaults are the shipped configuration (0.9507 NDCG@10):

    python -m pipeline.pro.pro_ranker

Writes results_data/<corpus>/listwise_<tag>.json (OOF ranked lists). No model is
saved yet, so Pro has no serving path.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, KFold

from pipeline.common import evaluate as E
from pipeline.pro import pro_features as P
from pipeline.lightweight import rerank_ltr as R

GAIN = {0: 0.0, 1: 33.0, 2: 67.0, 3: 100.0}


class SetRanker(nn.Module):
    def __init__(self, n_feat, d=64, layers=2, heads=4, dropout=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                       # [B, N, F]
        h = self.enc(self.inp(x))
        return self.head(h).squeeze(-1)         # [B, N]


def pairwise_loss(scores, gains, mask):
    """Pairwise logistic, weighted by how much better i is than j. Unlabelled
    candidates have their gain zeroed, so they never contribute weight."""
    g = gains * mask.float()
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)      # s_i - s_j
    w = torch.relu(g.unsqueeze(2) - g.unsqueeze(1))       # >0 only when g_i > g_j
    return -(torch.nn.functional.logsigmoid(diff) * w).sum() / (w.sum() + 1e-8)


def build_x(by_query, feats, queries):
    """[gigs, 50, features], standardised within each gig and clipped to +-5."""
    out = []
    for q in queries:
        x = np.asarray([[r[f] for f in feats] for r in by_query[q]], dtype=np.float32)
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        out.append(np.clip((x - mu) / sd, -5, 5))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--ce-tags", default="v5ce_minilm,v5ce_bge,v5ce_gte",
                    help="cross-encoder score files to merge, in order")
    ap.add_argument("--ce-data-dir", default="data_set_v5",
                    help="features_data/<this>/ holds the ce_scores_<tag>.csv files")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="0 = GroupKFold by gig; >0 = shuffled KFold over gigs with this seed")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=7, help="models averaged per fold")
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--tag", default="ceil_s7")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = json.loads((R.BASE / args.data_dir / "ground_truth_llm.json").read_text())

    # ---- features -------------------------------------------------------
    rows, by_query, feats = P.build_rows(args.data_dir)
    ce_cols = P.merge_ce_scores(rows, R.BASE / "features_data" / args.ce_data_dir,
                                [t.strip() for t in args.ce_tags.split(",") if t.strip()])
    feats += ce_cols
    feats += P.add_ce_rank_features(by_query, ce_cols)

    # ---- labels: continuous grader score where available, else the 0-3 grade
    queries = sorted(by_query)
    q_index = {q: i for i, q in enumerate(queries)}
    soft = P.load_soft(R.BASE / "data" / "data_set_v5")
    print(f"soft labels loaded: {len(soft)} pairs")
    G = np.stack([[
        100.0 * soft[(r["hire_id"], r["provider_id"])]
        if (r["hire_id"], r["provider_id"]) in soft and r["label"] is not None
        else (GAIN.get(r["label"], 0.0) if r["label"] is not None else 0.0)
        for r in by_query[q]] for q in queries]).astype(np.float32)
    M = np.stack([[r["label"] is not None for r in by_query[q]] for q in queries])
    print(f"queries={len(queries)} features={len(feats)} device={device}")

    # ---- out-of-fold training ------------------------------------------
    oof = {}
    if args.split_seed == 0:
        folds_iter = GroupKFold(n_splits=args.folds).split(queries, groups=queries)
    else:
        folds_iter = KFold(n_splits=args.folds, shuffle=True,
                           random_state=args.split_seed).split(queries)
    for fold, (tr, te) in enumerate(folds_iter, 1):
        tr_q = [queries[i] for i in tr]
        te_q = [queries[i] for i in te]
        P.prior_by_fold(rows, set(tr_q))

        Xtr = torch.tensor(build_x(by_query, feats, tr_q), device=device)
        Xte = torch.tensor(build_x(by_query, feats, te_q), device=device)
        Gtr = torch.tensor(G[[q_index[q] for q in tr_q]], device=device)
        Mtr = torch.tensor(M[[q_index[q] for q in tr_q]], device=device)

        fold_scores = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = SetRanker(len(feats), d=args.d_model, layers=args.layers,
                              dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            n = len(tr_q)
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, args.batch_queries):
                    idx = perm[i:i + args.batch_queries]
                    opt.zero_grad()
                    loss = pairwise_loss(model(Xtr[idx]), Gtr[idx], Mtr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                fold_scores.append(model(Xte).cpu().numpy())
        sc = np.mean(fold_scores, axis=0)
        for k, q in enumerate(te_q):
            for r, s in zip(by_query[q], sc[k]):
                oof[(q, r["provider_id"])] = float(s)

        lists = {q: [r["provider_id"] for r in sorted(by_query[q], key=lambda r: -oof[(q, r["provider_id"])])]
                 for q in te_q}
        m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()},
                                  {k: v for k, v in gt.items() if int(k) in set(te_q)}, ks=(10,))
        print(f"  fold {fold}: NDCG@10={m['ndcg@10']:.4f}")

    lists = {q: [r["provider_id"] for r in sorted(rs, key=lambda r: -oof[(q, r["provider_id"])])]
             for q, rs in by_query.items()}
    m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()}, gt, ks=(5, 10))
    print(f"OOF: P@5={m['precision@5']:.4f} NDCG@5={m['ndcg@5']:.4f} "
          f"P@10={m['precision@10']:.4f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}")
    out = Path(R.RESULTS_DIR) / f"listwise_{args.tag}.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
