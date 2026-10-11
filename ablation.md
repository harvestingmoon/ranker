# Ablation — Stage-1 retrieval and Stage-2 re-ranking

**Corpus:** `pipeline/data/data_set_v5` — 1,106 gig queries, 3,024 provider showcases,
**21,696 LLM-graded pairs**, graded 0–3 by `qwen3.8:27b` on prompt `rubric_0_3.v5`
(0: 28.1% · 1: 10.5% · 2: 44.6% · 3: 16.9%). Relevance for P/R/MRR is **grade ≥ 2**,
so **61.5%** of judged pairs are relevant, median **13** relevant per query.

Reproduce with:

```bash
python pipeline/run_pipeline.py   --data-dir data/data_set_v5
python pipeline/features.py       --data-dir data/data_set_v5
python pipeline/rerank_ltr.py     --data-dir data/data_set_v5 \
                                  --linear-gain --normalize-scores zscore --tag v5_lin
python pipeline/final_ablation.py
```

All metrics are **out-of-fold** (`GroupKFold` by query — no query is scored by a model
that saw it). Significance is a **paired bootstrap over queries** (10,000 resamples);
the noise floor at n=1,106 is ≈ **0.007**.

---

## 1. Main results

| system | P@1 | P@5 | P@10 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| Stage 1 — BM25 (refined) | 0.9159 | 0.7937 | 0.6683 | 0.5693 | 0.7717 | 0.9466 |
| Stage 1 — Dense (mxbai-embed-large, 1024-d) | 0.9340 | 0.8420 | 0.7155 | 0.6157 | 0.8160 | 0.9597 |
| Stage 1 — RRF k=10 | 0.9439 | 0.8727 | 0.7980 | 0.6855 | 0.8812 | 0.9682 |
| Stage 1 — RRF k=60 | 0.9485 | 0.8745 | **0.8038** | 0.6887 | 0.8861 | 0.9698 |
| Stage 1 — RRF k=100 | 0.9494 | 0.8727 | 0.8025 | 0.6870 | 0.8848 | 0.9703 |
| **Stage 2 — LambdaMART (9 feat)** | **0.9548** | 0.8756 | 0.8020 | **0.6896** | **0.8867** | 0.9724 |
| Stage 2 — + industry features | 0.9575 | 0.8812 | 0.7826 | 0.6751 | 0.8729 | 0.9740 |
| Stage 2 — + taxonomy features (15 feat) | 0.9584 | 0.8807 | 0.7451 | 0.6459 | 0.8473 | 0.9746 |
| Stage 1 + industry recall (w=1.0) | 0.8626 | 0.7986 | 0.7149 | 0.6184 | 0.7952 | 0.9206 |
| Stage 1 + taxonomy recall (w=1.0) | 0.8969 | 0.7745 | 0.6611 | 0.5781 | 0.7526 | 0.9387 |
| Stage 1 + taxonomy recall (w=0.3) | 0.9584 | **0.8859** | 0.7883 | 0.6794 | 0.8756 | **0.9749** |
| *ORACLE (ceiling)* | *0.9991* | *0.9856* | *0.9365* | *0.8081* | *1.0000* | *0.9991* |

**Shipped configuration: row 6** — refined BM25 ⊕ dense → RRF(k=60) → LambdaMART,
9 features, `rank:ndcg` with linear gain and per-query z-score normalisation.
It is the best NDCG@10 and every alternative is either a significant regression or a
non-significant trade.

---

## 2. Significance vs the shipped Stage-2 baseline

| variant | metric | Δ | 95% CI | verdict |
|---|---|---|---|---|
| + industry features | NDCG@10 | −0.0138 | [−0.0173, −0.0102] | **worse** |
| | P@5 | +0.0056 | [+0.0005, +0.0107] | better |
| | P@10 | −0.0193 | [−0.0245, −0.0143] | **worse** |
| | R@10 | −0.0145 | [−0.0197, −0.0094] | **worse** |
| | MRR | +0.0016 | [−0.0026, +0.0059] | crosses 0 |
| + taxonomy features (15 feat) | NDCG@10 | −0.0394 | [−0.0443, −0.0345] | **worse** |
| | P@5 | +0.0051 | [−0.0014, +0.0118] | crosses 0 |
| | P@10 | −0.0569 | [−0.0637, −0.0503] | **worse** |
| | R@10 | −0.0436 | [−0.0500, −0.0371] | **worse** |
| | MRR | +0.0022 | [−0.0032, +0.0077] | crosses 0 |

Every recall-fusion row is significantly worse than the baseline on NDCG@10.

---

## 3. Findings

**1. Hybrid fusion is the largest single win.** BM25 0.7717 → dense 0.8160 →
RRF 0.8861 NDCG@10. Fusing the two ranked lists is worth **+0.070 NDCG@10** over the
better single retriever, and it improves every metric. `k=60` is the optimum; k=10 is
worse (0.8812) and k=100 is level (0.8848).

**2. Stage 2 does not earn its place on this corpus.** The LambdaMART beats RRF by
+0.0006 NDCG@10, +0.0011 P@5, +0.0026 MRR — and **every paired CI crosses zero**
against a 0.007 noise floor. Feature importance sits **80% on `rrf_rank` + `rrf_score`**,
i.e. the model largely re-expresses the retriever. This is a consequence of the corpus
shape: with ~13 relevant providers per query out of a 50-candidate pool, ~26% of the
pool is relevant, so almost any ordering scores well and there is very little for a
ranker to fix.

Reproduce that check from the artifact `results_data/data_set_v5/v5_lin.json`.

**3. The third (taxonomy) recall channel adds nothing, because there is nothing to add.**
`run_pipeline.py` calls `rrf_fuse(sparse, dense, k=k)` — the taxonomy channel is never
passed, although `retrieval_rrf.rrf_fuse` has always supported it. Wiring it in was
tested here and:

```
candidates the third channel adds: 0 per query in the pool, ~1–2 inside the top-10
```

BM25 and dense already rank **all 3,024 providers**, so the candidate pool is
exhaustive and a third channel can only reorder. Reordering loses: at w=1.0 the
taxonomy fusion costs **−0.13 NDCG@10**; at w=0.3 it costs −0.011.

**4. A coarse label makes a degenerate channel.** `data_set_v5` ships only an
`industry` field (36 values). Mapped onto the taxonomy category axis it produces a
**binary** feature — one distinct value in the whole file:

```
tax_category    distinct: ['1.000000']    444,104 rows
taxonomy_score  distinct: ['0.100000']    444,104 rows
largest tied group per query: median 660
```

~660 candidates tie at the same score, so RRF converts an arbitrary tie order into
ranks 1…660 and industry-matched noise floods the top-10. Cosine similarity would not
fix this — 2,772 providers share 36 vectors, so the ties survive.

**5. The full SkillsFuture tree was recovered and still did not help.** v5 dropped the
taxonomy, but the `mcjuh/BT4103-Scrape-and-Tag` records carry the complete `sf_v1`
tree at 100% coverage and join to v5 by `source_file` with zero loss (1,106/1,106 gigs,
3,024/3,024 providers). `enrich_v5_with_tags.py` grafts it on, giving 33/37 categories,
103/134 specialisations, 108/143 search tags and **500/745 skills**. The features are
genuinely graded (means 0.11–0.78) and the model uses them at ~30% importance — yet
NDCG@10 falls **−0.0394** (significant). The signal is real but redundant: it trades
graded ordering for head membership (P@5 and MRR up, NDCG@10/P@10/R@10 down).

**6. Recall is capped by K, not by the system.** With a median of 13 relevant items per
query, a top-5 list can hold at most 5 of them, so R@5 is arithmetically capped at
≈ 0.385 — and the system sits at 84.6% of the *oracle* ceiling:

| K | observed | oracle | % of achievable |
|---|---|---|---|
| 5 | 0.3924 | 0.4638 | 84.6% |
| 10 | 0.6896 | 0.8081 | 85.3% |
| 20 | 0.9268 | 0.9998 | 92.7% |
| 50 | **0.9908** | 1.0000 | **99.1%** |

Report the curve, never R@5 alone.

**7. Three features are structurally dead.** `budget_fit`, `seniority_fit` and
`avail_immediacy` read `budget_lo/hi`, `rate_per_hour`, `seniority_needed`,
`seniority` and `availability` — none of which the v5 schema carries. They contribute
`0.0000` importance in every run. They are retained so the column contract is stable,
but they carry no information on this corpus.

---

## 4. Caveats

- **Out-of-fold only.** The served model is trained on all queries
  (`models_data/data_set_v5/v5_lin_xgb.json`); the reported numbers come from OOF
  predictions (`results_data/data_set_v5/v5_lin.json`). Do not mix them.
- **The exported ranked lists are top-50**, so R@50 here is measured on the top-50
  only; deep recall is higher.
- **One query in 1,106 has no relevant item**, so `recall_at_k` returns `None` for it;
  all recall figures average over the 1,105 queries where recall is defined.
- **`data_set_v5_tax` shares v5's text byte-for-byte** — only `tags` was added — so its
  BM25/dense rankings are identical to v5's and are reused rather than recomputed.
- The recall-fusion rows were rebuilt from saved ranked lists, which is valid because
  RRF consumes **ranks only**. The 2-channel rebuild was asserted to reproduce the
  saved `rrf_k60.json` exactly before any 3-channel delta was reported.
