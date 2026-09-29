# Senseigigs — Stage-2 re-ranking for the gig↔showcase matching search

Retrieval (Stage 1) returns candidates; this repo holds the **Stage-2 rankers** that order
them, plus the feature builder and evaluation tooling used to measure both. The pipeline is
trained and evaluated on **`pipeline/data_sat`** — real gig/provider data from the
Scrape-and-Tag pipeline (1,023 gigs × 2,165 providers, 22,289 LLM-graded pairs; see
`pipeline/label.md`).

Pipeline shape:

```
gig query → [refined BM25 ‖ mxbai dense] → RRF → top-50 candidates
          → cross-encoder score (Stage 2a)
          → LambdaMART over [text scores + business signals + ce_score] (Stage 2b)
          → 0-100 calibrated scores
```

## Layout

| Path | What |
|---|---|
| `pipeline/data_sat/` | Real corpus + LLM-graded relevance labels (the dataset the results below use) |
| `pipeline/label.md` | How `data_sat` was built and graded — pool construction, grader model, caveats |
| `pipeline/import_scrape_and_tag.py`, `pipeline/build_judging_pools_sat.py` | Build `data_sat/` from raw Scrape-and-Tag CSVs and generate its judging pools |
| `pipeline/features.py` | Feature builder: candidate pool + BM25/dense/RRF scores + budget/seniority/availability signals + labels |
| `pipeline/rerank_crossencoder.py` | Cross-encoder re-ranker: zero-shot scoring, and fine-tuning under **per-query K-fold CV** |
| `pipeline/rerank_ltr.py` | LambdaMART (XGBoost `rank:ndcg`) ranker + ablation report |
| `pipeline/finetune_embeddings.py` | Dense embedding fine-tune — Matryoshka-aware loss, CUDA, collapse guard |
| `pipeline/RERANK_README.md` | Runbook: exact commands, VRAM guidance, pitfalls, worked query examples |
| `pipeline/evaluate.py` | P@K / R@K / NDCG@K / MRR (linear gain, relevance bar = score ≥ 40 ⇔ grade ≥ 2) |
| `pipeline/retrieval_*.py` | Stage-1 retrievers (BM25, dense, RRF) |
| `pipeline/features_data_sat/`, `pipeline/results_data_sat/*`, `pipeline/models_data_sat/` | Built feature tables, result lists and trained model for `data_sat` |
| `pipeline/data/`, `pipeline/results/`, `pipeline/features/`, `pipeline/models/` | Archived synthetic-data baseline — kept for reference only, not covered below; see git history |

## Quickstart (CUDA)

```bash
conda create -n fi-bench python=3.12 && conda activate fi-bench   # torch with CUDA
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-rerank.txt   # sentence-transformers, xgboost, scikit-learn, rank-bm25

python pipeline/run_pipeline.py --data-dir data_sat                       # Stage 1: BM25/dense/RRF
python pipeline/features.py --data-dir data_sat --top-k 50
python pipeline/rerank_crossencoder.py --mode score --data-dir data_sat --tag minilm   # zero-shot CE
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --top-k 50           # RRF → +LTR
```

Fine-tune the cross-encoder (5-fold per-query CV, ~65 min on an RTX 3060 — completed for
`data_sat`; the OOF scores are kept for measurement, but the CE is deliberately **not** a ranker
feature, see Results):

```bash
python pipeline/rerank_crossencoder.py --mode finetune --data-dir data_sat --cv-folds 5 --epochs 2 --tag cecv --quiet
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --ce-tag cecv
```

Every command above also runs against the archived synthetic corpus by omitting `--data-dir`
(defaults to `data`).

## Results

RRF top-50 → LambdaMART, evaluated out-of-fold (GroupKFold by query) against `data_sat`'s
1,023 LLM-graded queries:

| Stage | NDCG@10 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| RRF k=60 (Stage 1 baseline) | **0.671** | 0.116 | 0.532 | 0.774 | 0.278 |
| + cross-encoder, zero-shot (`ms-marco-MiniLM-L6-v2`) | 0.365 | 0.064 | 0.291 | 0.407 | 0.183 |
| + cross-encoder, fine-tuned (per-query 5-fold CV, out-of-fold) | 0.555 | 0.116 | 0.548 | 0.728 | 0.295 |
| **+ LambdaMART, no CE — the shipped ranker** | 0.658 | **0.153** | **0.682** | **0.860** | 0.338 |
| + LambdaMART with `ce_score` — trained, **rejected** | 0.633 | 0.154 | 0.705 | 0.854 | **0.363** |

Paired bootstrap over the 1,023 queries (10k resamples) — shipped ranker vs RRF:

| Metric | Δ | 95% CI | Verdict |
|---|---|---|---|
| P@5 | **+0.0375** | [+0.0309, +0.0444] | improved |
| R@10 | **+0.0870** | [+0.0601, +0.1144] | improved |
| MRR | **+0.0606** | [+0.0471, +0.0742] | improved |
| NDCG@10 | **−0.0133** | [−0.0230, −0.0035] | **worse — significant** |

- **The cross-encoder is now fine-tuned on `data_sat`** (per-query 5-fold `GroupKFold`, 2 epochs,
  ~65 min on an RTX 3060, OOF scores only so it stays leak-free). It improves hugely over
  zero-shot — **0.365 → 0.555 NDCG@10 (+0.190)** — but is still below RRF standalone.
- **The CE is deliberately NOT a ranker feature.** Adding `ce_score` to LambdaMART costs
  **−0.0246 NDCG@10 [−0.0349, −0.0141]** while buying **+0.0259 MRR [+0.0123, +0.0396]**. It is
  the single highest-importance feature (0.25 mean gain) *and* it makes the top-10 worse — the
  model leans on it hard, and that lean is mispriced. On the synthetic corpus the same feature
  *helped* (+0.0143); **that result did not replicate on real data**, so it is excluded here.
  Shipped ranker: `models_data_sat/noce_xgb.json` (9 features). The fine-tuned CE weights are
  kept in `models_data_sat/ce-cecv/` for the semantic-score role, not as a ranking feature.
- **The ranker's real win is coverage, not top-of-list quality.** P@5/R@5/R@10/MRR all improve
  significantly (spreading relevant items across the top-10); NDCG@10 — which weights *how good*
  the items at the very top are — is **significantly worse than RRF**. An earlier revision called
  this "flat"; with CIs it is not flat, it is a small real regression.
- **Zero-shot cross-encoder regresses hard** — confirmed out-of-domain for consulting/finance
  text (0.365 vs 0.671), so its score is not used anywhere downstream.

**Full write-up** — dataset stats, the fixes made to support real data (structured
budget/seniority/availability fields, a date-parsing bug found and fixed during review), feature
importances, worked examples, and caveats — is in `pipeline/RERANK_README.md`.

## Honest caveats

- **47% of gigs have no graded-≥2 provider _in their pool_.** 481/1,023 gigs have no provider
  graded ≥2 among the ~22 pooled for that gig. Only ~1% of the 2,165-provider catalog is graded
  for any given gig, so this is a **pool-coverage** statement, not a claim that the catalog lacks
  a strong match.
- **Recall is undefined for 47% of gigs.** `evaluate.recall_at_k` returns `None` when a gig has no
  relevant item at all, so every R@5/R@10 in the table above is averaged over only the **542
  answerable gigs**, not all 1,023. NDCG@10/P@5/MRR average over all 1,023.
- **The noise threshold is ~0.007 on this corpus, not 0.02.** `rerank_ltr.py` prints a "treat
  deltas under ~0.02 as noise" reminder that was calibrated for the *130-query synthetic* corpus.
  At 1,023 queries the standard error is ~2.8× smaller, so deltas near 0.01 are real here. Use the
  bootstrap CIs above, not the printed reminder.
- **Grades cluster low.** 50.8% grade-0, 43.7% grade-1, 2.5% grade-2, 3.0% grade-3 (grader
  `qwen3.8:27b`, `pipeline/label.md`) — only 5.5% of judged pairs are ≥2, so most of the top-10
  for any query is "plausible, not excellent" by construction.
- **Zero-shot cross-encoder is a net negative** on this domain, and the fine-tuned one is a
  negative *as a ranker feature*. Both are measured, not assumed.
- **`budget_fit` penalizes below-budget rates as harshly as above-budget ones** — a cheaper-than-
  requested provider gets demoted the same as an over-budget one, which is a plausible source of
  some of LambdaMART's query-level regressions (see the worked example in `RERANK_README.md`).
  Not fixed in this run.
- **Pre-existing, not `data_sat`-specific**: `retrieval_bm25.py`'s query-side title weighting
  effectively 4×s the title instead of the documented 3× (a token-concatenation bug in
  `BM25Retriever._query_tokens`). Affects every BM25/RRF run in this repo. Flagged in
  `RERANK_README.md`, not fixed, since fixing it changes results for both datasets.
- **Only one seed / one fold split.** No bootstrap CIs yet on the LambdaMART deltas — treat the
  P@5/R@5/MRR gains as promising, not statistically proven.
- Model weights and embedding/results caches are git-ignored (regenerable, and large at this
  corpus scale); rebuild with the quickstart commands.
- An archived synthetic-data baseline and a Matryoshka-aware dense-encoder fine-tune validated
  against it are preserved in `pipeline/data/`, `results/`, `features/`, `models/` and
  `pipeline/finetune_embeddings.py` for reference — not documented here; see git history for
  their numbers and methodology.
