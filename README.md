# Senseigigs — Stage-2 re-ranking for the gig↔showcase matching search

Retrieval (Stage 1) returns candidates; this repo holds the **Stage-2 rankers** that order
them, plus the feature builder and evaluation tooling used to measure both. The pipeline is
trained and evaluated on **`pipeline/data_sat`** — real gig/provider data from the
Scrape-and-Tag pipeline (1,023 gigs × 2,165 providers, 23,873 LLM-graded pairs; see
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

Fine-tune the cross-encoder (5-fold per-query CV — multi-hour on CPU; not yet completed for
`data_sat`, see caveats below):

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
| RRF k=60 (Stage 1 baseline) | 0.671 | 0.116 | 0.532 | 0.774 | 0.278 |
| + cross-encoder, zero-shot (`ms-marco-MiniLM-L6-v2`) | 0.365 | 0.064 | 0.291 | 0.407 | 0.183 |
| **+ LambdaMART (RRF/BM25/dense + budget/seniority/avail, no CE)** | **0.659** | **0.152** | **0.677** | **0.861** | **0.340** |

- **Zero-shot cross-encoder regresses hard** — confirmed out-of-domain for consulting/finance
  text — so its score is **not** used downstream. Fine-tuning under per-query CV was started but
  stopped before completion (multi-hour CPU cost); an open item, not closed.
- **LambdaMART**: NDCG@10 is flat vs. RRF (within the ~0.02 noise threshold this project treats
  as insignificant), but P@5/R@5/R@10/MRR all improve meaningfully — the ranker is better at
  spreading relevant items across the top-10 and reducing misses further down the list, not at
  pulling more items into the very top rank.
- **Why the numbers look low**: 481/1,023 gigs (47.0%) have no provider graded ≥2 anywhere in
  the 2,165-provider catalog — a catalog-coverage gap, not a ranker failure. See
  `pipeline/RERANK_README.md` for four worked (query → ranking) examples that make this concrete,
  including a case where the ranker correctly recovers from a BM25 lexical collision and a case
  where a "regression" is actually pool-sparsity measurement noise.

**Full write-up** — dataset stats, the fixes made to support real data (structured
budget/seniority/availability fields, a date-parsing bug found and fixed during review), feature
importances, worked examples, and caveats — is in `pipeline/RERANK_README.md`.

## Honest caveats

- **47% of gigs have no strong match in the catalog.** This bounds P@K/R@K/NDCG@K below what a
  denser catalog would allow — it's a property of the corpus (see `pipeline/label.md`), not
  something re-ranking can fix.
- **Grades cluster low.** 40.9% grade-0, 46.0% grade-1, 9.7% grade-2, 3.4% grade-3 (grader
  `qwen3.8:27b`, `pipeline/label.md`) — only 3.4% of judged pairs are the strongest positive
  signal, so most of the top-10 for any query is "plausible, not excellent" by construction.
- **Zero-shot cross-encoder is a net negative** on this domain; cross-encoder fine-tuning (which
  the project's own methodology recommends trying next) has not yet been completed here — worth
  revisiting with GPU access.
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
