# Senseigigs — Stage-2 re-ranking for the gig↔showcase matching search

Retrieval (Stage 1) returns candidates; this repo holds the **Stage-2 rankers** that
order them, plus the feature builder and evaluation tooling used to measure them.

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
| `pipeline/features.py` | Feature builder: candidate pool + BM25/dense/RRF scores + budget/seniority/availability signals + labels |
| `pipeline/rerank_crossencoder.py` | Cross-encoder re-ranker: zero-shot scoring, and fine-tuning under **per-query K-fold CV** |
| `pipeline/rerank_ltr.py` | LambdaMART (XGBoost `rank:ndcg`) ranker + ablation report |
| `pipeline/RERANK_README.md` | Runbook: exact commands, VRAM guidance, pitfalls |
| `pipeline/evaluate.py` | P@K / R@K / NDCG@K / MRR (linear gain, relevance bar = score ≥ 40 ⇔ grade ≥ 2) |
| `pipeline/retrieval_*.py` | Stage-1 retrievers (BM25, dense, RRF) |
| `pipeline/data/` | Synthetic corpus + LLM-judged relevance labels |
| `pipeline/features/`, `pipeline/results/` | Built feature tables and frozen result lists (audit trail) |

## Quickstart (CUDA)

```bash
conda create -n fi-bench python=3.12 && conda activate fi-bench   # torch with CUDA
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-rerank.txt   # sentence-transformers, xgboost, scikit-learn, rank-bm25

python pipeline/features.py --top-k 50
python pipeline/rerank_crossencoder.py --mode score --tag minilm      # zero-shot CE
python pipeline/rerank_ltr.py --mode ablation --ce-tag minilm         # RRF → +CE → +LTR
```

Fine-tune the cross-encoder (5-fold per-query CV, ~6 min on an RTX 3060 Laptop):

```bash
python pipeline/rerank_crossencoder.py --mode finetune --cv-folds 5 --epochs 2 --tag cecv --quiet
python pipeline/rerank_ltr.py --mode ablation --ce-tag cecv
```

## Results (130 queries, LLM-judged ground truth, all out-of-fold)

| Stage | NDCG@10 | MRR | P@5 | AP@10 | R@10 |
|---|---|---|---|---|---|
| Stage 1 only — RRF k=60 | 0.8919 | 0.956 | 0.495 | 0.833 | 0.964 |
| + cross-encoder, zero-shot | 0.8615 | 0.950 | 0.477 | 0.800 | 0.938 |
| + cross-encoder, fine-tuned (per-query CV) | 0.8903 | 0.960 | 0.492 | — | — |
| **+ LambdaMART (full stack)** | **0.9062** | **0.976** | **0.508** | **0.864** | **0.976** |

Paired bootstrap over queries (10k resamples), full stack vs RRF baseline:

| Metric | Δ | 95% CI | Verdict |
|---|---|---|---|
| NDCG@10 | +0.0143 | [+0.0034, +0.0254] | improved |
| AP@10 | +0.0315 | [+0.0105, +0.0535] | improved |
| MRR | +0.0192 | [−0.0000, +0.0410] | marginal |
| P@5 | +0.0123 | [−0.0015, +0.0277] | inconclusive |
| R@10 | +0.0115 | [−0.0067, +0.0311] | inconclusive |

Zero-shot MS MARCO rerankers are **out-of-domain** for consulting/finance text and lose to
plain RRF; fine-tuning on in-domain judgments is what makes the cross-encoder useful, and it
only pays once it is fed to LambdaMART as a feature rather than used as the final order.

## Honest caveats

- **P@10 is ceiling-bound, not weak.** There are ~3 grade≥2 providers per gig against 10 slots,
  so the achievable ceiling is **0.299**; the stack sits at 0.290 (97%). Lead with NDCG@10 / MRR / AP@10.
- **The corpus is synthetic** (`data/` from `generate_synthetic_data.py`) — 104 providers, 26
  specialities, 4 providers per speciality. A real catalogue has far more plausible matches per
  query, so these numbers understate performance rather than the reverse.
- **Labels are LLM-judged** and stored as literals in `llm_judgments.py`
  (`LLM_JUDGMENTS` + `LLM_JUDGMENTS_V2_DELTA`, 3,769 judgments over a widened pool). The model,
  prompts and date are not recorded in-repo — treat that as a provenance gap to document.
- Unjudged pairs are counted as irrelevant (TREC convention), which caps recall-style metrics.
- Model weights and embedding caches are git-ignored; rebuild them with the quickstart commands.
