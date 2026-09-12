# Senseigigs — Stage-2 re-ranking for the gig↔showcase matching search

Retrieval (Stage 1) returns candidates; this repo holds the **Stage-2 rankers** that order
them, an optional **dense-encoder fine-tune** that upgrades Stage 1, plus the feature builder
and evaluation tooling used to measure both.

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
| `pipeline/finetune_embeddings.py` | Dense embedding fine-tune — Matryoshka-aware loss, CUDA, collapse guard |
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

Fine-tune the dense encoder (Matryoshka-aware, ~25 s on an RTX 3060 Laptop):

```bash
python pipeline/finetune_embeddings.py --epochs 3 --batch-size 8 --grad-accum 2 --out-name mxbai-finetuned
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

## Dense embedding fine-tune (Matryoshka-aware)

`finetune_embeddings.py` contrastively fine-tunes the dense encoder on
(gig, best provider, same-category-but-wrong-speciality provider) triplets. The base
encoder was pretrained with Matryoshka Representation Learning and the dense leg depends
on that (256-dim default, 64–1024 sweep), so the loss is wrapped in `MatryoshkaLoss` — a
plain `MultipleNegativesRankingLoss` would silently invalidate truncation after
fine-tuning, making a truncated fine-tuned model incomparable with a truncated base model.

| | |
|---|---|
| Data | 111 training triplets, 19 held out — no new labels |
| Setup | 3 epochs, effective batch 16, lr 5e-6, `MatryoshkaLoss(n_dims_per_step=1)`, warmup 10%, grad-clip 1.0 |
| Cost | 24 s on an RTX 3060 Laptop, peak VRAM 5.20 GB |
| Healthy run | loss 0.99 → 0.17; embedding spread 0.5625 → 0.4238 (more discriminative) |

Dense leg only, scored against the LLM ground truth (130 Hirers):

| Variant | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|---|
| base 1024 (mxbai) | 0.472 | 0.830 | 0.839 | 0.275 | 0.941 | 0.865 | 0.948 |
| fine-tuned 1024 | 0.498 | 0.876 | 0.858 | 0.275 | 0.930 | 0.860 | 0.955 |
| base 256 (MRL trunc.) | 0.425 | 0.751 | 0.776 | 0.252 | 0.868 | 0.804 | 0.917 |
| **fine-tuned 256** | **0.477** | **0.844** | **0.839** | **0.275** | **0.928** | **0.856** | **0.958** |

Paired bootstrap over queries (10k resamples):

| Comparison | Metric | Δ | 95% CI | Verdict |
|---|---|---|---|---|
| FT-256 vs base-256 | NDCG@10 | +0.0520 | [+0.0214, +0.0816] | improved |
| | AP@10 | +0.0936 | [+0.0423, +0.1450] | improved |
| | P@5 | +0.0523 | [+0.0200, +0.0862] | improved |
| | R@10 | +0.0601 | [+0.0154, +0.1049] | improved |
| | MRR | +0.0415 | [−0.0011, +0.0858] | marginal |
| FT-1024 vs base-1024 | NDCG@10 | −0.0055 | [−0.0316, +0.0199] | inconclusive |
| | P@5 | +0.0262 | [+0.0015, +0.0508] | improved |
| | R@10 | −0.0103 | [−0.0442, +0.0214] | inconclusive |
| FT-256 vs base-1024 | NDCG@10 | −0.0098 | [−0.0361, +0.0156] | inconclusive |
| | AP@10 | +0.0255 | [−0.0188, +0.0704] | inconclusive |

**Read this as: fine-tuning redistributes quality down the Matryoshka ladder rather than
adding quality at the top.** At 1024 dims it is a wash; at 256 dims it is a large,
significant gain; and the 256-dim fine-tuned model is statistically indistinguishable from
the 1024-dim base model — 4× smaller embeddings at full-dimension quality. Quantitatively:
the base model kept 93% of its own full-dim NDCG@10 at 256 dims (0.804/0.865), the
fine-tuned model keeps 99.5% (0.856/0.860).

Still below RRF (0.892) as a standalone leg, so this only matters if it lifts the fused
Stage-1/Stage-2 stack — which has not been re-run with the fine-tuned encoder yet.

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
- **The embedding fine-tune trains on the taxonomy-formula labels** (`ground_truth.json`), not the
  LLM grades — the same formula-vs-text circularity this project criticises elsewhere. It
  generalised to the LLM ground truth anyway, but the training label source has to be stated.
  Switching it to `llm_judgments_merged.json` (grade-3 positives, grade≤1 same-category hard
  negatives) is the follow-up that makes training labels match evaluation labels.
- **Single seed, 111 triplets.** The bootstrap CIs cover query-sampling variance, not training
  variance; seed averaging is not done yet.
- Hyperparameters for the embedding fine-tune (lr, warmup, clipping, `n_dims_per_step`) were chosen
  by a **collapse criterion**, not by these metrics — three earlier runs collapsed the encoder and
  the failure modes are documented in the commit history and in-line in the script.
- Model weights and embedding caches are git-ignored; rebuild them with the quickstart commands.
