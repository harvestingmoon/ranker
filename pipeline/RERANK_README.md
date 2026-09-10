# Stage-2 Re-ranking — cross-encoder + LambdaMART (runbook)

Search-engine shape: **retrieve wide (Stage 1, already built) → rank narrow (Stage 2, this).**

```
query → [refined BM25 ‖ mxbai dense] → RRF → top-50
      → cross-encoder over top-K         (relevance ordering)      rerank_crossencoder.py
      → LambdaMART over all features     (business/structured signals) rerank_ltr.py
      → calibrate → 0–100 JSON           (contract handoff)
```

Frozen baseline to beat: **RRF k=60, NDCG@10 = 0.8918, P@5 = 0.495, MRR = 0.956**.

---

## 1. Environment

```bash
conda activate fi-bench
pip install -r pipeline/requirements-rerank.txt
python -c "import torch, xgboost, sentence_transformers; print(torch.cuda.is_available())"
```

`fi-bench` already has torch 2.11.0+cu130 and sees the RTX 3060 Laptop GPU (6.4 GB).
If `import sentence_transformers` fails after install, it's a transformers-version clash —
`sentence-transformers` may pin `transformers<5`; check with `pip check` before assuming the env is fine.

## 2. Build the feature table (once)

```bash
python pipeline/features.py --top-k 50
```

Writes `features/candidates_top50.csv` (all candidates), `features/train_pairs.csv` (judged only),
and caches embeddings under `cache/`. It also prints the recall-loss line — how many grade≥2
providers fall **outside** the pool, which no re-ranker can recover.

## 3. Cross-encoder, zero-shot (Stage 2a)

```bash
# CPU-friendly default, safe starting point
python pipeline/rerank_crossencoder.py --mode score --tag minilm

# stronger/longer-input option on the 3060
python pipeline/rerank_crossencoder.py --mode score --model Alibaba-NLP/gte-reranker-modernbert-base --batch-size 16 --tag gte --fp16

# heavy option — batch small, expect it to be slow on 6 GB
python pipeline/rerank_crossencoder.py --mode score --model BAAI/bge-reranker-v2-m3 --batch-size 4 --fp16 --tag bge
```

Each run writes `results/ce_<tag>.json` (full ranked lists) and `features/ce_scores_<tag>.csv`
(feature column for the ranker). 130 queries × 50 candidates = 6,500 pairs — seconds on the GPU.

## 4. LambdaMART (Stage 2b) + ablation table

```bash
python pipeline/rerank_ltr.py --mode ablation                    # text signals only
python pipeline/rerank_ltr.py --mode ablation --ce-tag minilm    # + cross-encoder score
```

Prints the ablation (RRF → +CE → +LTR), per-fold NDCG@10 with mean/std, and feature importances.
Predictions are **out-of-fold** (GroupKFold by query), so no query is scored by a model that saw it.

## 5. Optional: fine-tune the cross-encoder (do this last)

```bash
# 5-fold per-query CV: the ce_scores file it writes is OUT-OF-FOLD
python pipeline/rerank_crossencoder.py --mode finetune --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --epochs 2 --cv-folds 5 --tag cecv --quiet
python pipeline/rerank_ltr.py --mode ablation --ce-tag cecv     # re-run with the honest scores
```

**Why CV and not a single split.** An earlier version trained on 80% of queries and evaluated on
the other 20%, then wrote *in-sample* scores for the 104 training queries — which silently made
every downstream LambdaMART number optimistic. Now each fold trains on K−1 folds and scores only
its held-out fold, so **every query's `ce_score` comes from a model that never saw that query**.
The ablation can then be trusted.

**Read this first.** You have **173 grade-3 positives (389 at grade≥2) across 130 queries**.
That is a very small fine-tuning set, so the run prints zero-shot vs fine-tuned OOF over all 130
queries and tells you outright whether to keep zero-shot. Expect the delta to be small; if it is
negative, keep the zero-shot scores.

**Two artefacts, two purposes.** `features/ce_scores_<tag>.csv` is OOF → use it for evaluation.
`models/ce-<tag>/` is trained on *all* pairs → use it for serving (a served query is unseen by
construction, so there is no leak there).

## GPU / VRAM guidance (6.4 GB)

| Model | Params | Suggested | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 23M | batch 32, no fp16 | the control; fast even on CPU |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 149M | batch 16, `--fp16` | Apache-2.0, 8k input — best value |
| `mixedbread-ai/mxbai-rerank-large-v1` | 435M | batch 8, `--fp16` | sibling of your encoder; 512-token input |
| `BAAI/bge-reranker-v2-m3` | 568M | batch 4, `--fp16` | strongest permissive; heaviest |
| `jinaai/jina-reranker-v2-base-multilingual` | 278M | avoid | **CC-BY-NC-4.0 — non-commercial** |

If scores come back NaN, drop `--fp16` first — same failure class as the MPS NaN bug in
`finetune_embeddings.py`.

## Guardrails (these decide whether the numbers are trustworthy)

1. **Split by query, never randomly.** 130 queries; a random split leaks and inflates.
2. **Labels come from `ground_truth_llm.json`**, never the retired taxonomy formula.
3. **Train/serve skew:** `budget_fit` / `seniority_fit` use internal synthetic fields
   (`budget_lo/hi`, `seniority_needed`, `seniority`). Confirm the production gig payload carries
   them *before* shipping a ranker that depends on them — otherwise the model learns on features
   that are missing at serve time.
4. **Significance:** with 130 queries, treat any delta under **~0.02** as noise.
   Report paired bootstrap CIs before claiming a win.
5. **Calibration:** `--export-scores` writes min-max 0–100 per query for the JSON contract.
   The sponsor's merge (`0.6·taxonomy + 0.4·semantic`) uses magnitude, not just order — document
   the normalisation method, which is still an open item with the sponsor.
6. **Keep the plain RRF path.** The developer integrating this has no AI background; the service
   must always be able to return valid JSON without the learned components.

## Outputs

| Path | What |
|---|---|
| `features/candidates_top50.csv` | candidate pool + all features |
| `features/train_pairs.csv` | judged subset (training labels) |
| `features/ce_scores_<tag>.csv` | cross-encoder score per pair (LTR feature) |
| `results/ce_<tag>.json` | CE-reranked lists |
| `results/ltr_scores.json` | calibrated 0–100 semantic scores (with `--export-scores`) |
| `models/ltr_xgb.json`, `models/ce-<tag>/` | trained models |
| `cache/*.npy` | cached embeddings |
