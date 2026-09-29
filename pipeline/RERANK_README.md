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

---

## Real-data run — `data_sat` (2026-09-29)

Every command above also accepts `--data-dir data_sat` (added when this run was done) and
namespaces its outputs (`features_data_sat/`, `results_data_sat/`, `models_data_sat/`,
`cache_data_sat/`) so this run never touches the frozen synthetic baseline above.

**Scale.** 1,023 hirers × 2,165 providers (vs. 130 × 104 synthetic), 23,873 LLM-graded pairs
total (grader `qwen3.8:27b`, see `label.md`), 19,193 of those inside the top-50 RRF pool used
for training. Grades cluster low: 40.9% grade-0, 46.0% grade-1, 9.7% grade-2, 3.4% grade-3.
Measured directly against this run's `ground_truth_llm.json` (not `label.md`'s figure, which
describes the earlier, superseded 774-gig `data_sat` before the data team backfilled
budget/seniority/availability): **481/1,023 hirers (47.0%) have no provider graded ≥2 at all**
(RELEVANCE_THRESHOLD=40 in `evaluate.py`). `evaluate_all_hirers` excludes these from the recall
average (recall is undefined with zero relevant items) but scores them as a hard 0 in NDCG/MRR —
so R@5/R@10 above are averaged over the ~542 hirers with a real match, while NDCG@10/MRR are not.
A hard ceiling on P@K/NDCG@K/R@K is expected and is a property of the corpus, not the ranker.

**Fields resolved.** `budget_lo/hi`, `seniority_needed` (hirers) and `rate_per_hour`, `seniority`,
`available_from`, `availability` (providers) are now present inline on every record — the earlier
version of `data_sat` was missing these (see `label.md`), which had paused this retrain. Since
`data_sat` carries them directly (no separate `_with_taxonomy.json`), `features.py` now falls
back to the main `providers.json`/`hirers.json` when that file is absent.

`avail_immediacy()` was changed to use the structured `available_from` (provider) vs. `start_by`
(hirer) date gap when present, instead of matching synthetic-only phrases like "immediately"
against `data_sat`'s "Available from `<date>`, N days a week" strings, which never match and
would otherwise silently collapse to a constant 0.5 for every real record. That first version of
the fix still had a bug caught during a review pass before push: `available_from` and `start_by`
aren't always ISO dates — 966/2,165 providers use the literal sentinel `"now"` and 75/1,023
hirers use `"asap"` — and unhandled `date.fromisoformat()` on those raised `ValueError`, which
fell through to the same free-text matcher (which also doesn't recognise "Available now, ..."),
landing back on the 0.5 default for **47.5% of all candidate rows**. Both sentinels now resolve
to `date.today()` before comparison; the fallback-default rate dropped to 0.43% of rows (genuine
missing/unparseable dates only), and `avail_immediacy`'s LambdaMART feature importance moved from
0.054 (barely above noise) to 0.085 (5th of 9 features) once it carried real signal.

**Results:**

| Stage | NDCG@10 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| RRF k=60 (Stage 1 baseline) | 0.6710 | 0.116 | 0.532 | 0.774 | 0.278 |
| + cross-encoder, zero-shot (`ms-marco-MiniLM-L6-v2`) | 0.3649 | 0.064 | 0.291 | 0.407 | 0.183 |
| + LambdaMART (RRF/BM25/dense + budget/seniority/avail, no CE) | 0.6591 | 0.152 | 0.677 | 0.861 | 0.340 |

- **Zero-shot cross-encoder regresses hard** on real data (NDCG@10 0.671 → 0.365) — confirmed
  out-of-domain for consulting/finance text, much more visibly than on the synthetic corpus.
  Per the project's own "zero-shot first, fine-tune only if the gain is real" rule, its score was
  **not** carried into LambdaMART as a feature. Fine-tuning under per-query 5-fold CV was started
  but deliberately **skipped/stopped** for this run (project-owner call, given the zero-shot
  regression and the multi-hour CPU cost) — this stays a documented open item, not a closed one.
- **LambdaMART on structured + retrieval features alone**: NDCG@10 is flat vs. RRF (0.659 vs
  0.671 — inside the project's own ~0.02 noise threshold), but P@5/R@5/R@10/MRR all improve
  meaningfully. Read this as: the learned ranker isn't pulling more relevant items into the very
  top of a tied NDCG score, it's doing a better job spreading relevant items across the top-10 and
  reducing misses lower in the list — consistent with 1,023 queries giving GroupKFold much more
  signal than the synthetic run's 130.
- **Feature importance** (mean over 5 folds): `rrf_rank` 0.223, `rrf_score` 0.164, `dense_rank`
  0.154, `seniority_fit` 0.116, `avail_immediacy` 0.085, `budget_fit` 0.085, `dense_cosine` 0.070,
  `bm25_score` 0.056, `bm25_rank` 0.048. The three structured fields the earlier `data_sat`
  couldn't compute (`seniority_fit`, `budget_fit`, `avail_immediacy`) rank 4th, 5th and 6th —
  real, non-trivial weight, which is the main payoff of getting those fields backfilled.

**Worked examples** (inspected directly against `results_data_sat/rrf_k60.json` /
`results_data_sat/ltr_sat.json` / `data_sat/ground_truth_llm.json`; regenerate `results_data_sat/`
with `python pipeline/run_pipeline.py --data-dir data_sat` to reproduce these):

- **Why the aggregate numbers look low even on a "success" query** (hire_id=1, "Design AI Agent
  Sales Support System for Life Insurer"): 3 grade-3 matches exist among 2,165 providers. Both
  RRF and LambdaMART surface all 3 within the top-6–9, but the other 7-of-10 slots are grade-1
  "in the neighborhood but not it" profiles — with only 3 truly-relevant providers in the whole
  catalog, P@5≈0.15 is close to the ceiling even when retrieval is working correctly.
- **A genuine LambdaMART win, and why**: hire_id=745 ("Design RCT for Refugee Cash Transfer
  Program") — RRF's top-5 is all grade-0, including two "Transfer *Pricing* Specialists": a pure
  BM25 lexical collision on the word "Transfer" (cash *Transfer* program vs. *Transfer* pricing),
  unrelated in meaning. LambdaMART's top-3 are all genuinely relevant (poverty/social-protection/
  policy-evaluation specialists) — dense embeddings + structured features override the bad
  lexical match. NDCG@10: 0.095 → 0.917.
- **A "loss" that is measurement noise, not a real regression**: hire_id=14 ("Assess Acoustic
  Performance for Staff Village Building") has exactly **one judged pair in its entire pool**
  (provider 1926, grade-1/score=33). RRF happened to rank it #1 (NDCG@10=1.0); LambdaMART ranked
  it #10 (NDCG@10=0.289). With a single graded item, NDCG@10 swings on one rank position — this
  is pool-sparsity variance, not evidence LambdaMART got the query wrong.
- **A genuine LambdaMART weakness**: hire_id=247 ("Consolidate Contract Playbooks into Single
  Storage", budget $160-215/hr) — RRF ranked grade-2 match provider 252 at #3; LambdaMART demoted
  it to #17. Provider 252's rate ($135/hr) sits *below* the hirer's budget floor, and `budget_fit`
  penalizes any rate outside `[budget_lo, budget_hi]` symmetrically — but a cheaper-than-budget
  provider is usually still a fine match, not a mismatch. `budget_fit`'s below-band penalty should
  likely be softer than its above-band penalty; this is a plausible explanation for the flat
  NDCG@10 in aggregate (LambdaMART fixes some BM25 failures like the case above while introducing
  some new ones like this). Not changed in this run -- noted as a follow-up, not fixed silently.

**Caveats specific to this run** (in addition to the general ones in `label.md`):
- Only one seed / one fold split was run — no bootstrap CI yet on the LambdaMART deltas, so per
  guardrail #4 above, treat the P@5/R@5/MRR gains as promising, not proven.
- Cross-encoder fine-tuning remains untried on this corpus. Given 19,193 judged pairs (vs. a few
  hundred on synthetic), it has a much better chance of working than the synthetic run's fine-tune
  did — worth revisiting with GPU access rather than CPU-only.
- **Pre-existing bug, not introduced by this run**: `retrieval_bm25.py`'s query-side field
  weighting effectively 4×-weights the title instead of the documented 3× (`title_weight=3`).
  `BM25Retriever._query_tokens` rebuilds query tokens via `hirer_bm25_tokens(h_like, ...)`, where
  `h_like["hire_description"]` is set to the *already-concatenated* `hirer_text(h)` (which itself
  includes the title) rather than the raw `hire_description` field — so the title's tokens are
  counted once via `title_tokens * title_weight` and once more inside `body_tokens`. This affects
  every BM25/RRF run in this repo, synthetic and `data_sat` alike (it predates this session), so
  it does not change the *relative* comparisons above, but it means "refined BM25" is stronger
  than its own documentation claims. Flagged here, not fixed here — fixing it would change the
  frozen synthetic baseline numbers this README already documents elsewhere, which needs its own
  decision, not a silent edit inside a data_sat retrain.
- `--export-scores` (calibrated 0–100 contract output) was not run for this dataset yet.
