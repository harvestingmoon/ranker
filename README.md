# Senseigigs — gig ↔ provider matching: retrieval + Stage-2 ranking

Given a gig (a hirer's request), return the providers best suited to do it. Stage 1 retrieves
candidates with two text retrievers fused by RRF; Stage 2 re-orders the top 50 with a LambdaMART
ranker that combines the retrieval scores with **SkillsFuture tag-match features**.

The corpus is **`pipeline/data/data_set_v5`** — 1,106 gigs × 3,024 provider showcases, with
**24,070 LLM-graded pairs** (0–3, `qwen3.8:27b`, prompt `rubric_01.v5`). The ranker reads the
same corpus with the SkillsFuture tags attached, **`pipeline/data/data_set_v5_tax`** (identical
text; `tags` added by `enrich_v5_with_tags.py`).

```
gig ─► [refined BM25 ‖ mxbai dense (1024-d)] ─► RRF (k=60) ─► top-50 candidates
    ─► LambdaMART over 17 features:
         retrieval   bm25 / dense / rrf  scores and ranks (scores z-scored per gig)
         tags        tag_level, spec / skill coverage, rarity-weighted skill coverage,
                     soft skill similarity, category match, has-tags flags
    ─► ranked providers (+ 0–100 score)
```

## Results

Out-of-fold (GroupKFold by gig, no gig is scored by a model that saw it), all 1,106 gigs.
Relevance for P/R/MRR is grade ≥ 2. Noise floor at this size ≈ **0.007**.

All rows are scored against the full 24,070-pair label set. "Trained on" says which labels
the model learned from; the clean comparison is between the two rows trained on the original
labels (see "Why the tags help" below).

| System | Trained on | Artifact (`results_data/data_set_v5_tax/`) | NDCG@10 | NDCG@5 | P@5 | P@10 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|
| Stage 1 — RRF k=60 | — | — | 0.8757 | 0.8817 | 0.8745 | 0.8039 | 0.6197 | 0.9698 |
| Stage 2 — LambdaMART, 9 features (previous) | original 21,696 | `v5tax_lin_orig.json` | 0.8766 | 0.8840 | 0.8750 | 0.8037 | 0.6218 | 0.9714 |
| Stage 2 — LambdaMART + tag features | original 21,696 | `v5tax_tags_orig.json` | 0.8954 | 0.9002 | 0.8967 | 0.8288 | 0.6432 | 0.9770 |
| **Stage 2 — LambdaMART + tag features (shipped)** | all 24,070 | `v5tax_tags.json` | **0.8982** | **0.9018** | **0.8991** | **0.8304** | **0.6449** | **0.9774** |

Tags vs previous, both trained on the original labels, paired bootstrap (10k resamples):

| Metric | Δ | 95% CI | Verdict |
|---|---|---|---|
| NDCG@10 | **+0.0188** | [+0.0145, +0.0231] | better — positive on all 3 fold splits (+0.019 / +0.017 / +0.012) |
| P@5 | **+0.0217** | [+0.0150, +0.0286] | better |
| R@10 | **+0.0215** | [+0.0153, +0.0278] | better |
| MRR | +0.0055 | [−0.0001, +0.0115] | crosses 0 |

Full write-up, including the earlier rejected variants: [`ablation.md`](ablation.md).

### Why the tags help, and why an earlier test said they didn't

- **The tags carry signal the text retrievers miss.** Bucket the candidates by tag match
  (none → shared category → shared specialisation → shared specialisation *and* skill): at
  RRF ranks 31–50 the share graded ≥ 2 climbs 27% → 37% → 52% → 74%. Controlling for RRF rank,
  the match level still correlates with the grade (Spearman +0.25).
- **The original labels could not see it.** The judging pool was built from the text
  retrievers' top results, so a provider those retrievers ranked low was usually never graded —
  and `evaluate.py` scores an ungraded pair as grade 0. The tag ranker promotes exactly those
  providers: before re-grading, 13% of its top-10 was ungraded, which read as a −0.04 NDCG@10
  regression (`ablation.md` §2, the "+ taxonomy features" row).
- **So we graded them.** The 2,383 ungraded pairs that any tag variant placed in its top-10 were
  graded with the same model and prompt (a 60-pair re-grade of already-graded pairs reproduced
  **60/60** grades exactly; 9 pairs failed on malformed replies). 67% came back grade ≥ 2.
  They are in `pipeline/data/data_set_v5/judgments_topup_tags.jsonl` and merged by
  `merge_topup_grades.py`. With them, both systems' top-10s are ~99% graded and the tags win.
- **Exact matching is the right default for the hierarchy.** Category (37 values) and
  specialisation (144) are a closed vocabulary. Skills (825) are not: exact string match misses
  near-synonyms, so `skill_soft` (embedding similarity of the skill names) adds a middle tier.
  Industry is left out: a provider's industry is always one of its tag categories.

## Layout

| Path | What |
|---|---|
| `pipeline/data/data_set_v5/` | Corpus + labels: `hirers.json`, `providers.json`, `llm_judgments_merged.json` (0–3), `ground_truth_llm.json` (0/33/67/100), judging pools, `judgments_topup_tags.jsonl` (the 2,374 grades added for the tag test) |
| `pipeline/data/data_set_v5_tax/` | The same corpus with the SkillsFuture `tags` grafted on — what the shipped ranker reads |
| `pipeline/run_pipeline.py`, `retrieval_*.py` | Stage 1: refined BM25, mxbai dense, RRF |
| `pipeline/features.py` | Builds `features_data/<corpus>/candidates_top50.csv` (top-50 pool + retrieval features + labels) |
| `pipeline/tag_features.py` | Builds `tag_features.csv`, the 8 tag-match features. Run in its own process (torch) |
| `pipeline/rerank_ltr.py` | LambdaMART (XGBoost `rank:ndcg`). Shipped flags: `--linear-gain --normalize-scores zscore --tag-features` |
| `pipeline/tag_ablation.py` | The tags-vs-no-tags comparison above, train-on-original / evaluate-on-all |
| `pipeline/merge_topup_grades.py` | Folds the top-up grades into a corpus's label files (idempotent) |
| `pipeline/evaluate.py` | P@K / R@K / NDCG@K / MRR (linear gain; relevant = score ≥ 40 ⇔ grade ≥ 2) |
| `pipeline/final_ablation.py`, `paired_bootstrap.py`, `residual_analysis.py`, `why_recall_low.py` | Stage-1 / Stage-2 ablation harnesses behind `ablation.md` |
| `pipeline/features_data/`, `results_data/`, `models_data/` | Feature tables, out-of-fold ranked lists, trained models, per corpus |
| `pipeline/RERANK_README.md` | Earlier runbook (written for the retired `data_sat` corpus); the cross-encoder notes still apply |

## Reproduce

```bash
pip install -r pipeline/requirements.txt -r pipeline/requirements-rerank.txt
cd pipeline

# Stage 1 + feature table (the tax corpus has the same text, so the same rankings)
python run_pipeline.py --data-dir data/data_set_v5_tax
python features.py     --data-dir data/data_set_v5_tax

# labels: fold in the top-up grades (already merged in the committed files; a no-op then)
python merge_topup_grades.py --data-dir data/data_set_v5_tax

# tag features -- its own process: torch + xgboost in one process segfaults on macOS
python tag_features.py --data-dir data/data_set_v5_tax

# shipped ranker: out-of-fold lists -> results_data/data_set_v5_tax/v5tax_tags.json,
# full-data model -> models_data/data_set_v5_tax/v5tax_tags_xgb.json
python rerank_ltr.py --mode ablation --data-dir data/data_set_v5_tax --top-k 50 \
    --linear-gain --normalize-scores zscore --tag-features --tag v5tax_tags

# the significance test in the Results section
python tag_ablation.py
```

## Caveats

- **The top-up grades were chosen by the tag rankers.** They cover every ungraded pair in the
  tag variants' top-10s (on the original fold split), so they cannot favour the baseline — but
  on other fold splits a few ungraded pairs remain (top-10 still ~99% graded). Training on the
  top-up as well changes what the 9-feature baseline learns and pulls ungraded providers into
  its top-10 (94% graded), so that comparison is reported in `tag_ablation.py` but not used
  for the headline.
- **The labels are LLM grades, not hirer outcomes.** One model (`qwen3.8:27b`) at temperature 0,
  on text only — the grader never saw the tags, so the tags cannot be "agreeing with
  themselves".
- **Numbers before this change are on the original 21,696 labels.** The new grades raise the
  ideal DCG for gigs where they found strong providers, so every system's NDCG@10 drops a
  little on the larger label set (RRF 0.8861 → 0.8757). Compare rows only within one label set.
- **The remaining ~1% ungraded top-10 slots cannot flip the result**: scoring every ungraded
  provider in the baseline's top-10 as grade 3, and every one only in the tag ranker's as
  grade 0, still gives the tags +0.0122 NDCG@10 [+0.0080, +0.0165].
- **Pre-existing, unchanged: the BM25 query counts the gig title 4×, not the documented 3×**
  (`hirer_text()` already contains the title, and `hirer_bm25_tokens()` adds it 3× more). It
  applies to every system equally; fixing it would change every Stage-1 number, so it is left
  as is and flagged here.
- **`budget_fit`, `seniority_fit`, `avail_immediacy` are dead on v5** (the schema carries none of
  their inputs) and score 0 importance (so does `exp_years`, which `features.py` writes but no model reads). They are
  kept so the feature contract stays stable.
- **Untagged records**: 87 gigs and 252 providers have no tags; `has_tags_*` lets the model tell
  "untagged" from "mismatched", and those records fall back to retrieval-only ordering.
- Stage-1 ranked lists, the embedding cache and the skill-vector cache are git-ignored and
  regenerate from the commands above.
