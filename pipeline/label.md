# LLM relevance labels for the Scrape-and-Tag corpus

The Stage-2 rerankers were built and evaluated on the synthetic corpus in `pipeline/data/`
(130 gigs × 104 providers). This is the labelled real-data set that replaces it: gigs and
provider profiles scraped and extracted by the BT4103-Scrape-and-Tag pipeline, with every
(gig, candidate provider) pair graded 0–3 by an LLM.

Everything lives in `pipeline/data_sat/`. The synthetic `data/` folder and every result
built on it are untouched.

## Summary

| | |
|---|---|
| Gigs | 774 |
| Providers (search corpus) | 1,696 |
| Graded pairs | 23,873 (every pair in every gig's pool) |
| Pool size per gig | 25–39, median 31 |
| Grader | `qwen3.8:27b`, prompt `rubric_0_3.v1`, temperature 0 |
| Grades | 0: 9,767 (40.9%) · 1: 10,979 (46.0%) · 2: 2,313 (9.7%) · 3: 814 (3.4%) |
| Gigs with no provider graded ≥ 2 | 262 (34%) |

## Pipeline

```
BT4103-Scrape-and-Tag/docs/{hirers,providers}.csv
  │  pipeline/import_scrape_and_tag.py          filter + convert  → hirers.json, providers.json, id_map.json
  ▼
  │  pipeline/build_judging_pools_sat.py        pick pairs to grade → judging_pools.json, pool_sources.json
  ▼
  │  BT4103-Scrape-and-Tag/labeller/judge_pools.py          grade   → judgments.jsonl
  ▼
  │  judge_pools.py --export                                      → llm_judgments_merged.json, ground_truth_llm.json
```

### 1. Import (`import_scrape_and_tag.py`)

Converts Scrape-and-Tag's CSVs into this repo's JSON format.

**Gigs** are kept only if they pass the same quality bar Scrape-and-Tag's `labeller/sample.py` uses:

- the grounding review kept the extraction on the first pass (no repair);
- the description is in the current prompt format (has a `Deliverable:` line);
- it has a real industry tag (not `OTHER`);
- it has no additional notes;
- its title is unique;
- the description is 300–1,200 characters.

The `Engagement duration: …` line is stripped from every description. The extraction prompt
appends it to every gig, so it says nothing about fit.

**Providers** are kept if classified `PROVIDER` (not `UNCERTAIN`) and they have a headline and an
About section. Profiles with other fields empty are kept on purpose: the provider side is the
search corpus, so it should be as large as the data allows. LLM cost is set by the pool size,
not the corpus size.

IDs follow source-file order, so re-importing the same CSVs gives the same IDs. `id_map.json`
maps each ID back to its source file.

### 2. Judging pools (`build_judging_pools_sat.py`)

Grading all 774 × 1,696 = 1.3M pairs is neither affordable nor useful. Almost all random pairs are
obviously irrelevant. As with the synthetic ground truth, each gig gets a TREC-style pool:
the union of

| Source | Picks per gig | Why |
|---|---|---|
| RRF (k=60) top 20 | 20 | the candidates the reranker will actually re-order |
| refined BM25 top 10 | 10 | lexical matches, including keyword-overlap near-misses |
| dense (mxbai, full dim) top 10 | 10 | semantic matches |
| random providers outside the above | 5 | so the labels aren't only retrieval-shaped |

BM25's and dense's top 10 mostly overlap RRF's top 20, so the 45 picks collapse to 25–39
unique providers per gig (median 31). `pool_sources.json` records which source(s) put each pair
in the pool.

### 3. Grading (`labeller/judge_pools.py`, in the Scrape-and-Tag repo)

The prompt (`labeller/prompts/rubric_0_3.md`) uses the same 0–3 rubric as the synthetic
labels in `llm_judgments.py`. The budget and seniority clauses are dropped because the scraped
data has no fields for them.

| Grade | Meaning |
|---|---|
| 3 | excellent: the provider's specific expertise directly addresses this gig's specific need |
| 2 | good: genuinely relevant domain and skill set, but not a perfect fit (e.g. adjacent sub-focus) |
| 1 | weak: surface relevance (adjacent area, transferable skill), not a real fit |
| 0 | not relevant |

The gig is shown as title and scope. The provider is shown as headline, about, service,
service details and experience.

**Scoring.** The model outputs a single token. Instead of parsing the text, the script reads the
log-probability of each grade token at the answer position and normalises them into a
distribution over 0–3. The grade is its argmax. Each record also keeps the full distribution
(`probs`) and its mean (`expected_grade`). All 23,873 pairs were scored from logprobs; none fell
back to text parsing.

**One model for every pair.** Training labels have to mean the same thing across gigs, and models
disagree far more than prompts do. The Scrape-and-Tag 100×100 comparison found `llama3.1:8b` rated
82% of pairs relevant, the Qwen models ≤ 6%.

**Run history.**

- A 20-gig pilot on `qwen3.6:35b` (Sep 26) graded 600 pairs and hit 6 rate-limit (429) errors.
  Those records stay in `judgments.jsonl` but are not used.
- The full run used `qwen3.8:27b` at 1 call/s, the API's sustainable rate. It ran Sep 26
  21:10 – Sep 27 03:12 and stopped after gig 687. It was resumed on Sep 28 21:43 and
  finished the last 2,672 pairs by 22:32 with no errors. Reruns skip pairs the model has
  already graded, so resuming is just re-running the same command.

### 4. Export

`judge_pools.py --export` writes the two files this repo's `features.py` / `evaluate.py` read:

| File | Contents | Used for |
|---|---|---|
| `llm_judgments_merged.json` | `{hire_id: {provider_id: grade}}`, all pairs including zeros | training |
| `ground_truth_llm.json` | `{hire_id: {provider_id: 0/33/67/100}}`, grade > 0 only | evaluation |

Only gigs whose whole pool is graded are exported, so an ungraded candidate is never mistaken
for a confirmed negative. All 774 gigs are exported.

## What the labels look like

**By pool source.** A pair found by several sources counts under each.

| Source | Pairs | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| RRF | 15,480 | 27.6% | 54.2% | 13.3% | 4.9% |
| Dense | 7,740 | 22.6% | 55.6% | 15.2% | 6.6% |
| BM25 | 7,740 | 35.6% | 45.3% | 12.8% | 6.3% |
| Random | 3,870 | 85.0% | 14.4% | 0.4% | 0.2% |

Random pairs are almost all graded 0, which suggests the grader isn't handing out relevance
freely. Dense retrieval surfaces slightly more good matches than BM25; BM25 surfaces more
clear misses.

**Per gig: providers graded ≥ 2.**

| ≥ 2 per gig | 0 | 1 | 2 | 3 | 4–9 | 10+ |
|---|---|---|---|---|---|---|
| Gigs | 262 | 96 | 107 | 57 | 127 | 125 |

Mean 4.1, median 2. 536 gigs (69%) have no grade 3. 7 gigs have nothing above 0.

**Confidence.** The median top-grade probability is 0.71, and 24% of pairs have a top-grade
probability below 0.6. Mean `expected_grade` by assigned grade: 0 → 0.15, 1 → 0.91,
2 → 1.68, 3 → 2.77. Most uncertainty is between neighbouring grades.

## Caveats

- **Grades cluster at 1.** 46% of pairs are "weak" matches, so the training set has a
  large, fuzzy middle class. `probs` / `expected_grade` are available if a soft target
  works better than the hard grade.
- **A third of gigs have no good match.** 262 gigs have no provider graded ≥ 2. They give the
  reranker no strong positive to learn from, and NDCG/MRR computed on ≥ 2 relevance is undefined
  for them. Decide whether to drop them from evaluation or report them separately.
- **The grade scale is model-specific.** On the 600 pilot pairs, `qwen3.6:35b` and `qwen3.8:27b`
  agree exactly only 45% of the time (98% within one grade). The 35b model grades much higher:
  54% of those pairs ≥ 2, against 15% for 27b. Don't mix labels from the two, and re-grade
  everything if the grader changes.
- **Pool bias.** Only pooled pairs are graded. A relevant provider none of the three retrievers
  ranked highly, and the 5 random picks missed, is invisible. It can't hurt the metrics at the
  cutoffs evaluated, but recall against the whole corpus is unknown.
- **LLM labels, not human labels.** No human spot-check has been done on this set yet.

## Reproduce

```bash
# in ranker/
python pipeline/import_scrape_and_tag.py            # --src ../BT4103-Scrape-and-Tag/docs
python pipeline/build_judging_pools_sat.py          # --rrf-top 20 --method-top 10 --random 5

# in BT4103-Scrape-and-Tag/
py -3 labeller/judge_pools.py --model qwen3.8:27b             # ~7 h at 1 call/s; rerun to resume
py -3 labeller/judge_pools.py --model qwen3.8:27b --export
```

Both ranker scripts are deterministic. But if the CSVs or pool settings change, the IDs and pairs
change, and the grades already in `judgments.jsonl` no longer line up with them. Treat that as a
new labelling run.
