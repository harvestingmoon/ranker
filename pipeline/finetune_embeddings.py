"""
Contrastive fine-tuning of the dense embedding model on our own graded
synthetic pairs, following the ConFit / ESCO-taxonomy job-title-matching
approach: (anchor=hirer query, positive=best-matching provider,
hard_negative=a same-category-but-different-speciality provider) triplets,
trained with MultipleNegativesRankingLoss.

The hard negative is exactly the "partial match" case our own grading
scheme already distinguishes (taxonomy_base=35) -- a provider that's
plausible-but-wrong, which is precisely the boundary Round 1's results were
weakest on (recall past the very top of the list).

Uses the modern sentence-transformers Trainer API directly (rather than the
`model.fit()` convenience wrapper) with `use_cpu=True` explicitly set: an
initial run using `.fit()` silently auto-selected this Mac's MPS (Apple
GPU) backend regardless of the model's declared device, and produced NaN
embeddings after training -- a known class of numerical-stability issue
with some transformer ops on MPS. Forcing the actual Trainer's device
setting to CPU (not just the model's) is what actually avoids it.

Given the small dataset (~234 hirers), this trains in well under an hour on
CPU.

Run: python3 pipeline/finetune_embeddings.py
"""
import json
import random
import time
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from corpus import provider_text, hirer_text
from retrieval_dense import BASE_MODEL_NAME, QUERY_PREFIX_GENERIC

random.seed(7)

DATA_DIR = Path(__file__).parent / "data"
MODEL_OUT_DIR = Path(__file__).parent / "models" / "mxbai-finetuned"
TRAIN_OUTPUT_DIR = Path(__file__).parent / "models" / "_trainer_output"

EPOCHS = 3
BATCH_SIZE = 16
EVAL_HOLDOUT_FRACTION = 0.15


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def build_training_rows(hirers, providers, ground_truth):
    providers_by_id = {p["provider_id"]: p for p in providers}
    by_category = {}
    for p in providers:
        by_category.setdefault(p["_category_id"], []).append(p)

    rows = []
    for h in hirers:
        row = ground_truth.get(str(h["hire_id"]), {})
        if not row:
            continue
        best_pid = max(row, key=lambda pid: row[pid])
        positive = providers_by_id[int(best_pid)]

        candidates = [
            p for p in by_category.get(h["_category_id"], [])
            if p["provider_id"] != positive["provider_id"]
            and row.get(str(p["provider_id"]), 0) < 70
        ]
        if not candidates:
            # fallback: any provider outside this hirer's category, so every
            # training row has a uniform 3-column (anchor, positive, negative)
            # shape -- this case is rare given how the taxonomy is populated.
            candidates = [p for p in providers if p["_category_id"] != h["_category_id"]]

        negative = random.choice(candidates)
        rows.append(dict(
            anchor=QUERY_PREFIX_GENERIC + hirer_text(h),
            positive=provider_text(positive),
            negative=provider_text(negative),
        ))
    return rows


def main():
    providers = load_json("_providers_with_taxonomy.json")
    hirers = load_json("_hirers_with_taxonomy.json")
    ground_truth = load_json("ground_truth.json")

    rows = build_training_rows(hirers, providers, ground_truth)
    random.shuffle(rows)
    n_eval = max(1, int(len(rows) * EVAL_HOLDOUT_FRACTION))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"{len(train_rows)} training rows, {len(eval_rows)} held out for eval")

    train_dataset = Dataset.from_list(train_rows)

    model = SentenceTransformer(BASE_MODEL_NAME, device="cpu")
    train_loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(TRAIN_OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        use_cpu=True,          # the actual fix: forces the Trainer itself off MPS
        fp16=False,
        bf16=False,
        report_to=[],
        logging_steps=5,
        save_strategy="no",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    t0 = time.time()
    trainer.train()
    print(f"training done in {time.time() - t0:.1f}s")

    MODEL_OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_OUT_DIR))
    print(f"saved fine-tuned model to {MODEL_OUT_DIR}")

    # quick before/after sanity check on the held-out eval rows: cosine
    # similarity of (anchor, positive) pairs should go up after fine-tuning,
    # and NaNs would show up here immediately.
    base_model = SentenceTransformer(BASE_MODEL_NAME, device="cpu")
    anchors = [r["anchor"] for r in eval_rows]
    positives = [r["positive"] for r in eval_rows]

    def avg_pos_similarity(m):
        a = m.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
        p = m.encode(positives, convert_to_numpy=True, normalize_embeddings=True)
        return float((a * p).sum(axis=1).mean())

    before = avg_pos_similarity(base_model)
    after = avg_pos_similarity(model)
    print(f"avg anchor-positive cosine similarity on held-out eval pairs: "
          f"before={before:.4f}  after={after:.4f}  delta={after - before:+.4f}")


if __name__ == "__main__":
    main()
