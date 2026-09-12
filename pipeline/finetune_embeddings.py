"""
Contrastive fine-tuning of the dense embedding model on our own graded
synthetic pairs, following the ConFit / ESCO-taxonomy job-title-matching
approach: (anchor=hirer query, positive=best-matching provider,
hard_negative=a same-category-but-different-speciality provider) triplets.

The hard negative is exactly the "partial match" case our own grading
scheme already distinguishes (taxonomy_base=35) -- a provider that's
plausible-but-wrong, which is precisely the boundary Round 1's results were
weakest on (recall past the very top of the list).

Matryoshka-aware training
-------------------------
The base encoder (mixedbread-ai/mxbai-embed-large-v1) was pretrained with
Matryoshka Representation Learning, and our dense leg depends on that
property: `retrieval_dense.py` truncates to 256 dims by default and
`run_pipeline.py` sweeps 1024/512/256/128/64. A plain
MultipleNegativesRankingLoss does NOT preserve that nesting -- after such a
fine-tune the first 256 dims are no longer a valid MRL slice, so comparing a
truncated fine-tuned model against a truncated base model would be
comparing two different things. The loss is therefore wrapped in
MatryoshkaLoss over the same dim ladder, so truncation stays meaningful
after fine-tuning. Weights are equal across dims (1.0 each), so the full-dim
objective still dominates the loss.

Device
------
Uses the modern sentence-transformers Trainer API directly (rather than the
`model.fit()` convenience wrapper). `use_cpu=True` is passed *only* when no
CUDA device is present: the original run used `.fit()`, which silently
auto-selected a Mac's MPS backend regardless of the declared device and
produced NaN embeddings after training. That is a real numerical-stability
issue with some transformer ops on MPS, so the CPU-forcing guard is kept for
Macs. On a CUDA box we train on the GPU.

Label source
------------
This script builds triplets from `ground_truth.json` (the taxonomy-formula
ground truth). The report's evaluation uses the LLM-judged ground truth
(`ground_truth_llm.json`), and §8.2 of the write-up describes the recipe as
"grade-3 pairs as positives, grade-0/1 same-category pairs as hard negatives"
-- i.e. the LLM grades. Switching this script to `llm_judgments_merged.json`
is the follow-up that makes the training labels match the evaluation labels;
it is deliberately NOT done here so this run stays a like-for-like
replacement of the previously-verified recipe.

Run: python3 pipeline/finetune_embeddings.py [--epochs 3] [--batch-size 8]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from corpus import provider_text, hirer_text
from retrieval_dense import BASE_MODEL_NAME, QUERY_PREFIX_GENERIC

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# must match retrieval_dense.DEFAULT_TRUNCATE_DIM and run_pipeline.TRUNCATE_DIMS
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64]


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def build_training_rows(hirers, providers, ground_truth, rng):
    providers_by_id = {p["provider_id"]: p for p in providers}
    by_category = {}
    for p in providers:
        by_category.setdefault(p["category_id"], []).append(p)

    rows = []
    for h in hirers:
        row = ground_truth.get(str(h["hire_id"]), {})
        if not row:
            continue
        best_pid = max(row, key=lambda pid: row[pid])
        positive = providers_by_id[int(best_pid)]

        candidates = [
            p for p in by_category.get(h["category_id"], [])
            if p["provider_id"] != positive["provider_id"]
            and row.get(str(p["provider_id"]), 0) < 70
        ]
        if not candidates:
            # fallback: any provider outside this hirer's category, so every
            # training row has a uniform 3-column (anchor, positive, negative)
            # shape -- this case is rare given how the taxonomy is populated.
            candidates = [p for p in providers if p["category_id"] != h["category_id"]]

        negative = rng.choice(candidates)
        rows.append(dict(
            anchor=QUERY_PREFIX_GENERIC + hirer_text(h),
            positive=provider_text(positive),
            negative=provider_text(negative),
        ))
    return rows


def pick_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="per-device batch size (8 x accum 2 keeps the original effective 16)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=5e-6,
                    help="2e-5 collapsed (no warmup); 1e-5 also collapsed while the loss "
                         "summed dims; 5e-6 with n_dims_per_step=1 + warmup + clipping")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="gradient clipping; the first run hit grad_norm=712 without it")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-rows", type=int, default=0,
                    help="subsample training rows (smoke tests); 0 = use all")
    ap.add_argument("--out-name", default="mxbai-finetuned",
                    help="subdirectory of models/ for the saved model")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = pick_device(args.device)
    model_out_dir = MODELS_DIR / args.out_name
    trainer_out_dir = MODELS_DIR / f"_{args.out_name}-trainer"

    providers = load_json("_providers_with_taxonomy.json")
    hirers = load_json("_hirers_with_taxonomy.json")
    ground_truth = load_json("ground_truth.json")

    rows = build_training_rows(hirers, providers, ground_truth, rng)
    rng.shuffle(rows)
    if args.max_rows:
        rows = rows[: args.max_rows]
    n_eval = max(1, int(len(rows) * 0.15))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    print(f"{len(train_rows)} training rows, {len(eval_rows)} held out for eval")
    print(f"device={device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    train_dataset = Dataset.from_list(train_rows)

    model = SentenceTransformer(BASE_MODEL_NAME, device=device)

    # Matryoshka-aware: keeps the truncation ladder valid after fine-tuning.
    # n_dims_per_step=1 trains ONE sampled dim per step instead of summing the
    # loss over all five. Summing multiplies the gradient magnitude by ~5 (a free
    # 5x learning-rate multiplier), which diverged the encoder in testing (loss
    # rose 5.8 -> 15.2, embeddings collapsed to a constant).
    base_loss = losses.MultipleNegativesRankingLoss(model)
    train_loss = losses.MatryoshkaLoss(model, base_loss, matryoshka_dims=MATRYOSHKA_DIMS,
                                       n_dims_per_step=1)

    train_args = dict(
        output_dir=str(trainer_out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        # Transformers v5 ignores `warmup_ratio` (it warns and proceeds with NO
        # warmup), which is how the first run hit grad_norm=712 at step 1 and
        # collapsed. v5 wants the ratio as a float `warmup_steps`.
        warmup_steps=0.1,
        max_grad_norm=args.max_grad_norm,
        fp16=False,
        bf16=False,
        report_to=[],
        logging_steps=5,
        save_strategy="no",
        seed=args.seed,
    )
    if device == "cpu":
        # the original fix: keeps the Trainer itself off MPS on Macs
        train_args["use_cpu"] = True
    st_args = SentenceTransformerTrainingArguments(**train_args)

    trainer = SentenceTransformerTrainer(
        model=model, args=st_args, train_dataset=train_dataset, loss=train_loss,
    )

    t0 = time.time()
    trainer.train()
    print(f"training done in {time.time() - t0:.1f}s"
          + (f"  peak VRAM={torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device == "cuda" else ""))

    model_out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out_dir))
    print(f"saved fine-tuned model to {model_out_dir}")

    # ---------------- sanity checks on the held-out rows ------------------
    # 1. NaNs would show up immediately here (the MPS failure mode).
    # 2. (anchor, positive) similarity should rise.
    # 3. The truncated views must still work -- that is the whole point of the
    #    Matryoshka loss, and it is what run_pipeline.py's dim sweep relies on.
    base_model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    anchors = [r["anchor"] for r in eval_rows]
    positives = [r["positive"] for r in eval_rows]

    def avg_pos_similarity(m, dim):
        a = m.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
        p = m.encode(positives, convert_to_numpy=True, normalize_embeddings=True)
        if dim is not None:
            a, p = a[:, :dim], p[:, :dim]
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
        return float((a * p).sum(axis=1).mean()), bool(np.isnan(a).any() or np.isnan(p).any())

    for dim in (None, 256):
        label = f"{dim or 'full'}d"
        b, b_nan = avg_pos_similarity(base_model, dim)
        a, a_nan = avg_pos_similarity(model, dim)
        print(f"  [{label:>5}] held-out anchor-positive cosine: "
              f"before={b:.4f}  after={a:.4f}  delta={a - b:+.4f}")
        print(f"  [{label:>5}] NaN embeddings: base={b_nan}  fine_tuned={a_nan}")

    # Collapse check. A collapsed encoder maps every input to (nearly) the same
    # point, which shows up as anchor-positive cosine = 1.0000 -- it reads like a
    # spectacular improvement while actually destroying retrieval, because all
    # cosine similarities become ~equal. This failure was observed for real
    # (lr=2e-5, no warmup, no clipping), so it gets a loud, non-zero exit.
    # How to read it: a healthy fine-tune nudges spread slightly (base mxbai sits
    # near 0.5 mean pairwise cosine on this corpus); >0.95 means collapse.
    def mean_pairwise_cos(m, texts):
        v = m.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        c = v @ v.T
        return float(c[~np.eye(len(v), dtype=bool)].mean())

    probe = anchors + positives
    b_spread = mean_pairwise_cos(base_model, probe)
    a_spread = mean_pairwise_cos(model, probe)
    collapsed = a_spread > 0.95
    print(f"  embedding spread (mean pairwise cosine, {len(probe)} held-out texts): "
          f"before={b_spread:.4f}  after={a_spread:.4f}")
    print(f"  COLLAPSE CHECK: "
          f"{'FAIL -- near-constant embeddings, model is unusable' if collapsed else 'ok'}")
    if collapsed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
