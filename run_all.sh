#!/usr/bin/env bash
#
# Run the Senseigigs pipeline end to end, from the repo root.
#
#   ./run_all.sh                 shared stages + Lightweight (skips finished ones)
#   ./run_all.sh --force         re-run every stage even if its artifact exists
#   ./run_all.sh --pro           add the Pro stages (cross-encoders need a GPU)
#   ./run_all.sh --eval          add the tags-vs-no-tags significance test
#   ./run_all.sh --serve         serve one gig at the end, to prove the output
#   ./run_all.sh --dry-run       print the commands without running them
#
# PYTHON=/path/to/python ./run_all.sh picks an interpreter (default: python3).
#
# Stage order matters and nothing downstream checks it, so this script is the
# single place that encodes it. Decompress order:
#   Stage 1 recall -> candidate features -> tag features -> reranker -> serving
#
set -euo pipefail

cd "$(dirname "$0")"

DATA=data/data_set_v5_tax
# Artifact paths are filesystem paths from the repo root, so they need the
# `pipeline/` prefix. DATA above does not: --data-dir is resolved against
# pipeline/ by common/paths.py.
FEAT=pipeline/features_data/data_set_v5_tax
RES=pipeline/results_data/data_set_v5_tax
MOD=pipeline/models_data/data_set_v5_tax

PYTHON=${PYTHON:-python3}
FORCE=0 PRO=0 EVAL=0 SERVE=0 DRY=0

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=1 ;;
    --pro)     PRO=1 ;;
    --eval)    EVAL=1 ;;
    --serve)   SERVE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# run_stage <label> <artifact> <command...>
# Skips when <artifact> already exists -- cheap re-runs after a failed stage --
# unless --force. The artifact is the stage's own output, so "done" means done.
run_stage() {
  local label=$1 artifact=$2
  shift 2
  if [ "$FORCE" -eq 0 ] && [ -e "$artifact" ]; then
    echo "  skip  $label   ($artifact exists; --force to rerun)"
    return 0
  fi
  step "$label"
  if [ "$DRY" -eq 1 ]; then
    echo "  would run: $*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------------------
[ -d pipeline ] || { echo "run this from the repo root (no pipeline/ here)" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "python not found: $PYTHON" >&2; exit 1; }

step "checking dependencies"
if [ "$DRY" -eq 1 ]; then
  echo "  would run: import check with $PYTHON"
else
  "$PYTHON" - <<'PY'
import importlib.util
import sys

need = {
    "numpy": "numpy",
    "rank_bm25": "rank_bm25",
    "sentence_transformers": "sentence-transformers",
    "xgboost": "xgboost",
    "sklearn": "scikit-learn",
}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    sys.exit("  missing: " + ", ".join(missing) + "\n  fix: pip install -r pipeline/requirements.txt")
print("  all present")
PY
fi

# ---- shared: Stage 1, features, labels ------------------------------------
run_stage "Stage 1 recall (BM25 + dense -> RRF)" "$RES/rrf_k60.json" \
  "$PYTHON" -m pipeline.recall.run_pipeline --data-dir "$DATA"

run_stage "Stage 2 candidate features" "$FEAT/candidates_top50.csv" \
  "$PYTHON" -m pipeline.features.features --data-dir "$DATA"

# Idempotent by design (a no-op once the committed files already carry the
# top-up), so it runs unconditionally rather than by artifact.
step "merge grading top-up (idempotent)"
if [ "$DRY" -eq 1 ]; then
  echo "  would run: $PYTHON -m pipeline.data_prep.merge_topup_grades --data-dir $DATA"
else
  "$PYTHON" -m pipeline.data_prep.merge_topup_grades --data-dir "$DATA"
fi

# Separate process on purpose: it loads torch, and torch + xgboost in one
# process segfaults on macOS.
run_stage "Stage 2 tag features (own process: torch)" "$FEAT/tag_features.csv" \
  "$PYTHON" -m pipeline.features.tag_features --data-dir "$DATA"

# ---- Lightweight ----------------------------------------------------------
run_stage "Lightweight reranker (LambdaMART)" "$MOD/v5tax_tags_xgb.json" \
  "$PYTHON" -m pipeline.lightweight.rerank_ltr

if [ "$EVAL" -eq 1 ]; then
  step "tags-vs-no-tags significance test"
  if [ "$DRY" -eq 1 ]; then
    echo "  would run: $PYTHON -m pipeline.evaluation.tag_ablation"
  else
    "$PYTHON" -m pipeline.evaluation.tag_ablation
  fi
fi

# ---- Pro (GPU) ------------------------------------------------------------
if [ "$PRO" -eq 1 ]; then
  run_stage "Pro cross-encoder: MiniLM (GPU)" "pipeline/features_data/data_set_v5/ce_scores_v5ce_minilm.csv" \
    "$PYTHON" -m pipeline.pro.rerank_crossencoder --mode finetune --data-dir data/data_set_v5 \
      --model cross-encoder/ms-marco-MiniLM-L6-v2 --tag v5ce_minilm --epochs 3 --cv-folds 5

  run_stage "Pro cross-encoder: bge-reranker-v2-m3 (GPU)" "pipeline/features_data/data_set_v5/ce_scores_v5ce_bge.csv" \
    "$PYTHON" -m pipeline.pro.rerank_crossencoder --mode finetune --data-dir data/data_set_v5 \
      --model BAAI/bge-reranker-v2-m3 --tag v5ce_bge --epochs 2 --cv-folds 3 --max-length 512

  run_stage "Pro cross-encoder: gte-reranker-modernbert (GPU)" "pipeline/features_data/data_set_v5/ce_scores_v5ce_gte.csv" \
    "$PYTHON" -m pipeline.pro.rerank_crossencoder --mode finetune --data-dir data/data_set_v5 \
      --model Alibaba-NLP/gte-reranker-modernbert-base --tag v5ce_gte --epochs 2 --cv-folds 3 --max-length 768

  run_stage "Pro set-wise ranker" "$RES/listwise_ceil_s7.json" \
    "$PYTHON" -m pipeline.pro.pro_ranker
fi

# ---- prove it runs --------------------------------------------------------
if [ "$SERVE" -eq 1 ]; then
  step "serving one gig (tier=lightweight)"
  if [ "$DRY" -eq 1 ]; then
    echo "  would run: $PYTHON -m pipeline.serving.serve --tier lightweight --hire-id 1 --top-k 3"
  else
    "$PYTHON" -m pipeline.serving.serve --tier lightweight --hire-id 1 --top-k 3
  fi
fi

step "done"
echo "  Lightweight model : $MOD/v5tax_tags_xgb.json"
echo "  OOF ranked lists  : $RES/v5tax_tags.json"
if [ "$PRO" -eq 1 ]; then
  echo "  Pro OOF lists     : $RES/listwise_ceil_s7.json"
fi
echo "  Verify the tiers  : $PYTHON -m pipeline.serving.tier --check"
