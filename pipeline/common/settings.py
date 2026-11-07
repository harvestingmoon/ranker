"""Every tunable value the pipeline reads, in one place.

Resolution order, highest priority first:

  1. a real environment variable (so CI or a shell can override a run)
  2. a value in `.env` at the repo root (gitignored; see `.env.example`)
  3. the default written below

A missing `.env` is **not** an error. The defaults are the shipped configuration,
so a fresh clone trains and serves exactly the numbers in the README. `.env`
exists to make the knobs discoverable and overridable, not to gate execution.

Why these live here and the word lists do not
---------------------------------------------
These are *settings*: they differ per machine, per run, or per experiment. The
stopword list and the suffix rules are *data* -- they are part of the algorithm's
definition, they are the same on every machine, and they are long enough that
encoding them in an env var would produce an unreadable single line that a shell
would mangle. Those live in `pipeline/data/` next to `synonyms.json`, which is the
same class of thing. Keeping settings and reference data apart is what stops
`.env` (which is gitignored) from becoming load-bearing for correctness.
"""
import os
from pathlib import Path

from pipeline.common.paths import PIPELINE

REPO_ROOT = PIPELINE.parent
ENV_FILE = REPO_ROOT / ".env"

try:  # python-dotenv is optional: env vars plus the defaults below still work
    from dotenv import load_dotenv

    if ENV_FILE.exists():
        # override=False so a real environment variable always beats the file.
        load_dotenv(ENV_FILE, override=False)
except ImportError:  # pragma: no cover - only hit in a minimal install
    pass


def _str(name: str, default: str) -> str:
    """String setting. Blank means 'unset', so an empty .env line is harmless."""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


# --- logging ---------------------------------------------------------------
LOG_LEVEL = _str("PIPELINE_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = _str("PIPELINE_LOG_FORMAT", "%(asctime)s %(levelname)-8s %(name)-28s %(message)s")
LOG_DATE_FORMAT = _str("PIPELINE_LOG_DATE_FORMAT", "%H:%M:%S")
LOG_FILE = _str("PIPELINE_LOG_FILE", "")

# --- Stage 1: recall -------------------------------------------------------
DENSE_MODEL_NAME = _str("DENSE_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
DENSE_TRUNCATE_DIM = _int("DENSE_TRUNCATE_DIM", 1024)
DENSE_QUERY_PREFIX = _str(
    "DENSE_QUERY_PREFIX", "Represent this sentence for searching relevant passages:"
)
BM25_K1 = _float("BM25_K1", 1.5)
BM25_B = _float("BM25_B", 0.75)
BM25_TITLE_WEIGHT = _int("BM25_TITLE_WEIGHT", 3)
RRF_K = _int("RRF_K", 60)
CANDIDATE_POOL = _int("CANDIDATE_POOL", 50)

# --- Stage 2: features and evaluation --------------------------------------
RELEVANCE_THRESHOLD = _int("RELEVANCE_THRESHOLD", 40)
STEM_MIN_LENGTH = _int("STEM_MIN_LENGTH", 4)
SKILL_MODEL_NAME = _str("SKILL_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")

# --- paths -----------------------------------------------------------------
DATA_DIR = _str("PIPELINE_DATA_DIR", "data/data_set_v5_tax")

# --- reproduction ----------------------------------------------------------
CV_FOLDS = _int("CV_FOLDS", 5)
CV_SEED = _int("CV_SEED", 7)
