"""
Turns a provider/hirer JSON record into matchable text.

BM25 (sparse) and the dense embedding model are both evaluated against text
built from the same underlying fields -- what differs between
`provider_text`/`hirer_text` (used by dense) and `provider_bm25_tokens`/
`hirer_bm25_tokens` (used by BM25) is deliberate: BM25 additionally gets
light stemming and title-field weighting, refinements that only make sense
for a token-overlap scorer, not an embedding model.
"""
import json
import re
from pathlib import Path

from pipeline.common import settings
from pipeline.common.paths import PIPELINE

# ---------------------------------------------------------------------------
# Plain text builders (used by the dense embedding model, and as the base
# text BM25's tokenizer also starts from)
# ---------------------------------------------------------------------------

def provider_text(p: dict) -> str:
    """Provider showcase -> one text blob (this is what would get embedded /
    upserted in the real /showcase endpoint per CLAUDE.md).

    `credentials` is deliberately excluded: in the v2 synthetic data it's
    derived from the same background sentence as `about_description`, so
    including both would double-count identical tokens/semantics.
    """
    parts = [
        p.get("about_title"),
        p.get("about_description"),
        p.get("services_offered_title"),
        p.get("services_offered_description"),
        p.get("relevant_experience"),
    ]
    return " ".join(x for x in parts if x)


def hirer_text(h: dict) -> str:
    """Hirer gig posting -> one query text blob (this is what would get
    embedded live at /match time per CLAUDE.md)."""
    parts = [
        h.get("hire_title"),
        h.get("hire_description"),
        h.get("hire_description_additional_notes"),
    ]
    return " ".join(x for x in parts if x)


# ---------------------------------------------------------------------------
# Tokenization + stemming (BM25 only)
# ---------------------------------------------------------------------------

# Stopwords and the stemming rules are REFERENCE DATA, not settings: they define
# what tokenisation means, they are identical on every machine, and .env is
# gitignored -- a fresh clone must still have them. They live in pipeline/data/
# beside synonyms.json. Loaded once at import and cached in module state, so this
# is a single read per process rather than a parse per token.
_STOPWORDS_PATH = PIPELINE / "data" / "stopwords.txt"
_STEM_RULES_PATH = PIPELINE / "data" / "stem_rules.json"


def _load_stopwords(path: Path = _STOPWORDS_PATH) -> set[str]:
    """One word per line; blank lines and # comments ignored."""
    if not path.exists():
        raise FileNotFoundError(f"stopword list missing: {path}")
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if word and not word.startswith("#"):
            words.add(word)
    return words


def _load_stem_rules(path: Path = _STEM_RULES_PATH) -> tuple[list[tuple[str, str]], int]:
    """(rules, min_stem_length). Rules are ORDERED: first match wins, so a
    specific suffix must precede a general one ('ations' before 's')."""
    if not path.exists():
        raise FileNotFoundError(f"stem rule table missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [(str(suffix), str(replacement)) for suffix, replacement in data["rules"]]
    return rules, int(data.get("min_stem_length", settings.STEM_MIN_LENGTH))


STOPWORDS = _load_stopwords()
_SUFFIX_RULES, _MIN_STEM_LEN = _load_stem_rules()


def simple_stem(word: str) -> str:
    """Conservative suffix stripping -- folds inflections (regulate / regulatory /
    regulation, provision / provisioning) onto a shared root.

    Deliberately not Porter/Snowball: the vocabulary here is small and
    domain-specific, and an aggressive stemmer would collide distinct terms.
    A rule only fires if what remains is still >= _MIN_STEM_LEN characters.
    """
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= _MIN_STEM_LEN:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords; optionally stem."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    if stem:
        words = [simple_stem(w) for w in words]
    return words


# ---------------------------------------------------------------------------
# Field-weighted BM25 document/query builders (BM25F-style: title-like
# fields count more than body text, done by repeating their tokens rather
# than needing a different scoring library)
# ---------------------------------------------------------------------------

def provider_bm25_tokens(p: dict, title_weight: int = 3) -> list[str]:
    """Field-weighted document tokens for BM25 (BM25F-style).

    Title-like fields (`about_title`, `services_offered_title`) are repeated
    `title_weight` times so they outrank body text in the score. Repetition is
    used instead of a per-field scorer so a stock BM25 library suffices.
    Tokens are stemmed, unlike `provider_text()` which is fed to the embedder.
    """
    title_text = " ".join(x for x in [p.get("about_title"), p.get("services_offered_title")] if x)
    body_text = " ".join(x for x in [p.get("about_description"), p.get("services_offered_description"), p.get("relevant_experience")] if x)
    title_tokens = tokenize(title_text, stem=True) * title_weight
    body_tokens = tokenize(body_text, stem=True)
    return title_tokens + body_tokens


def hirer_bm25_tokens(h: dict, title_weight: int = 3) -> list[str]:
    """Field-weighted query tokens for BM25 -- the gig-title counterpart of
    `provider_bm25_tokens`, so both sides of the score are built the same way.

    Caveat (known, documented in the README): the BM25 query text also runs
    through `hirer_text()`, which already contains the title, so the gig title
    effectively counts title_weight + 1 times, not title_weight. It affects
    every system equally, so it is flagged rather than changed.
    """
    title_tokens = tokenize(h.get("hire_title") or "", stem=True) * title_weight
    body_text = " ".join(x for x in [h.get("hire_description"), h.get("hire_description_additional_notes")] if x)
    body_tokens = tokenize(body_text, stem=True)
    return title_tokens + body_tokens
