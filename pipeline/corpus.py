"""
Turns a provider/hirer JSON record into matchable text.

BM25 (sparse) and the dense embedding model are both evaluated against text
built from the same underlying fields -- what differs between
`provider_text`/`hirer_text` (used by dense) and `provider_bm25_tokens`/
`hirer_bm25_tokens` (used by BM25) is deliberate: BM25 additionally gets
light stemming and title-field weighting, refinements that only make sense
for a token-overlap scorer, not an embedding model.
"""
import re

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

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is",
    "are", "was", "were", "our", "we", "it", "its", "this",
    "that", "with", "at", "as", "be", "by", "has", "have", "had",
    "not", "just", "so", "than", "then", "need", "needs",
}

# A small, deliberately conservative suffix-stripping stemmer -- not a full
# Porter/Snowball implementation, but our vocabulary is bounded and
# domain-specific (consulting/compliance/finance terms), so this is enough
# to fold "regulate/regulatory/regulation" or "provision/provisioning" onto
# a shared root without a new dependency or a corpora download.
_SUFFIX_RULES = [
    ("ations", ""), ("ation", ""), ("atory", "ate"),
    ("edly", ""), ("ing", ""), ("ed", ""),
    ("ies", "y"), ("ives", "ive"), ("ly", ""),
    ("es", ""), ("s", ""),
]
_MIN_STEM_LEN = 4


def simple_stem(word: str) -> str:
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
    title_text = " ".join(x for x in [p.get("about_title"), p.get("services_offered_title")] if x)
    body_text = " ".join(x for x in [p.get("about_description"), p.get("services_offered_description"), p.get("relevant_experience")] if x)
    title_tokens = tokenize(title_text, stem=True) * title_weight
    body_tokens = tokenize(body_text, stem=True)
    return title_tokens + body_tokens


def hirer_bm25_tokens(h: dict, title_weight: int = 3) -> list[str]:
    title_tokens = tokenize(h.get("hire_title") or "", stem=True) * title_weight
    body_text = " ".join(x for x in [h.get("hire_description"), h.get("hire_description_additional_notes")] if x)
    body_tokens = tokenize(body_text, stem=True)
    return title_tokens + body_tokens
