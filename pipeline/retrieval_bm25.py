"""
Sparse retrieval: Okapi BM25 over provider showcase text.

Two modes:
  - baseline (`refined=False`): plain lowercase/punctuation-stripped
    tokenization, no stemming, no field weighting, no synonym expansion --
    the Round 1 behavior, kept so the notebook can show a clear before/after.
  - refined (`refined=True`, default): stemmed + title-field-weighted
    document tokens (see corpus.provider_bm25_tokens/hirer_bm25_tokens),
    plus query-side synonym expansion from data/synonyms.json to bridge the
    "M&A Advisory" vs "Mergers & Acquisitions"-style vocabulary gap the
    project docs call out. Synonyms.json is a hand-authored stand-in for
    Team 30's not-yet-available `synonyms` taxonomy column.
"""
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from corpus import (
    provider_text, tokenize, provider_bm25_tokens, hirer_bm25_tokens, simple_stem,
)

SYNONYMS_PATH = Path(__file__).parent / "data" / "synonyms.json"


def _load_stemmed_synonyms() -> dict[str, list[str]]:
    raw = json.loads(SYNONYMS_PATH.read_text())
    stemmed = {}
    for key, values in raw.items():
        stemmed_key = simple_stem(key.lower())
        stemmed_values = [simple_stem(v.lower()) for v in values]
        stemmed.setdefault(stemmed_key, []).extend(stemmed_values)
    return stemmed


class BM25Retriever:
    def __init__(self, providers: list[dict], k1: float = 1.5, b: float = 0.75,
                 refined: bool = True, title_weight: int = 3):
        self.providers = providers
        self.provider_ids = [p["provider_id"] for p in providers]
        self.refined = refined
        self.title_weight = title_weight
        self._synonyms = _load_stemmed_synonyms() if refined else {}

        if refined:
            self.tokenized_docs = [provider_bm25_tokens(p, title_weight) for p in providers]
        else:
            self.tokenized_docs = [tokenize(provider_text(p)) for p in providers]

        self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)

    def _expand_with_synonyms(self, tokens: list[str]) -> list[str]:
        expanded = list(tokens)
        for tok in tokens:
            stem = simple_stem(tok)
            for syn in self._synonyms.get(stem, []):
                expanded.append(syn)
        return expanded

    def _query_tokens(self, query_text: str, query_title: str | None = None) -> list[str]:
        if not self.refined:
            return tokenize(query_text)
        # rebuild with field weighting matching the document side
        h_like = {"hire_title": query_title, "hire_description": query_text, "hire_description_additional_notes": None}
        tokens = hirer_bm25_tokens(h_like, self.title_weight) if query_title is not None else \
            tokenize(query_text, stem=True)
        return self._expand_with_synonyms(tokens)

    def rank(self, query_text: str, query_title: str | None = None) -> list[tuple[int, float]]:
        """Returns [(provider_id, score), ...] sorted descending by score.
        Pass `query_title` separately (e.g. h['hire_title']) to apply the
        same title-weighting the refined document index uses; omit it (or
        set refined=False) to reproduce the Round 1 baseline behavior."""
        tokens = self._query_tokens(query_text, query_title)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.provider_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked
