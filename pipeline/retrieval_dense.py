"""
Dense retrieval: Matryoshka embeddings + cosine similarity.

Base model is mixedbread-ai/mxbai-embed-large-v1, trained with Matryoshka
Representation Learning (MRL) -- its 1024-dim output can be truncated to a
smaller size and re-normalized without retraining, trading some quality for
a smaller/faster vector. `truncate_dim=None` uses the full 1024 dims;
passing 512/256/128/64 demonstrates the tradeoff explicitly (see
notebooks/evaluation.ipynb). Round 1's sweep found 256 keeps ~98% of
full-dim NDCG@10 at a quarter of the size, so that's the new production
default (`DEFAULT_TRUNCATE_DIM`).

`model_name` can also point at a local fine-tuned checkpoint (see
finetune_embeddings.py), which is saved in the same sentence-transformers
format and loads identically.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from corpus import provider_text, hirer_text

BASE_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_TRUNCATE_DIM = 256

# mxbai-embed-large-v1 is asymmetric: queries get an instruction prefix,
# passages/documents do not. See the model card. A domain-tuned alternative
# is also provided for the prefix A/B experiment in the notebook.
QUERY_PREFIX_GENERIC = "Represent this sentence for searching relevant passages: "
QUERY_PREFIX_DOMAIN = "Represent this consulting engagement request for matching to a provider showcase: "

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = BASE_MODEL_NAME) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _truncate_and_normalize(vecs: np.ndarray, truncate_dim: int | None) -> np.ndarray:
    if truncate_dim is not None:
        vecs = vecs[:, :truncate_dim]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def encode_docs(texts: list[str], model_name: str = BASE_MODEL_NAME) -> np.ndarray:
    """Batch-encode passage/document texts (no query instruction prefix)."""
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def encode_queries(
    texts: list[str], model_name: str = BASE_MODEL_NAME, query_prefix: str = QUERY_PREFIX_GENERIC,
) -> np.ndarray:
    """Batch-encode query texts (with the asymmetric instruction prefix)."""
    model = _get_model(model_name)
    return model.encode([query_prefix + t for t in texts], convert_to_numpy=True, show_progress_bar=False)


def rank_from_raw(
    doc_raw: np.ndarray,
    query_raw_vec: np.ndarray,
    provider_ids: list[int],
    truncate_dim: int | None = None,
) -> list[tuple[int, float]]:
    """Rank providers for one query given pre-computed raw (untruncated,
    unnormalized) embeddings -- lets a Matryoshka truncation sweep reuse a
    single encode() pass instead of re-embedding per dimension."""
    doc_vecs = _truncate_and_normalize(doc_raw, truncate_dim)
    q_vec = _truncate_and_normalize(query_raw_vec[None, :], truncate_dim)[0]
    scores = doc_vecs @ q_vec
    return sorted(zip(provider_ids, scores.tolist()), key=lambda x: x[1], reverse=True)


class DenseRetriever:
    def __init__(
        self,
        providers: list[dict],
        truncate_dim: int | None = DEFAULT_TRUNCATE_DIM,
        model_name: str = BASE_MODEL_NAME,
        query_prefix: str = QUERY_PREFIX_GENERIC,
    ):
        self.providers = providers
        self.provider_ids = [p["provider_id"] for p in providers]
        self.truncate_dim = truncate_dim
        self.model_name = model_name
        self.query_prefix = query_prefix
        model = _get_model(model_name)
        docs = [provider_text(p) for p in providers]
        raw = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        self.doc_vecs = _truncate_and_normalize(raw, truncate_dim)

    def rank(self, query_text: str) -> list[tuple[int, float]]:
        model = _get_model(self.model_name)
        raw = model.encode([self.query_prefix + query_text], convert_to_numpy=True, show_progress_bar=False)
        q_vec = _truncate_and_normalize(raw, self.truncate_dim)[0]
        scores = self.doc_vecs @ q_vec  # cosine similarity (vectors are unit-normalized)
        ranked = sorted(zip(self.provider_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked
