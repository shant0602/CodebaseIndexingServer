"""Utilities for embedding symbol cards with a local transformer model."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - dependency presence is environment specific
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    faiss = None  # type: ignore
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:  # pragma: no cover - dependency presence is environment specific
    import numpy as np
except ImportError:  # pragma: no cover - handled when features are used
    np = None  # type: ignore

try:  # pragma: no cover
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    AutoModel = AutoTokenizer = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from src.models.symbol_card import SymbolCard

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path(".cpp_vector")


@dataclass
class SearchResult:
    card: SymbolCard
    score: float


class LocalEmbedder:
    """Embeds text using a sentence transformer style encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        index_dir: Path = INDEX_DIR,
        device: str | None = None,
    ) -> None:
        if faiss is None:
            raise RuntimeError("faiss is required for LocalEmbedder") from _FAISS_IMPORT_ERROR
        if torch is None:
            raise RuntimeError("torch is required for LocalEmbedder") from _TORCH_IMPORT_ERROR
        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("transformers is required for LocalEmbedder") from _TRANSFORMERS_IMPORT_ERROR

        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading embedding model %s on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Embed an iterable of texts and return L2 normalized vectors."""

        if np is None:  # pragma: no cover - exercised via tests with monkeypatching
            raise RuntimeError("numpy is required for LocalEmbedder")

        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                output = self.model(**tokens)
                pooled = self._mean_pool(output.last_hidden_state, tokens["attention_mask"])
                embeddings.append(pooled.cpu().numpy())
        vectors = np.vstack(embeddings).astype("float32")
        _normalize_vectors(vectors)
        return vectors

    def index_symbol_cards(
        self,
        cards: Sequence[SymbolCard],
        index_name: str = "default",
        metadata_filename: str = "metadata.jsonl",
    ) -> Path:
        if not cards:
            raise ValueError("cards must not be empty")

        texts = [self._card_to_text(card) for card in cards]
        vectors = self.embed_texts(texts)
        matrix = _as_matrix(vectors)
        dimension = _matrix_dimension(matrix)

        index_path = self.index_dir / f"{index_name}.faiss"
        if faiss is not None and np is not None:
            index = faiss.IndexFlatIP(dimension)
            index.add(_ensure_numpy_array(vectors))
            faiss.write_index(index, str(index_path))
            LOGGER.info("Persisted FAISS index to %s", index_path)
        else:
            _write_fallback_index(index_path, matrix)
            LOGGER.info("Persisted fallback index to %s", index_path)

        metadata_path = self.index_dir / metadata_filename
        with metadata_path.open("w", encoding="utf-8") as handle:
            for card, text in zip(cards, texts):
                meta = card.to_metadata()
                meta["snippet"] = card.snippet
                meta["embedding_text"] = text
                json.dump(meta, handle)
                handle.write("\n")
        LOGGER.info("Wrote metadata for %s symbols", len(cards))

        return index_path

    def search(
        self,
        query: str,
        k: int = 5,
        index_name: str = "default",
        metadata_filename: str = "metadata.jsonl",
    ) -> List[SearchResult]:
        index_path = self.index_dir / f"{index_name}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index {index_path} does not exist")

        metadata = list(self._load_metadata(metadata_filename))

        if faiss is not None and np is not None:
            index = faiss.read_index(str(index_path))
            query_vectors = self.embed_texts([query])
            scores, indices = index.search(_ensure_numpy_array(query_vectors), k)
            index_order = zip(indices[0], scores[0])
        else:
            vectors = _read_fallback_index(index_path)
            if not vectors:
                raise FileNotFoundError(f"Index {index_path} does not contain vectors")
            query_vectors = self.embed_texts([query])
            query_vector = _as_matrix(query_vectors)[0]
            scored = [
                (idx, _dot_product(query_vector, candidate))
                for idx, candidate in enumerate(vectors)
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            index_order = scored[:k]

        results: List[SearchResult] = []
        for idx, score in index_order:
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            card = SymbolCard(
                usr=meta["usr"],
                name=meta["name"],
                kind=meta["kind"],
                signature=meta["signature"],
                namespaces=list(meta.get("namespaces", [])),
                file=meta["file"],
                line_span=meta["line_span"],
                doc_comment=meta.get("doc_comment"),
                includes=list(meta.get("includes", [])),
                callees=list(meta.get("callees", [])),
                snippet=meta.get("snippet", ""),
                score=float(score),
            )
            results.append(SearchResult(card=card, score=float(score)))
        return results

    # ------------------------------------------------------------------
    def _load_metadata(self, metadata_filename: str) -> Iterable[dict]:
        path = self.index_dir / metadata_filename
        if not path.exists():
            raise FileNotFoundError(f"Metadata file {path} missing")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _mean_pool(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _card_to_text(self, card: SymbolCard) -> str:
        doc = card.doc_comment or ""
        parts = [card.signature, doc, card.snippet]
        return "\n\n".join(part for part in parts if part)


def _normalize_vectors(vectors: "np.ndarray") -> None:
    if faiss is not None:
        faiss.normalize_L2(vectors)
        return
    if np is None:  # pragma: no cover - embed_texts guards this path
        raise RuntimeError("numpy is required to normalize vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms


def _as_matrix(vectors: object) -> List[List[float]]:
    if np is not None and isinstance(vectors, np.ndarray):
        return vectors.tolist()
    if hasattr(vectors, "tolist"):
        return list(getattr(vectors, "tolist")())
    return [list(map(float, row)) for row in vectors]  # type: ignore[arg-type]


def _matrix_dimension(matrix: List[List[float]]) -> int:
    if not matrix:
        return 0
    return len(matrix[0])


def _ensure_numpy_array(vectors: object) -> "np.ndarray":
    assert np is not None  # for type checking
    if isinstance(vectors, np.ndarray):
        return vectors.astype("float32")
    return np.array(_as_matrix(vectors), dtype="float32")


def _write_fallback_index(path: Path, matrix: List[List[float]]) -> None:
    payload = {"vectors": matrix}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _read_fallback_index(path: Path) -> List[List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Index {path} does not exist")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [list(map(float, row)) for row in data.get("vectors", [])]


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))
