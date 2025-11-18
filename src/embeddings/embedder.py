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
    from transformers import AutoModel, AutoTokenizer, AutoConfig
except ImportError as exc:  # pragma: no cover
    AutoModel = AutoTokenizer = AutoConfig = None  # type: ignore
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

# Patch DynamicCache for NV-EmbedCode compatibility
# NV-EmbedCode's custom code calls get_usable_length() which doesn't exist in older transformers
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, kv_seq_len=None, layer_idx=0):
            """Compatibility method for NV-EmbedCode's custom code."""
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = get_usable_length
except ImportError:
    pass  # transformers not available, skip patch

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
        trust_remote_code: bool = False,
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
        
        # Check if this model needs sentence-transformers
        # Check model name or path for models that require sentence-transformers
        model_name_lower = self.model_name.lower()
        model_name_normalized = model_name_lower.replace("-", "_").replace("/", "_")
        
        # Check for NV-EmbedCode in name (handles various formats)
        is_nv_embedcode_name = (
            "nv-embedcode" in model_name_lower or 
            "nv_embedcode" in model_name_lower or
            "nv_embed_code" in model_name_lower or
            "nvembedcode" in model_name_lower
        )
        
        # Check for Jina embeddings (typically use sentence-transformers)
        is_jina_model = (
            "jina" in model_name_lower and "embed" in model_name_lower
        )
        
        # Check if it's a local path that contains nv_embed_code directory
        # Handle both /nv_embed_code and /nv-embedcode patterns
        is_nv_embedcode_path = (
            "/nv_embed_code" in self.model_name or 
            "/nv-embedcode" in self.model_name or
            "nv_embed_code" in model_name_normalized or
            self.model_name.endswith("nv_embed_code") or
            self.model_name.endswith("nv-embedcode") or
            self.model_name.endswith("/nv_embed_code") or
            self.model_name.endswith("/nv-embedcode")
        )
        
        # Also check if the directory contains sentence_bert_config.json (indicator of sentence-transformers model)
        is_sentence_transformer_model = False
        if Path(self.model_name).exists():
            config_path = Path(self.model_name) / "sentence_bert_config.json"
            if config_path.exists():
                is_sentence_transformer_model = True
                LOGGER.info("Found sentence_bert_config.json - treating as sentence-transformers model")
        
        self.is_nv_embedcode = is_nv_embedcode_name or is_nv_embedcode_path or is_sentence_transformer_model or is_jina_model
        
        # Log detection result for debugging
        if self.is_nv_embedcode:
            LOGGER.info("Detected NV-EmbedCode/sentence-transformers model from path/name: %s", self.model_name)
        
        if self.is_nv_embedcode:
            # NV-EmbedCode and Jina models require sentence-transformers library
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is required for NV-EmbedCode and Jina models. "
                    "Install it with: pip install sentence-transformers"
                ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR
            
            # Jina v2 and NV-EmbedCode models require trust_remote_code=True to load custom architecture
            is_jina = "jina" in self.model_name.lower()
            # Both Jina and NV-EmbedCode need trust_remote_code=True for custom code
            use_trust_remote_code = True  # Always True for these models
            
            if is_jina:
                LOGGER.info("Detected Jina model - using SentenceTransformer with trust_remote_code=True")
            else:
                LOGGER.info("Detected NV-EmbedCode model - using SentenceTransformer with trust_remote_code=True")
            
            # Pre-load config with trust_remote_code=True to ensure it's cached before SentenceTransformer loads it
            # This is critical because SentenceTransformer internally calls AutoConfig.from_pretrained
            # and needs trust_remote_code=True for models with custom code
            config = None
            if AutoConfig is not None:
                try:
                    config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=use_trust_remote_code)
                    LOGGER.debug("Pre-loaded config with trust_remote_code=True (config type: %s)", type(config).__name__)
                except Exception as e:
                    LOGGER.warning("Failed to pre-load config, but continuing: %s", e)
            
            # Pass trust_remote_code to SentenceTransformer in multiple ways to ensure it propagates
            # to all internal transformers calls (config loading, model loading, etc.)
            st_kwargs = {
                "device": self.device,
                "trust_remote_code": use_trust_remote_code,
            }
            
            # Also pass via model_kwargs to ensure it reaches AutoModel.from_pretrained
            # This is important for models with custom code
            st_kwargs["model_kwargs"] = {"trust_remote_code": use_trust_remote_code}
            
            self.sentence_model = SentenceTransformer(self.model_name, **st_kwargs)
            self.tokenizer = None
            self.model = None
        else:
            # Use standard transformers for other models
            if AutoTokenizer is None or AutoModel is None:
                raise RuntimeError("transformers is required for LocalEmbedder") from _TRANSFORMERS_IMPORT_ERROR
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=trust_remote_code
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=trust_remote_code
            )
            self.model.to(self.device)
            self.model.eval()
            self.sentence_model = None

    # ------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str], batch_size: int = 8, is_query: bool = False) -> np.ndarray:
        """Embed an iterable of texts and return L2 normalized vectors.
        
        Args:
            texts: Texts to embed
            batch_size: Batch size for processing
            is_query: If True, add "query: " prefix (for NV-EmbedCode). If False, add "passage: " prefix.
        """

        if np is None:  # pragma: no cover - exercised via tests with monkeypatching
            raise RuntimeError("numpy is required for LocalEmbedder")

        # Use sentence-transformers for NV-EmbedCode and Jina models
        if self.is_nv_embedcode:
            # Check if this is NV-EmbedCode (needs instruction prefixes) or Jina (no prefixes needed)
            is_jina = "jina" in self.model_name.lower()
            
            if is_jina:
                # Jina embeddings don't need instruction prefixes
                prefixed_texts = texts
            else:
                # NV-EmbedCode requires instruction prefixes
                prefix = "query: " if is_query else "passage: "
                prefixed_texts = [f"{prefix}{text}" for text in texts]
            
            # Clear GPU cache before encoding to avoid OOM
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            embeddings = self.sentence_model.encode(
                prefixed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,  # Show progress for large batches
                device=self.device,
            )
            return embeddings.astype("float32")

        # Standard transformers path for other models
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
                
                # Check if model has an encode method (some embedding models provide this)
                if hasattr(self.model, "encode") and callable(self.model.encode):
                    try:
                        pooled = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
                        embeddings.append(pooled)
                        continue
                    except Exception:
                        # Fall back to standard forward pass
                        pass
                
                # Remove past_key_values to avoid cache issues with some models
                tokens.pop("past_key_values", None)
                
                # Some models need position_ids explicitly set
                if "input_ids" in tokens and "position_ids" not in tokens:
                    batch_size_val, seq_length = tokens["input_ids"].shape
                    # Create position_ids - handle padding properly
                    if "attention_mask" in tokens:
                        # For padded sequences, create position_ids that respect padding
                        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
                        position_ids = position_ids.unsqueeze(0).expand(batch_size_val, -1)
                        # Mask out positions for padding tokens
                        attention_mask = tokens["attention_mask"]
                        position_ids = position_ids * attention_mask.long()
                    else:
                        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
                        position_ids = position_ids.unsqueeze(0).expand(batch_size_val, -1)
                    tokens["position_ids"] = position_ids
                
                # Try to call model
                try:
                    output = self.model(
                        **tokens, 
                        use_cache=False,
                        output_attentions=False,
                        return_dict=True,
                    )
                except (TypeError, AttributeError, ValueError) as e:
                    # If that fails, try without return_dict
                    LOGGER.debug("First model call failed, trying without return_dict: %s", e)
                    try:
                        output = self.model(
                            **tokens, 
                            use_cache=False,
                            output_attentions=False,
                            return_dict=False,
                        )
                    except Exception as e2:
                        # If still failing, try with minimal arguments
                        LOGGER.debug("Second attempt failed, trying minimal args: %s", e2)
                        minimal_tokens = {"input_ids": tokens["input_ids"]}
                        if "attention_mask" in tokens:
                            minimal_tokens["attention_mask"] = tokens["attention_mask"]
                        output = self.model(
                            **minimal_tokens,
                            use_cache=False,
                            return_dict=True,
                        )
                
                # Handle different output structures
                if hasattr(output, "pooler_output") and output.pooler_output is not None:
                    pooled = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    pooled = self._mean_pool(output.last_hidden_state, tokens["attention_mask"])
                elif hasattr(output, "hidden_states") and output.hidden_states:
                    # Use last hidden state from hidden_states tuple
                    pooled = self._mean_pool(output.hidden_states[-1], tokens["attention_mask"])
                else:
                    # Fallback: try to get embeddings directly
                    pooled = output[0] if isinstance(output, tuple) else output
                embeddings.append(pooled.cpu().numpy())
        vectors = np.vstack(embeddings).astype("float32")
        _normalize_vectors(vectors)
        return vectors

    def index_symbol_cards(
        self,
        cards: Sequence[SymbolCard],
        index_name: str = "default",
        metadata_filename: str = "metadata.jsonl",
        batch_size: int = 8,
        chunk_size: int = 1000,
    ) -> Path:
        if not cards:
            raise ValueError("cards must not be empty")

        texts = [self._card_to_text(card) for card in cards]
        
        # Process embeddings in chunks to avoid OOM
        # This prevents sentence-transformers from trying to allocate memory for all texts at once
        index_path = self.index_dir / f"{index_name}.faiss"
        metadata_path = self.index_dir / metadata_filename
        
        if faiss is not None and np is not None:
            # Initialize FAISS index - we'll get dimension from first batch
            dimension = None
            index = None
            
            # Process in chunks and add incrementally to FAISS
            total_cards = len(cards)
            LOGGER.info("Processing %d cards in chunks of %d (embedding batch size: %d)", 
                       total_cards, chunk_size, batch_size)
            
            with metadata_path.open("w", encoding="utf-8") as handle:
                for chunk_start in range(0, total_cards, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_cards)
                    chunk_cards = cards[chunk_start:chunk_end]
                    chunk_texts = texts[chunk_start:chunk_end]
                    
                    LOGGER.info("Processing chunk %d-%d of %d", chunk_start, chunk_end, total_cards)
                    
                    # Embed this chunk
                    chunk_vectors = self.embed_texts(chunk_texts, batch_size=batch_size)
                    
                    # Initialize index on first chunk
                    if index is None:
                        dimension = chunk_vectors.shape[1]
                        index = faiss.IndexFlatIP(dimension)
                        LOGGER.info("Initialized FAISS index with dimension %d", dimension)
                    
                    # Add to FAISS index
                    index.add(_ensure_numpy_array(chunk_vectors))
                    
                    # Write metadata for this chunk
                    for card, text in zip(chunk_cards, chunk_texts):
                        meta = card.to_metadata()
                        meta["snippet"] = card.snippet
                        meta["embedding_text"] = text
                        json.dump(meta, handle)
                        handle.write("\n")
                    
                    # Clear GPU cache after each chunk
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    LOGGER.info("Completed chunk %d-%d", chunk_start, chunk_end)
            
            # Write FAISS index
            faiss.write_index(index, str(index_path))
            LOGGER.info("Persisted FAISS index to %s", index_path)
        else:
            # Fallback: process all at once (may OOM)
            vectors = self.embed_texts(texts, batch_size=batch_size)
            matrix = _as_matrix(vectors)
            dimension = _matrix_dimension(matrix)
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
            query_vectors = self.embed_texts([query], is_query=True)
            scores, indices = index.search(_ensure_numpy_array(query_vectors), k)
            index_order = zip(indices[0], scores[0])
        else:
            vectors = _read_fallback_index(index_path)
            if not vectors:
                raise FileNotFoundError(f"Index {index_path} does not contain vectors")
            query_vectors = self.embed_texts([query], is_query=True)
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
