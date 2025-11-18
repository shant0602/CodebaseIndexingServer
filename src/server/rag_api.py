"""FastAPI server exposing retrieval augmented generation endpoints."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

try:  # pragma: no cover - optional dependency shim for local testing
    from fastapi import Depends, FastAPI, HTTPException
except ImportError:  # pragma: no cover - exercised when fastapi missing
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Depends:  # type: ignore[override]
        def __init__(self, dependency):
            self.dependency = dependency

    class FastAPI:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.routes: list[tuple[str, str, object]] = []

        def post(self, path: str, response_model=None):
            def decorator(func):
                self.routes.append(("POST", path, func))
                return func

            return decorator


try:  # pragma: no cover - optional dependency shim for local testing
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised when pydantic missing
    class _FieldDefault:
        def __init__(self, default=None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default=None, default_factory=None):  # type: ignore[override]
        return _FieldDefault(default=default, default_factory=default_factory)

    class BaseModel:  # type: ignore[override]
        def __init__(self, **data):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in data:
                    value = data[name]
                elif hasattr(self.__class__, name):
                    attr = getattr(self.__class__, name)
                    if isinstance(attr, _FieldDefault):
                        if attr.default_factory is not None:
                            value = attr.default_factory()
                        else:
                            value = attr.default
                    else:
                        value = attr
                else:
                    value = None
                setattr(self, name, value)

        def dict(self):
            annotations = getattr(self.__class__, "__annotations__", {})
            return {name: getattr(self, name) for name in annotations}

        def model_dump(self):  # pragma: no cover - compatibility helper
            return self.dict()

from src.embeddings.embedder import LocalEmbedder
from src.models.symbol_card import SymbolCard

APP = FastAPI(title="C++ RAG Server")


class SymbolCardPayload(BaseModel):
    usr: str
    name: str
    kind: str
    signature: str
    namespaces: List[str]
    file: str
    line_span: dict
    doc_comment: str | None = None
    includes: List[str] = Field(default_factory=list)
    callees: List[str] = Field(default_factory=list)
    snippet: str

    def to_symbol_card(self) -> SymbolCard:
        return SymbolCard(
            usr=self.usr,
            name=self.name,
            kind=self.kind,
            signature=self.signature,
            namespaces=self.namespaces,
            file=self.file,
            line_span=self.line_span,
            doc_comment=self.doc_comment,
            includes=self.includes,
            callees=self.callees,
            snippet=self.snippet,
        )


class IndexRequest(BaseModel):
    cards: List[SymbolCardPayload]
    index_name: str = "default"


class IndexResponse(BaseModel):
    index_path: str
    count: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    index_name: str = "default"


class SearchResponse(BaseModel):
    query: str
    results: List[dict]


def get_embedder() -> LocalEmbedder:
    if not hasattr(get_embedder, "_instance"):
        trust_remote_code = os.getenv("CPP_EMBEDDER_TRUST_REMOTE_CODE", "false").lower() == "true"
        get_embedder._instance = LocalEmbedder(  # type: ignore[attr-defined]
            model_name=os.getenv("CPP_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
            index_dir=Path(os.getenv("CPP_EMBEDDER_INDEX_DIR", ".cpp_vector")),
            trust_remote_code=trust_remote_code,
        )
    return get_embedder._instance  # type: ignore[attr-defined]


@APP.post("/index", response_model=IndexResponse)
def index_symbols(request: IndexRequest, embedder: LocalEmbedder = Depends(get_embedder)) -> IndexResponse:
    if not request.cards:
        raise HTTPException(status_code=400, detail="No symbol cards provided")

    cards = [payload.to_symbol_card() for payload in request.cards]
    index_path = embedder.index_symbol_cards(cards, index_name=request.index_name)
    return IndexResponse(index_path=str(index_path), count=len(cards))


@APP.post("/search", response_model=SearchResponse)
def search_symbols(request: SearchRequest, embedder: LocalEmbedder = Depends(get_embedder)) -> SearchResponse:
    try:
        results = embedder.search(request.query, k=request.top_k, index_name=request.index_name)
    except FileNotFoundError as exc:  # pragma: no cover - thin wrapper
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    serialized = [
        {
            "score": result.score,
            "card": result.card.to_metadata() | {"snippet": result.card.snippet},
        }
        for result in results
    ]
    return SearchResponse(query=request.query, results=serialized)
