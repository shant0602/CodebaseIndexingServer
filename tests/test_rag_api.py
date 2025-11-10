from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedder import SearchResult
from src.models.symbol_card import SymbolCard
from src.server import rag_api
from src.server.rag_api import (
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SymbolCardPayload,
    get_embedder,
    index_symbols,
    search_symbols,
)


class FakeEmbedder:
    def __init__(self):
        self.cards: list[SymbolCard] = []
        self.index_dir = Path("/tmp")

    def index_symbol_cards(self, cards, index_name="default"):
        self.cards = list(cards)
        self.index_name = index_name
        return Path(f"/tmp/{index_name}.faiss")

    def search(self, query, k=5, index_name="default"):
        if not self.cards:
            raise FileNotFoundError("index missing")
        top = self.cards[:k]
        return [SearchResult(card=card, score=1.0) for card in top]


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch):
    fake = FakeEmbedder()
    get_embedder._instance = fake  # type: ignore[attr-defined]
    yield fake
    if hasattr(get_embedder, "_instance"):
        delattr(get_embedder, "_instance")


def _make_payload() -> SymbolCardPayload:
    return SymbolCardPayload(
        usr="usr-add",
        name="add",
        kind="FUNCTION_DECL",
        signature="int add(int a, int b)",
        namespaces=["math"],
        file="example.cpp",
        line_span={"start": 1, "end": 2},
        doc_comment="Adds numbers",
        includes=[],
        callees=[],
        snippet="0001: int add(int a, int b)",
    )


def test_index_endpoint(stub_embedder):
    request = IndexRequest(cards=[_make_payload()], index_name="test")
    response = index_symbols(request, embedder=stub_embedder)
    assert isinstance(response, IndexResponse)
    assert response.count == 1
    assert stub_embedder.index_name == "test"


def test_search_endpoint(stub_embedder):
    stub_embedder.cards = [
        SymbolCard(
            usr="usr-add",
            name="add",
            kind="FUNCTION_DECL",
            signature="int add(int a, int b)",
            namespaces=["math"],
            file="example.cpp",
            line_span={"start": 1, "end": 2},
            doc_comment="Adds numbers",
            includes=[],
            callees=[],
            snippet="0001: int add(int a, int b)",
        )
    ]
    request = SearchRequest(query="add", top_k=1)
    response = search_symbols(request, embedder=stub_embedder)
    assert isinstance(response, SearchResponse)
    assert response.results[0]["card"]["name"] == "add"


def test_search_missing_index_returns_404(stub_embedder):
    stub_embedder.cards = []
    request = SearchRequest(query="missing")
    with pytest.raises(rag_api.HTTPException) as excinfo:
        search_symbols(request, embedder=stub_embedder)
    assert excinfo.value.status_code == 404
