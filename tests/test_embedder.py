import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest


class _SimpleArray:
    def __init__(self, rows: list[list[float]]):
        self._rows = [list(row) for row in rows]

    @property
    def shape(self) -> tuple[int, int]:
        if not self._rows:
            return (0, 0)
        return (len(self._rows), len(self._rows[0]))

    def astype(self, _dtype: str) -> "_SimpleArray":
        return self

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self._rows]

    @classmethod
    def vstack(cls, arrays: list["_SimpleArray"]) -> "_SimpleArray":
        combined: list[list[float]] = []
        for array in arrays:
            combined.extend(array.tolist())
        return cls(combined)

from src.embeddings.embedder import LocalEmbedder
from src.models.symbol_card import SymbolCard


@pytest.fixture()
def sample_cards(tmp_path: Path) -> list[SymbolCard]:
    file_path = tmp_path / "example.cpp"
    file_path.write_text("""// Example\nint add(int a, int b) { return a + b; }\n""")
    return [
        SymbolCard(
            usr="usr-add",
            name="add",
            kind="FUNCTION_DECL",
            signature="int add(int a, int b)",
            namespaces=["math"],
            file=str(file_path),
            line_span={"start": 2, "end": 2},
            doc_comment="Adds two numbers",
            includes=[],
            callees=[],
            snippet="0001: // Example\n0002: int add(int a, int b) { return a + b; }",
        )
    ]


@pytest.fixture(autouse=True)
def patch_model_loading(monkeypatch):
    def fake_init(self, model_name="BAAI/bge-small-en-v1.5", index_dir=Path(".cpp_vector"), device=None):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.device = device or "cpu"

    monkeypatch.setattr(LocalEmbedder, "__init__", fake_init)


def test_index_and_search_roundtrip(tmp_path: Path, sample_cards, monkeypatch):
    embedder = LocalEmbedder()
    embedder.index_dir = tmp_path

    vectors = {
        "int add(int a, int b)\n\nAdds two numbers\n\n0001: // Example\n0002: int add(int a, int b) { return a + b; }": _SimpleArray(
            [[1.0, 0.0, 0.0]]
        ),
        "search": _SimpleArray([[1.0, 0.0, 0.0]]),
    }

    def fake_embed_texts(self, texts, batch_size=16):
        arrs = [vectors[text] for text in texts]
        return _SimpleArray.vstack(arrs)

    monkeypatch.setattr(LocalEmbedder, "embed_texts", fake_embed_texts)

    embedder.index_symbol_cards(sample_cards, index_name="sample", metadata_filename="meta.jsonl")
    results = embedder.search("search", index_name="sample", metadata_filename="meta.jsonl")

    assert len(results) == 1
    assert results[0].card.name == "add"
    assert Path(tmp_path / "meta.jsonl").exists()
    data = json.loads(Path(tmp_path / "meta.jsonl").read_text().strip())
    assert data["snippet"].startswith("0001: // Example")


def test_index_symbol_cards_requires_cards(tmp_path: Path):
    embedder = LocalEmbedder()
    embedder.index_dir = tmp_path
    with pytest.raises(ValueError):
        embedder.index_symbol_cards([], index_name="empty")
