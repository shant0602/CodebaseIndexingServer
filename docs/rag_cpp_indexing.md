# C++ Retrieval Indexing and API Usage

This document describes how to generate the C++ symbol index, persist vector
embeddings, and query them via the RAG API server.

## Prerequisites

1. **Compile commands database**: The parser depends on
   `compile_commands.json`. Generate it using CMake:

   ```bash
   cmake -S <project-root> -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   cmake --build build
   cp build/compile_commands.json <project-root>/
   ```

   If you already maintain a compile database via another build system, ensure
   the resulting JSON file is placed at the project root before running the
   parser.

2. **Python dependencies**: Install the project requirements, which pin
   compatible versions for libclang bindings, FAISS, FastAPI, PyTorch, and
   Transformers:

   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

## Parsing and Building Symbol Cards

Use `CppParser` to translate C++ sources into structured `ParsedSymbol`
instances. The helper `SymbolCard.from_parsed_symbol` enriches these with code
snippets.

```python
from pathlib import Path

from src.parsers.cpp_parser import CppParser
from src.models.symbol_card import SymbolCard, build_snippet

parser = CppParser(Path("path/to/cpp/project"))
symbols = list(parser.parse())

cards = [
    SymbolCard.from_parsed_symbol(
        symbol,
        snippet=build_snippet(symbol.file, symbol.line_span[0], symbol.line_span[1]),
    )
    for symbol in symbols
]
```

## Embedding and Persisting the Index

`LocalEmbedder` wraps a local transformer (default `BAAI/bge-small-en-v1.5`) and
stores vectors plus metadata under `.cpp_vector/`.

```python
from src.embeddings.embedder import LocalEmbedder

embedder = LocalEmbedder()
embedder.index_symbol_cards(cards, index_name="cpp-symbols")
```

This creates:

- `.cpp_vector/cpp-symbols.faiss`: FAISS index of normalized vectors.
- `.cpp_vector/metadata.jsonl`: JSON Lines metadata aligned with the FAISS
  vector order.

## Running the RAG API

Launch the FastAPI server and expose `/index` and `/search` endpoints:

```bash
uvicorn src.server.rag_api:APP --reload
```

## Running with Docker

A reproducible environment is available via the repository `Dockerfile`. It
bundles libclang, FAISS, the embedding stack, and the FastAPI server. Build the
image from the repository root:

```bash
docker build -t cpp-rag .
```

Run the container and map the service to port `8000` on the host:

```bash
docker run --rm -p 8000:8000 cpp-rag
```

The container entrypoint automatically wires `LIBCLANG_PATH` if it is not set,
so the parser can load libclang without additional configuration. Mount project
sources or indexes as volumes when invoking the parser or embedder utilities.

### Indexing via the API

POST symbol cards to `/index`:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
        "index_name": "cpp-symbols",
        "cards": [
          {
            "usr": "usr-add",
            "name": "add",
            "kind": "FUNCTION_DECL",
            "signature": "int add(int a, int b)",
            "namespaces": ["math"],
            "file": "src/math/add.cpp",
            "line_span": {"start": 10, "end": 18},
            "doc_comment": "Adds two integers",
            "includes": [],
            "callees": [],
            "snippet": "..."
          }
        ]
      }'
```

### Querying the Index

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vector addition", "top_k": 3, "index_name": "cpp-symbols"}'
```

The response returns scored symbol cards with metadata and code snippets ready
for downstream RAG workflows.
