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
bundles libclang, FAISS, the embedding stack, and the FastAPI server so you can
launch the API without installing local Python dependencies.

### 1. Build the image

Run the following command from the repository root to build the image. The
resulting image is tagged as `cpp-rag`, but you can pick any name that fits your
workflow.

```bash
docker build -t cpp-rag .
```

### 2. Start the API server

Expose port `8000` from the container to your host and start the default
entrypoint (Uvicorn serving `src.server.rag_api:APP`):

```bash
docker run --rm -p 8000:8000 cpp-rag
```

After the server boots, visit <http://localhost:8000/docs> to explore the OpenAPI
schema, or send requests to `/index` and `/search` as shown below.

### 3. Run CLI utilities or custom commands

To execute one-off commands—such as parsing a repository or kicking off an
embedding job—override the default command. Mount your source repository and
vector output directory so the container can read and write files on the host:

```bash
docker run --rm \
  -v /absolute/path/to/project:/data/project \
  -v /absolute/path/to/vector_store:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
    --project-root /data/project \
    --output-dir /data/vector_store
```

**Note:** The script will automatically generate `compile_commands.json` inside the container if it doesn't exist (using CMake). This ensures all paths are correct for the container environment. To disable this behavior, use `--no-generate-compile-commands`.

The CLI accepts additional flags for advanced scenarios:

- `--generate-compile-commands` (default: enabled) automatically generates `compile_commands.json` using CMake if missing.
- `--no-generate-compile-commands` disables automatic generation and fails if `compile_commands.json` doesn't exist.
- `--compile-commands` for non-standard locations of `compile_commands.json`.
- `--model-name` to swap the embedding backbone used by `LocalEmbedder`.
- `--device` to pin inference to `cpu` or a specific CUDA device.
- `--context-lines` to control how many extra lines are captured in symbol snippets.

Replace the module invocation with whichever script or CLI you need to run. You
can also open an interactive shell for debugging:

```bash
docker run --rm -it cpp-rag /bin/bash
```

### Environment variables

The container entrypoint automatically wires `LIBCLANG_PATH` if it is not set,
so the parser can load libclang without additional configuration. If you need to
override the model that backs `LocalEmbedder`, supply `EMBEDDING_MODEL_NAME` at
runtime:

```bash
docker run --rm -p 8000:8000 \
  -e EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" \
  cpp-rag
```

Mount project sources or indexes as volumes when invoking the parser or
embedder utilities to persist data outside the container.

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
