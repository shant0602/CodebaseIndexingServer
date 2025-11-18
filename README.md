# Codebase Indexing Server

A comprehensive system for parsing C++ codebases, extracting symbol information, and generating vector embeddings for semantic code search using FAISS.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Symbol Card Generation](#symbol-card-generation)
- [FAISS Embedding Generation](#faiss-embedding-generation)
- [Docker Usage](#docker-usage)
- [Supported Embedding Models](#supported-embedding-models)
- [Querying the Index](#querying-the-index)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Codebase Indexing Server provides a complete pipeline for:

1. **Parsing C++ Projects**: Uses `libclang` to parse C++ source files and extract structured symbol information
2. **Symbol Card Generation**: Creates rich metadata cards for each symbol (functions, classes, methods, etc.)
3. **Vector Embedding**: Generates embeddings using state-of-the-art language models
4. **FAISS Indexing**: Builds efficient similarity search indices for fast retrieval
5. **Semantic Search**: Enables natural language queries over codebases

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    C++ Project                              │
│              (source files + CMakeLists.txt)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CMake Configuration                                      │
│     - Generate compile_commands.json                        │
│     - Extract compilation flags                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. C++ Parsing (libclang)                                  │
│     - Parse translation units                               │
│     - Extract symbols (functions, classes, etc.)           │
│     - Collect metadata (namespaces, signatures, docs)       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Symbol Card Generation                                  │
│     - Create SymbolCard objects                             │
│     - Extract code snippets with context                   │
│     - Build rich metadata                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Embedding Generation                                    │
│     - Load embedding model                                  │
│     - Generate vectors for each symbol card                 │
│     - Process in batches/chunks                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  5. FAISS Index Creation                                   │
│     - Build FAISS index from vectors                        │
│     - Write metadata.jsonl                                  │
│     - Persist index files                                   │
└─────────────────────────────────────────────────────────────┘
```

## Workflow

### Step-by-Step Process

1. **Project Preparation**
   - Ensure your C++ project has a `CMakeLists.txt`
   - The system will automatically generate `compile_commands.json` if needed

2. **Parsing Phase**
   - `CppParser` reads `compile_commands.json`
   - Uses `libclang` to parse each translation unit
   - Extracts symbols: functions, classes, methods, structs, etc.
   - Collects metadata: namespaces, signatures, doc comments, includes, callees

3. **Symbol Card Creation**
   - Each `ParsedSymbol` is converted to a `SymbolCard`
   - Code snippets are extracted with surrounding context lines
   - Rich metadata is assembled for each symbol

4. **Embedding Phase**
   - Symbol cards are converted to text representations
   - Embedding model processes text in batches
   - Vectors are generated and normalized

5. **Index Building**
   - FAISS index is created incrementally (chunked processing)
   - Metadata is written to `metadata.jsonl`
   - Index files are persisted to disk

## Symbol Card Generation

### What is a Symbol Card?

A `SymbolCard` is a rich data structure containing:

- **Basic Information**:
  - `usr`: Unique Symbol Reference (libclang identifier)
  - `name`: Symbol name
  - `qualified_name`: Fully qualified name (with namespaces)
  - `kind`: Symbol type (FUNCTION_DECL, CLASS_DECL, etc.)
  - `signature`: Function/class signature

- **Location**:
  - `file`: Source file path
  - `line_span`: Start and end line numbers

- **Context**:
  - `namespaces`: List of enclosing namespaces
  - `doc_comment`: Documentation comments (if available)
  - `includes`: Header files included
  - `callees`: Functions called by this symbol
  - `snippet`: Code snippet with surrounding context lines

### Symbol Card Generation Process

```python
# 1. Parse C++ code using libclang
parser = CppParser(project_root, compile_commands)
symbols = parser.parse()  # Returns ParsedSymbol objects

# 2. Build symbol cards with code snippets
for symbol in symbols:
    snippet = build_snippet(
        symbol.file,
        symbol.line_span[0],
        symbol.line_span[1],
        context=2  # Include 2 lines above/below
    )
    card = SymbolCard.from_parsed_symbol(symbol, snippet)
```

### Example Symbol Card

```json
{
  "usr": "c:@F@GetInstance#@S@UnitTest",
  "name": "GetInstance",
  "qualified_name": "testing::UnitTest::GetInstance",
  "kind": "FUNCTION_DECL",
  "signature": "static UnitTest* GetInstance()",
  "namespaces": ["testing"],
  "file": "/path/to/gtest.h",
  "line_span": {"start": 1118, "end": 1118},
  "doc_comment": "Returns singleton instance...",
  "includes": ["gtest/gtest.h"],
  "callees": [],
  "snippet": "1116: // Returns singleton instance\n1117: // Consecutive calls return same object\n1118: static UnitTest* GetInstance();\n1119: \n1120: // Runs all tests..."
}
```

## FAISS Embedding Generation

### Embedding Process

1. **Text Representation**: Each symbol card is converted to a text string containing:
   - Symbol name and qualified name
   - Signature
   - Documentation comment
   - Code snippet
   - Namespace information

2. **Model Loading**: The embedding model is loaded (supports multiple model types):
   - General-purpose models (BAAI/bge-small-en-v1.5)
   - Code-specific models (jinaai/jina-embeddings-v2-base-code, NV-EmbedCode)

3. **Batch Processing**: 
   - Texts are processed in batches (default: 8)
   - Reduces memory usage and improves throughput
   - GPU acceleration when available

4. **Chunked Indexing**:
   - Large codebases are processed in chunks (default: 1000 symbols)
   - Each chunk is embedded and added incrementally to FAISS
   - GPU cache is cleared after each chunk to prevent OOM

5. **FAISS Index Creation**:
   - Uses `IndexFlatIP` (Inner Product) for cosine similarity
   - Vectors are normalized before indexing
   - Index is written to `{index_name}.faiss`

6. **Metadata Storage**:
   - Each symbol's metadata is written to `metadata.jsonl`
   - Includes original symbol card data plus embedding text
   - Enables reconstruction of SymbolCard objects during search

### Output Files

After indexing, you'll have:

```
output_dir/
├── cpp-symbols.faiss          # FAISS vector index
└── metadata.jsonl             # Symbol metadata (one JSON per line)
```

## Docker Usage

### Building the Docker Image

```bash
cd /path/to/CodebaseIndexingServer
docker build -t cpp-rag .
```

### Basic Indexing Command

```bash
docker run --rm --gpus all \
  -v /path/to/your/project:/data/project \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store
```

### Complete Indexing Command with Options

```bash
docker run --rm --gpus all \
  -v /path/to/your/project:/data/project \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store \
  --model-name "jinaai/jina-embeddings-v2-base-code" \
  --device cuda \
  --trust-remote-code \
  --embedding-batch-size 8 \
  --embedding-chunk-size 1000 \
  --index-name "cpp-symbols" \
  --metadata-filename "metadata.jsonl"
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--project-root` | Path to C++ project root | Required |
| `--output-dir` | Directory for FAISS index and metadata | Required |
| `--model-name` | Embedding model name/path | `BAAI/bge-small-en-v1.5` |
| `--device` | Device for embedding (`cuda` or `cpu`) | Auto-detect |
| `--trust-remote-code` | Allow custom model code | `False` |
| `--embedding-batch-size` | Batch size for embeddings | `8` |
| `--embedding-chunk-size` | Cards per chunk | `1000` |
| `--index-name` | FAISS index filename (without .faiss) | `cpp-symbols` |
| `--metadata-filename` | Metadata JSONL filename | `metadata.jsonl` |
| `--context-lines` | Context lines around snippets | `2` |
| `--cards-jsonl` | Optional: Save symbol cards to JSONL | None |
| `--resume-from-cards` | Skip parsing, load from JSONL | None |
| `--skip-embedding` | Generate cards only, skip embedding | `False` |

### Using Local Models

If you have a model downloaded locally:

```bash
docker run --rm --gpus all \
  -v /path/to/project:/data/project \
  -v /path/to/output:/data/vector_store \
  -v /path/to/local/model:/models/my_model \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store \
  --model-name "/models/my_model" \
  --device cuda \
  --trust-remote-code
```

## Supported Embedding Models

### General-Purpose Models

#### BAAI/bge-small-en-v1.5 (Default)
- **Dimensions**: 384
- **Size**: ~130MB
- **Use Case**: General code search, fast inference
- **Command**:
```bash
--model-name "BAAI/bge-small-en-v1.5"
```

### Code-Specific Models

#### jinaai/jina-embeddings-v2-base-code
- **Dimensions**: 768
- **Size**: ~420MB
- **Use Case**: Code-specific semantic search
- **Requires**: `--trust-remote-code`
- **Command**:
```bash
--model-name "jinaai/jina-embeddings-v2-base-code" \
--trust-remote-code
```

#### NV-EmbedCode-7b-v1 (Local)
- **Dimensions**: 4096
- **Size**: ~14GB
- **Use Case**: High-quality code embeddings
- **Requires**: `--trust-remote-code`, local model path
- **Command**:
```bash
--model-name "/models/nv_embed_code" \
--trust-remote-code \
--device cuda \
--embedding-batch-size 4 \
--embedding-chunk-size 500
```

### Model Selection Guide

| Model | Best For | GPU Memory | Speed |
|-------|----------|------------|-------|
| `BAAI/bge-small-en-v1.5` | Quick indexing, general search | Low | Fast |
| `jinaai/jina-embeddings-v2-base-code` | Code-specific queries | Medium | Medium |
| `NV-EmbedCode-7b-v1` | Highest quality embeddings | High (16GB+) | Slower |

### Model-Specific Notes

**Jina Models**:
- Automatically detected and loaded via `sentence-transformers`
- No special query/passage prefixes needed
- Works well for code semantic search

**NV-EmbedCode**:
- Requires `trust_remote_code=True`
- Large model, needs significant GPU memory
- May require lower batch sizes (4-8)
- Best quality for complex code understanding

## Querying the Index

### Basic Query

```bash
docker run --rm --gpus all \
  -v /path/to/output:/data/vector_store \
  -v /path/to/CodebaseIndexingServer:/app \
  cpp-rag \
  python -c "
from pathlib import Path
from src.embeddings.embedder import LocalEmbedder

embedder = LocalEmbedder(
    index_dir=Path('/data/vector_store'),
    model_name='jinaai/jina-embeddings-v2-base-code',
    device='cuda',
    trust_remote_code=True
)

results = embedder.search('singleton pattern', k=5, index_name='cpp-symbols')
for r in results:
    print(f'{r.card.qualified_name} @ {r.card.location} (score: {r.score:.4f})')
    print(r.card.snippet[:200])
    print('---')
"
```

### Using the Query Script

```bash
docker run --rm --gpus all \
  -v /path/to/output:/data/vector_store \
  -v /path/to/CodebaseIndexingServer:/app \
  cpp-rag \
  python query_index.py
```

### API Server

Start the FastAPI server:

```bash
docker run --rm --gpus all \
  -v /path/to/output:/data/vector_store \
  -p 8000:8000 \
  cpp-rag \
  uvicorn src.server.rag_api:APP --host 0.0.0.0 --port 8000
```

Query via HTTP:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "singleton pattern",
    "top_k": 5,
    "index_name": "cpp-symbols"
  }'
```

## Examples

### Example 1: Index GoogleTest

```bash
docker run --rm --gpus all \
  -v /path/to/googletest:/data/project \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store \
  --model-name "jinaai/jina-embeddings-v2-base-code" \
  --trust-remote-code \
  --device cuda
```

### Example 2: Save Symbol Cards for Later

```bash
# Step 1: Generate and save symbol cards
docker run --rm \
  -v /path/to/project:/data/project \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store \
  --cards-jsonl /data/vector_store/cards.jsonl \
  --skip-embedding

# Step 2: Generate embeddings later with different model
docker run --rm --gpus all \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --resume-from-cards /data/vector_store/cards.jsonl \
  --output-dir /data/vector_store \
  --model-name "jinaai/jina-embeddings-v2-base-code" \
  --trust-remote-code \
  --device cuda
```

### Example 3: Large Codebase with Memory Management

```bash
docker run --rm --gpus all \
  -v /path/to/large/project:/data/project \
  -v /path/to/output:/data/vector_store \
  cpp-rag \
  python -m src.scripts.build_cpp_index \
  --project-root /data/project \
  --output-dir /data/vector_store \
  --model-name "jinaai/jina-embeddings-v2-base-code" \
  --trust-remote-code \
  --device cuda \
  --embedding-batch-size 4 \
  --embedding-chunk-size 500
```

### Example 4: Query Examples

```bash
# Find singleton patterns
embedder.search('singleton pattern get instance', k=5)

# Find test registration
embedder.search('test registration macro TEST', k=5)

# Find assertion functions
embedder.search('assertion failure message', k=5)

# Find main entry points
embedder.search('main function entry point', k=5)
```

## Troubleshooting

### Common Issues

#### 1. CMake Errors

**Problem**: `fatal: invalid reference: dawn` or similar CMake errors

**Solution**: The system automatically handles CMake configuration. If issues persist:
- Ensure `CMakeLists.txt` exists in project root
- Check that required dependencies are available
- Review CMake logs in output

#### 2. GPU Out of Memory

**Problem**: `CUDA out of memory` errors

**Solution**: Reduce batch size and chunk size:
```bash
--embedding-batch-size 2 \
--embedding-chunk-size 250
```

#### 3. Model Loading Errors

**Problem**: `trust_remote_code=True` required

**Solution**: Add `--trust-remote-code` flag for custom models:
```bash
--model-name "jinaai/jina-embeddings-v2-base-code" \
--trust-remote-code
```

#### 4. Only Few Translation Units Parsed

**Problem**: Only 4-10 files parsed instead of many

**Solution**: Ensure test files are included in `compile_commands.json`:
- The system automatically enables test building
- Check CMake configuration includes test targets
- Verify `compile_commands.json` contains all `.cc` files

#### 5. Index Not Found During Query

**Problem**: `FileNotFoundError: Index does not exist`

**Solution**: 
- Verify index path is correct
- Ensure `--index-name` matches the actual filename (without `.faiss`)
- Check that both `.faiss` and `metadata.jsonl` exist

### Performance Tips

1. **GPU Memory**: Use smaller batch/chunk sizes for large models
2. **CPU Fallback**: Remove `--gpus all` to use CPU (slower but works)
3. **Incremental Processing**: Use `--resume-from-cards` to avoid re-parsing
4. **Model Selection**: Use smaller models for faster indexing

## Project Structure

```
CodebaseIndexingServer/
├── src/
│   ├── embeddings/
│   │   └── embedder.py          # LocalEmbedder class
│   ├── parsers/
│   │   └── cpp_parser.py         # CppParser using libclang
│   ├── models/
│   │   └── symbol_card.py        # SymbolCard data model
│   ├── scripts/
│   │   └── build_cpp_index.py    # Main indexing script
│   └── server/
│       └── rag_api.py            # FastAPI server
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
└── query_index.py                # Query script example
```

## Requirements

- **Docker**: For containerized execution
- **NVIDIA GPU**: Recommended for fast embedding generation (optional)
- **Python 3.11+**: For local execution
- **CMake**: For generating compile_commands.json
- **libclang**: For C++ parsing (included in Docker image)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

