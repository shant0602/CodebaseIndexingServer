#!/usr/bin/env python3
"""Query the C++ symbol index."""
from pathlib import Path
from src.embeddings.embedder import LocalEmbedder

# Point to your vector store directory
# Use /data/vector_store when running in Docker, or host path when running locally
import os
INDEX_DIR = Path(os.getenv("INDEX_DIR", "/data/vector_store"))

embedder = LocalEmbedder(
    index_dir=INDEX_DIR,
    device="cuda"  # or "cpu" if no GPU
)

results = embedder.search(
    query="What is the repo doing?",
    k=5,
    index_name="cpp-symbols",  # default index name
    metadata_filename="metadata.jsonl",
)

for result in results:
    card = result.card
    print(f"{card.qualified_name} @ {card.location} (score: {result.score:.4f})")
    if card.doc_comment:
        print(f"  Doc: {card.doc_comment[:100]}...")
    print(f"  {card.snippet[:200]}...")
    print("---")

