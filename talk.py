from src.embeddings.embedder import LocalEmbedder

embedder = LocalEmbedder(index_dir=".cpp_vector")
results = embedder.search(
    query="What is the repo doing?",
    k=5,
    index_name="cpp-symbols",      # whatever prefix your .faiss file uses
    metadata_filename="metadata.jsonl",
)

for result in results:
    card = result.card
    print(f"{card.qualified_name} @ {card.location}")
    print(card.snippet)
    print("---")

