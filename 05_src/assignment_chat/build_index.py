import hashlib
import re

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from assignment_chat.rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PACKAGE_DIR,
)


def build_index() -> int:
    paths = sorted((PACKAGE_DIR / "documents").glob("*.md"))
    if not paths:
        raise ValueError("No Markdown activity guides were found.")
    records: list[tuple[str, str, dict[str, str]]] = []
    for path in paths:
        contents = path.read_text(encoding="utf-8")
        for section in re.split(r"(?m)^## ", contents)[1:]:
            title, _, body = section.partition("\n")
            source = re.search(r"(?m)^Source: (https://\S+)$", body)
            if not title.strip() or not source:
                raise ValueError(f"{path.name}: every activity needs a title and Source URL.")
            passage = f"{title.strip()}\n{body.strip()}"
            if len(passage.split()) > 500:
                raise ValueError(f"{path.name}: split '{title.strip()}' into shorter sections.")
            record_id = hashlib.sha256(
                f"{path.name}:{title.strip()}".encode("utf-8")
            ).hexdigest()
            records.append(
                (
                    record_id,
                    passage,
                    {"title": title.strip(), "source": source.group(1), "document": path.name},
                )
            )
    if not records or len({record[0] for record in records}) != len(records):
        raise ValueError("The guide must contain uniquely titled activity sections.")
    # Embed before replacing the collection so a download failure preserves the old index.
    embedding_function = DefaultEmbeddingFunction()
    documents = [record[1] for record in records]
    embeddings = embedding_function(documents)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"embedding_model": EMBEDDING_MODEL, "hnsw:space": "cosine"},
    )
    collection.add(
        ids=[record[0] for record in records],
        documents=documents,
        metadatas=[record[2] for record in records],
        embeddings=embeddings,
    )
    print(f"Saved {collection.count()} activity chunks to {INDEX_DIR}")
    return collection.count()


if __name__ == "__main__":
    build_index()
