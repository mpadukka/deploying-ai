from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


PACKAGE_DIR = Path(__file__).resolve().parent
INDEX_DIR = PACKAGE_DIR / "chroma_db"
COLLECTION_NAME = "toronto_activities"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    if not (INDEX_DIR / "chroma.sqlite3").is_file():
        raise RuntimeError(
            "The prepared activity index is missing. Restore assignment_chat/chroma_db "
            "from the submission; rebuilding is optional, not part of normal startup."
        )
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=DefaultEmbeddingFunction()
    )
    if not collection.count():
        raise RuntimeError("The activity index is empty. Restore the prepared collection.")
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError("The activity index uses an incompatible embedding model.")
    return collection


def search_activities(query: str) -> list[dict[str, str]]:
    if not query.strip():
        raise ValueError("Provide an activity question or preference.")
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(3, collection.count()),
        include=["documents", "metadatas"],
    )
    documents = results["documents"]
    metadatas = results["metadatas"]
    if not documents or not metadatas:
        raise RuntimeError("The activity index returned no document data.")
    passages = []
    for text, metadata in zip(documents[0], metadatas[0], strict=True):
        if text is None or metadata is None:
            raise RuntimeError("An activity result is missing its text or source.")
        passages.append(
            {
                "title": str(metadata["title"]),
                "source": str(metadata["source"]),
                "document": str(metadata["document"]),
                "passage": text,
            }
        )
    return passages
