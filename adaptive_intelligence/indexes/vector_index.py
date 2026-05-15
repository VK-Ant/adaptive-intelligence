"""Vector index for semantic retrieval using ChromaDB."""

import logging
from typing import List, Optional
from pathlib import Path

from adaptive_intelligence.indexes.base import BaseIndex, RetrievalResult
from adaptive_intelligence.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorIndex(BaseIndex):
    """Semantic vector index using ChromaDB."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "adaptive_intelligence"):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = None
        self._client = None
        self._chunk_map = {}  # chunk_id -> Chunk
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            if self._persist_dir:
                Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(f"VectorIndex initialized (collection: {self._collection_name})")

        except ImportError:
            raise ImportError(
                "Vector index requires chromadb. Install with: pip install chromadb"
            )

    @property
    def index_type(self) -> str:
        return "vector"

    def add(self, chunks: List[Chunk]) -> int:
        """Add chunks to the vector index."""
        self._ensure_initialized()
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            cid = chunk.chunk_id
            self._chunk_map[cid] = chunk
            ids.append(cid)
            documents.append(chunk.content)
            metadatas.append({
                "doc_id": chunk.doc_id,
                "source": chunk.source_document,
                "is_table": str(chunk.is_table),
                "chunk_index": chunk.chunk_index,
            })

        # ChromaDB handles embedding internally
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to vector index")
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic search using vector similarity."""
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                score = max(0.0, 1.0 - distance / 2.0)

                chunk = self._chunk_map.get(chunk_id)
                if chunk is None:
                    # Reconstruct from stored data
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        doc_id=results["metadatas"][0][i].get("doc_id", ""),
                        content=results["documents"][0][i],
                        source_document=results["metadatas"][0][i].get("source", ""),
                    )
                    self._chunk_map[chunk_id] = chunk

                retrieval_results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    index_type="vector",
                ))

        return retrieval_results

    def clear(self) -> None:
        """Clear the vector index."""
        self._ensure_initialized()
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunk_map.clear()

    def count(self) -> int:
        """Return number of items in the index."""
        self._ensure_initialized()
        return self._collection.count()
