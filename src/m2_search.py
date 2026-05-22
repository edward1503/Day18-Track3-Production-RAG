"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            segmented = segment_vietnamese(chunk["text"])
            tokens = segmented.lower().split()
            self.corpus_tokens.append(tokens)
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((QDRANT_HOST, QDRANT_PORT))
            sock.close()
            if result == 0:
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            else:
                print("⚠️ Local Qdrant not found on port 6333, using in-memory Qdrant Client (fallback)")
                self.client = QdrantClient(":memory:")
        except Exception:
            print("⚠️ Exception checking local Qdrant, using in-memory Qdrant Client (fallback)")
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        if not chunks:
            return
        
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
            points.append(PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=payload
            ))
            
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )
        
        results = []
        for hit in hits:
            metadata = {k: v for k, v in hit.payload.items() if k != "text"}
            results.append(SearchResult(
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                metadata=metadata,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    
    for results in results_list:
        for rank, result in enumerate(results):
            text = result.text
            if text not in rrf_scores:
                rrf_scores[text] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1]["score"], reverse=True)
    
    hybrid_results = []
    for text, data in sorted_items[:top_k]:
        orig_res = data["result"]
        hybrid_results.append(SearchResult(
            text=text,
            score=data["score"],
            metadata=orig_res.metadata,
            method="hybrid"
        ))
        
    return hybrid_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
