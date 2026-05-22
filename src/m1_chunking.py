"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast
    embeddings = model.encode(sentences)

    # 3. Compare consecutive sentences
    import numpy as np
    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunk_text = " ".join(current_group)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = []
        current_group.append(sentences[i])
    
    if current_group:
        chunk_text = " ".join(current_group)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))
    
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents = []
    current_parent_text = ""
    p_index = 0
    
    for para in paragraphs:
        if len(current_parent_text) + len(para) + (2 if current_parent_text else 0) > parent_size and current_parent_text:
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current_parent_text,
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            p_index += 1
            current_parent_text = para
        else:
            if current_parent_text:
                current_parent_text += "\n\n" + para
            else:
                current_parent_text = para

    if current_parent_text:
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current_parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    children = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text
        
        if len(parent_text) <= child_size:
            children.append(Chunk(
                text=parent_text,
                metadata={**metadata, "chunk_type": "child"},
                parent_id=pid
            ))
            continue
            
        step = int(child_size * 0.8)
        if step <= 0:
            step = child_size
        
        start = 0
        while start < len(parent_text):
            end = start + child_size
            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid
                ))
            if end >= len(parent_text):
                break
            start += step

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_content = ""
    
    for part in sections:
        if not part:
            continue
        if re.match(r'^#{1,3}\s+', part.strip()):
            if current_content.strip() or current_header.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, "section": current_header.strip() if current_header else "root", "strategy": "structure"}
                    ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    if current_content.strip() or current_header.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "section": current_header.strip() if current_header else "root", "strategy": "structure"}
            ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    basic_chunks = []
    semantic_chunks = []
    parent_chunks = []
    child_chunks = []
    structure_chunks = []

    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        
        basic_chunks.extend(chunk_basic(text, metadata=meta))
        semantic_chunks.extend(chunk_semantic(text, metadata=meta))
        
        p, c = chunk_hierarchical(text, metadata=meta)
        parent_chunks.extend(p)
        child_chunks.extend(c)
        
        structure_chunks.extend(chunk_structure_aware(text, metadata=meta))

    def get_stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }

    results = {
        "basic": get_stats(basic_chunks),
        "semantic": get_stats(semantic_chunks),
        "hierarchical": get_stats(child_chunks),
        "structure": get_stats(structure_chunks),
    }

    print("\n" + "=" * 55)
    print(f"{'Strategy':<15} | {'Chunks':<8} | {'Avg Len':<8} | {'Min':<5} | {'Max':<5}")
    print("-" * 55)
    
    for name, stats in results.items():
        if name == "hierarchical":
            chunks_str = f"{len(parent_chunks)}p/{len(child_chunks)}c"
        else:
            chunks_str = str(stats["num_chunks"])
            
        print(f"{name:<15} | {chunks_str:<8} | {stats['avg_length']:<8.1f} | {stats['min_length']:<5} | {stats['max_length']:<5}")
    print("=" * 55 + "\n")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
