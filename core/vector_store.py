"""VectorStore: parçaları embedding'e çevirip ChromaDB'ye yazar; anlam araması ve kelime (BM25) araması sunar."""
import json
import os
import re
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Plus

from config import COLLECTION_NAME, EMBEDDING_MODEL, MODEL_CACHE_DIR
from core.chunker import Chunk

BATCH_SIZE = 64

Embedder = Callable[[list[str]], list[list[float]]]


class VectorStore:
    def __init__(self, index_dir: Path, embed: Embedder | None = None) -> None:
        client = chromadb.PersistentClient(
            path=str(index_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        self.embed = embed or fastembed_embedder(EMBEDDING_MODEL)
        self.bm25_chunks: list[Chunk] = []
        self.bm25: BM25Plus | None = None
        self.bm25_stale = True
        self.hashes_path = index_dir / "file_hashes.json"
        self.hashes: dict[str, str] = (
            json.loads(self.hashes_path.read_text(encoding="utf-8")) if self.hashes_path.exists() else {}
        )

    def indexed_files(self) -> dict[str, str]:
        """İndekste olan dosyalar ve indekslendikleri andaki sha256 değerleri."""
        return dict(self.hashes)

    def mark_indexed(self, file_path: str, sha256: str) -> None:
        self.hashes[file_path] = sha256
        self.save_hashes()

    def save_hashes(self) -> None:
        self.hashes_path.write_text(json.dumps(self.hashes, indent=1), encoding="utf-8")

    def add(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            self.collection.upsert(
                ids=[chunk_id(c) for c in batch],
                embeddings=self.embed([embedding_text(c) for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[chunk_metadata(c) for c in batch],
            )
        self.bm25_stale = True

    def delete_file(self, file_path: str) -> None:
        """Değişen ya da silinen dosyanın eski parçalarını kaldırır."""
        self.collection.delete(where={"file_path": file_path})
        self.bm25_stale = True
        if self.hashes.pop(file_path, None) is not None:
            self.save_hashes()

    def search_dense(self, query: str, k: int) -> list[Chunk]:
        """Anlamca en yakın k parça."""
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(query_embeddings=self.embed([query]), n_results=min(k, count))
        return [to_chunk(doc, meta) for doc, meta in zip(result["documents"][0], result["metadatas"][0])]

    def search_bm25(self, query: str, k: int) -> list[Chunk]:
        """Sorgudaki kelimeleri (fonksiyon adları dahil) içeren en uygun k parça."""
        if self.bm25_stale:
            self.rebuild_bm25()
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.bm25_chunks[i] for i in best if scores[i] > 0]

    def symbols_by_file(self) -> dict[str, list[str]]:
        """İndeksteki fonksiyon/sınıf adları, dosyalara göre gruplanmış."""
        grouped: dict[str, set[str]] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            if meta["symbol"]:
                grouped.setdefault(meta["file_path"], set()).add(meta["symbol"])
        return {path: sorted(names) for path, names in sorted(grouped.items())}

    def rebuild_bm25(self) -> None:
        stored = self.collection.get()
        self.bm25_chunks = [to_chunk(d, m) for d, m in zip(stored["documents"], stored["metadatas"])]
        corpus = [tokenize(embedding_text(c)) for c in self.bm25_chunks]
        # BM25Okapi küçük indekslerde negatif skor verir (tek dosyalı repo gibi), BM25Plus vermez.
        self.bm25 = BM25Plus(corpus) if corpus else None
        self.bm25_stale = False


def fastembed_embedder(model_name: str) -> Embedder:
    """fastembed ile embedding üretir. Model ilk kullanımda indirilir."""
    model = None

    def embed(texts: list[str]) -> list[list[float]]:
        nonlocal model
        if model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding

            model = TextEmbedding(model_name, cache_dir=str(MODEL_CACHE_DIR))
        return [vector.tolist() for vector in model.embed(texts)]

    return embed


def embedding_text(chunk: Chunk) -> str:
    """Embedding ve BM25 için metin: dosya yolu ve sembol adı da aramaya katılır."""
    return f"{chunk.file_path}\n{chunk.symbol or ''}\n{chunk.text}"


def tokenize(text: str) -> list[str]:
    """get_user_by_id ve getUserById gibi adları kelimelerine ayırır."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def chunk_id(chunk: Chunk) -> str:
    return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"


def chunk_metadata(chunk: Chunk) -> dict:
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
        "symbol": chunk.symbol or "",  # Chroma None kabul etmez
    }


def to_chunk(document: str, metadata: dict) -> Chunk:
    return Chunk(
        text=document,
        file_path=metadata["file_path"],
        start_line=metadata["start_line"],
        end_line=metadata["end_line"],
        language=metadata["language"],
        symbol=metadata["symbol"] or None,
    )
