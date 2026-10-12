"""Indexer: scanner -> chunker -> vector store zincirini çalıştırır, sadece değişen dosyaları yeniden indeksler."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from coderag.core.chunker import chunk_file
from coderag.core.scanner import scan
from coderag.core.vector_store import VectorStore


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0


def index_directory(root: Path, store: VectorStore, on_file: Callable[[str], None] | None = None) -> IndexStats:
    """root altındaki kodu indeksler. Değişmeyen dosyalar atlanır, silinen dosyalar indeksten kaldırılır."""
    stats = IndexStats()
    known = store.indexed_files()
    seen = set()

    for source in scan(root):
        path = source.path.as_posix()
        seen.add(path)

        if known.get(path) == source.sha256:
            stats.unchanged += 1
            continue

        if on_file:
            on_file(path)
        if path in known:
            store.delete_file(path)
            stats.updated += 1
        else:
            stats.added += 1

        chunks = chunk_file(source)
        store.add(chunks)
        store.mark_indexed(path, source.sha256)
        stats.chunks += len(chunks)

    for path in known.keys() - seen:
        store.delete_file(path)
        stats.removed += 1

    return stats
