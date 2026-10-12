"""Merkezi ayarlar: uzantılar, model adları, dizinler."""
import os
from pathlib import Path

# İndekslenecek uzantılar ve dil adları (chunker aynı adları kullanır)
LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".cs": "csharp",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".md": "markdown",
}

# .gitignore'a ek olarak her zaman atlanacak klasörler
ALWAYS_IGNORE_DIRS = {".coderag", ".git", "node_modules", "bin", "obj", "__pycache__", ".venv", "venv", "dist", "build"}

# Tek dosya için üst boyut (bayt); bundan büyükler atlanır
MAX_FILE_BYTES = 1_000_000

# Chunking
MAX_CHUNK_CHARS = 1800

# Embedding (fastembed / ONNX)
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"

MODEL_CACHE_DIR = Path.home() / ".coderag" / "models"  # indirilen embedding modeli burada kalır

# Vektör DB
INDEX_DIR = Path(".coderag")
COLLECTION_NAME = "code_chunks"

# Retrieval
TOP_K = 8
CANDIDATES = 20  # anlam ve kelime aramasının her birinden alınan aday sayısı
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"  # çok dilli: Türkçe soruyu doğrudan puanlar
RERANK_WITH_ENGLISH = False  # eval/run_eval.py ölçümü: orijinal soruyla puanlamak daha iyi (MRR 0.54 / 0.42)
RERANK_CANDIDATES = 20  # birleştirilmiş adaylardan reranker'a verilen sayı

# LLM sağlayıcısı: ortam değişkenleriyle değiştirilir
LLM_PROVIDER = os.environ.get("CODERAG_PROVIDER", "ollama")  # ollama | openai | gemini
LLM_MODEL = os.environ.get("CODERAG_MODEL")  # boşsa sağlayıcının varsayılanı
DEFAULT_LLM_MODELS = {
    "ollama": "qwen2.5-coder:7b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}
