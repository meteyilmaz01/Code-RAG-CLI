"""coderag: terminal tabanlı Code-RAG aracı."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("code-rag-cli")
except PackageNotFoundError:  # paket kurulmadan doğrudan kaynaktan çalıştırılıyorsa
    __version__ = "0.0.0"
