"""Reranker: aramadan gelen adayları soruyla birlikte okuyup en alakalıdan en aza yeniden sıralar."""
import os
from typing import Callable

from coderag.config import MODEL_CACHE_DIR

# (sorgu, metinler) -> her metin için alaka puanı
Reranker = Callable[[str, list[str]], list[float]]


def fastembed_reranker(model_name: str) -> Reranker:
    """fastembed çapraz-kodlayıcı (cross-encoder) ile puanlar. Model ilk kullanımda indirilir."""
    model = None

    def score(query: str, texts: list[str]) -> list[float]:
        nonlocal model
        if model is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = TextCrossEncoder(model_name, cache_dir=str(MODEL_CACHE_DIR))
        return [float(s) for s in model.rerank(query, texts)]

    return score
