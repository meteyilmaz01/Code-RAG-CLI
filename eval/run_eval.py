"""Arama kalitesi ölçümü: eval/questions.json'daki her soru için beklenen fonksiyonlar ilk 8 sonuçta çıkıyor mu?

Kullanım:  python eval/run_eval.py [--root DIZIN]
Reranker yapılandırmalarını (model, sorgu dili) yan yana karşılaştırır.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import INDEX_DIR, TOP_K  # noqa: E402
from core.rag_engine import RagEngine  # noqa: E402
from core.reranker import fastembed_reranker  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402

V1_TURBO = "jinaai/jina-reranker-v1-turbo-en"
V2_MULTI = "jinaai/jina-reranker-v2-base-multilingual"

# (ad, reranker modeli, reranker çevrilmiş İngilizce sorguyla mı puanlasın)
CONFIGS = [
    ("RRF (reranker yok)", None, True),
    ("jina-v1-turbo, EN sorgu", V1_TURBO, True),
    ("jina-v2-multilingual, EN sorgu", V2_MULTI, True),
    ("jina-v2-multilingual, orijinal soru", V2_MULTI, False),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    store = VectorStore(args.root / INDEX_DIR)
    validate(store, questions)

    rewrites = RewriteCache(args.root / INDEX_DIR / "eval_rewrites.json", RagEngine(store))
    rerankers = {model: fastembed_reranker(model) for _, model, _ in CONFIGS if model}

    results = []
    for name, model, english in CONFIGS:
        engine = RagEngine(store, rerank=rerankers.get(model), rerank_with_english=english)
        engine.rewrite_query = rewrites.get
        started = time.perf_counter()
        scores = [score(engine.retrieve(q["question"]), q["expected"]) for q in questions]
        results.append((name, scores, time.perf_counter() - started))
        print(f"  {name} tamamlandı ({results[-1][2]:.0f} sn)", flush=True)

    print_summary(results, questions)


def validate(store: VectorStore, questions: list[dict]) -> None:
    """Beklenen her 'dosya::sembol' indekste var mı? Yoksa soru seti kodla uyumsuzdur."""
    present = {f"{m['file_path']}::{m['symbol']}" for m in store.collection.get()["metadatas"]}
    missing = [e for q in questions for e in q["expected"] if e not in present]
    if missing:
        sys.exit("İndekste olmayan beklentiler (önce `coderag index .` çalıştır ya da soru setini güncelle):\n  "
                 + "\n  ".join(missing))


class RewriteCache:
    """Sorgu çevirilerini diske yazar: her yapılandırma aynı çeviriyi kullansın, LLM bir kez çalışsın."""

    def __init__(self, path: Path, engine: RagEngine) -> None:
        self.path = path
        self.engine = engine
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, question: str) -> str:
        if question not in self.cache:
            self.cache[question] = self.engine.rewrite_query(question)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.cache[question]


def score(chunks, expected: list[str]) -> dict:
    """Her beklenen fonksiyonun sonuçlardaki sırasını bulur (yoksa None)."""
    keys = [f"{c.file_path}::{c.symbol}" for c in chunks]
    ranks = {e: (keys.index(e) + 1 if e in keys else None) for e in expected}
    found = [r for r in ranks.values() if r]
    return {
        "recall": len(found) / len(expected),
        "first": min(found) if found else None,
        "ranks": ranks,
    }


def print_summary(results, questions) -> None:
    n = len(questions)
    print(f"\n{'Yapılandırma':<38} {'Recall@' + str(TOP_K):>9} {'İlk isabet ort. sıra':>21} {'MRR':>6} {'Süre':>7}")
    for name, scores, seconds in results:
        recall = sum(s["recall"] for s in scores) / n
        firsts = [s["first"] for s in scores if s["first"]]
        mean_first = sum(firsts) / len(firsts) if firsts else float("nan")
        mrr = sum(1 / s["first"] for s in scores if s["first"]) / n
        print(f"{name:<38} {recall:>9.2f} {mean_first:>21.2f} {mrr:>6.2f} {seconds:>6.0f}s")

    print("\nKaçırılan beklentiler (ilk 8'de çıkmayanlar):")
    for name, scores, _ in results:
        misses = [
            f"    Q{i + 1}: {expected}"
            for i, s in enumerate(scores)
            for expected, rank in s["ranks"].items()
            if rank is None
        ]
        print(f"  {name}: {len(misses)} kaçırılan")
        print("\n".join(misses))


if __name__ == "__main__":
    main()
