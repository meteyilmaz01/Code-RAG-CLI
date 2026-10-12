"""RagEngine: hibrit arama (anlam + kelime), bulunan kodu LLM'e bağlam olarak verip referanslı yanıt üretir."""
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    CANDIDATES,
    DEFAULT_LLM_MODELS,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_SYMBOLS_IN_PROMPT,
    RERANK_CANDIDATES,
    TOP_K,
)
from core.chunker import Chunk
from core.reranker import Reranker
from core.vector_store import VectorStore, embedding_text

RRF_K = 60
SYMBOL_HINT_NEIGHBOURS = 60  # çok büyük indekste sembol listesi için bakılan en yakın parça sayısı
NOISE_WEIGHT = 0.6  # test ve doküman parçalarının puan çarpanı
DOC_WORDS = ("readme", "dok", "doc", "kurulum", "install")  # soruda geçerse doküman parçalarına ceza yok

SYSTEM_PROMPT = """You are a code assistant answering questions about a codebase.
You are given code snippets retrieved from the project, each headed by a location such as `src/app.py:10-25`.
Every code line starts with its real line number followed by `|`.

Rules:
- Answer using only the provided snippets. If they are not enough, say so plainly instead of guessing.
- Cite the exact lines your answer relies on, e.g. `src/app.py:17` or `src/app.py:10-25`, using the line numbers
  shown in the snippets. Never guess a line number.
- Reply in the same language as the question."""

REWRITE_PROMPT = """Rewrite the user's question as a short English search query for a code search engine.
Use keywords and, when a list of known symbols is given below, exact names copied from that list.
Never invent function, class or variable names that are not in the list.
Output only the query on a single line, nothing else."""

NOT_FOUND ="İndekste bu soruyla ilgili bir kod bulunamadı. Önce `coderag index` çalıştırdığından emin ol."


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


class RagEngine:
    def __init__(
        self,
        store: VectorStore,
        llm=None,
        rerank: Reranker | None = None,
        rerank_with_english: bool = True,
        progress: Callable[[str], None] | None = None,
        top_k: int = TOP_K,
    ) -> None:
        self.store = store
        self.top_k = top_k  # LLM'e bağlam olarak verilen kod parçası sayısı
        self.llm = llm  # verilmezse ilk soruda ayarlardan oluşturulur
        self.rerank = rerank  # verilmezse birleştirilmiş sıralama olduğu gibi kullanılır
        self.rerank_with_english = rerank_with_english  # False ise reranker orijinal soruyla puanlar
        self.report = progress or (lambda step: None)  # her uzun adımın başında çağrılır

    def retrieve(self, question: str) -> list[Chunk]:
        queries = [question]
        self.report("Soru İngilizce arama sorgusuna çevriliyor")
        english = self.rewrite_query(question)
        if english and english != question:
            queries.append(english)

        self.report(f"Kod aranıyor (anlam + kelime): {queries[-1]}")
        rankings = []
        for query in queries:
            rankings.append(self.store.search_dense(query, CANDIDATES))
            rankings.append(self.store.search_bm25(query, CANDIDATES))

        self.report("Sonuçlar RRF ile birleştiriliyor")
        fused = reciprocal_rank_fusion(rankings, weight=noise_weight(question))

        if self.rerank is None:
            return fused[: self.top_k]
        candidates = fused[:RERANK_CANDIDATES]
        self.report(f"Reranker ile yeniden sıralanıyor ({len(candidates)} aday)")
        rerank_query = queries[-1] if self.rerank_with_english else question
        return rerank_chunks(self.rerank, rerank_query, candidates)[: self.top_k]

    def rewrite_query(self, question: str) -> str:
        """Soruyu kod aramasına uygun kısa bir İngilizce sorguya çevirir (kod İngilizce, soru Türkçe olabilir)."""
        prompt = REWRITE_PROMPT
        symbols = self.symbol_hint(question)
        if symbols:
            prompt += f"\n\nKnown symbols in the project (file: symbols):\n{symbols}"
        reply = self.get_llm().invoke([SystemMessage(prompt), HumanMessage(question)])
        lines = message_text(reply.content).strip().splitlines()
        return lines[0].strip(" `\"'") if lines else ""

    def symbol_hint(self, question: str) -> str:
        """Çeviri adımına verilecek gerçek sembol listesi. Çok büyük indekste soruya yakın parçalarınki."""
        by_file = self.store.symbols_by_file()
        if sum(len(names) for names in by_file.values()) > MAX_SYMBOLS_IN_PROMPT:
            near: dict[str, set[str]] = {}
            for chunk in self.store.search_dense(question, SYMBOL_HINT_NEIGHBOURS):
                if chunk.symbol:
                    near.setdefault(chunk.file_path, set()).add(chunk.symbol)
            by_file = {path: sorted(names) for path, names in sorted(near.items())}
        return "\n".join(f"{path}: {', '.join(names)}" for path, names in by_file.items())

    def get_llm(self):
        if self.llm is None:
            self.llm = build_llm()
        return self.llm

    def ask(self, question: str, on_token: Callable[[str], None] | None = None) -> Answer:
        """Soruyu yanıtlar. on_token verilirse cevap yazıldıkça parça parça ona iletilir."""
        if self.store.collection.count() == 0:
            return Answer(NOT_FOUND, [])

        sources = self.retrieve(question)
        self.report(f"LLM cevabı yazıyor ({len(sources)} kod parçası bağlam olarak veriliyor)")
        return Answer(self.generate(question, sources, on_token), sources)

    def generate(self, question: str, sources: list[Chunk], on_token: Callable[[str], None] | None = None) -> str:
        messages = build_messages(question, sources)
        llm = self.get_llm()
        if on_token is None:
            return message_text(llm.invoke(messages).content)

        if not hasattr(llm, "stream"):  # akış desteklemeyen model: cevap tek parça gelir
            text = message_text(llm.invoke(messages).content)
            on_token(text)
            return text

        parts = []
        for piece in llm.stream(messages):
            delta = message_text(piece.content)
            if delta:
                parts.append(delta)
                on_token(delta)
        return "".join(parts)


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], weight: Callable[[Chunk], float] | None = None
) -> list[Chunk]:
    """Birkaç sıralamayı tek listede birleştirir: iki listede de üst sıradaki parça en öne çıkar.

    weight verilirse her parçanın toplam puanı onunla çarpılır (1.0 = değişmez).
    """
    scores: dict[tuple, float] = {}
    by_key: dict[tuple, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_key[key] = chunk
    if weight:
        scores = {key: score * weight(by_key[key]) for key, score in scores.items()}
    return [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)]


def noise_weight(question: str) -> Callable[[Chunk], float]:
    """Test ve doküman parçalarının puanını düşürür; soru onları istiyorsa dokunmaz."""
    text = question.lower()
    wants_tests = "test" in text
    wants_docs = any(word in text for word in DOC_WORDS)

    def weight(chunk: Chunk) -> float:
        if is_test_file(chunk.file_path) and not wants_tests:
            return NOISE_WEIGHT
        if is_doc_file(chunk.file_path) and not wants_docs:
            return NOISE_WEIGHT
        return 1.0

    return weight


def is_test_file(path: str) -> bool:
    *folders, name = path.lower().split("/")
    return "tests" in folders or "test" in folders or name.startswith("test_") or name.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js"))


def is_doc_file(path: str) -> bool:
    return path.lower().endswith((".md", ".rst"))


def rerank_chunks(rerank: Reranker, query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Adayları reranker puanına göre en alakalıdan en aza sıralar."""
    if not chunks:
        return []
    scores = rerank(query, [embedding_text(c) for c in chunks])
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda pair: pair[0], reverse=True)
    return [chunks[i] for _, i in ranked]


def build_messages(question: str, sources: list[Chunk]) -> list:
    snippets = "\n\n".join(format_snippet(chunk) for chunk in sources)
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"Code snippets:\n\n{snippets}\n\nQuestion: {question}"),
    ]


def format_snippet(chunk: Chunk) -> str:
    """Parçayı başlığı ve her satırın başında gerçek satır numarasıyla biçimlendirir (model numara uydurmasın)."""
    symbol = f" ({chunk.symbol})" if chunk.symbol else ""
    header = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}{symbol}"
    width = len(str(chunk.end_line))
    numbered = "\n".join(
        f"{number:>{width}} | {line}" for number, line in enumerate(chunk.text.split("\n"), start=chunk.start_line)
    )
    return f"### {header}\n```{chunk.language}\n{numbered}\n```"


def message_text(content) -> str:
    """Bazı sağlayıcılar yanıtı parça listesi olarak döndürür; düz metne çevirir."""
    if isinstance(content, str):
        return content
    return "".join(part if isinstance(part, str) else part.get("text", "") for part in content)


def build_llm(provider: str = LLM_PROVIDER, model: str | None = LLM_MODEL):
    """Ayarlardaki sağlayıcı için LangChain sohbet modeli oluşturur."""
    if provider not in DEFAULT_LLM_MODELS:
        raise ValueError(f"Bilinmeyen sağlayıcı '{provider}'. Seçenekler: {', '.join(DEFAULT_LLM_MODELS)}")
    model = model or DEFAULT_LLM_MODELS[provider]

    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            # Ollama'nın varsayılan bağlamı 8 kod parçasını kırpar; çıktı sınırı da döngüyü keser.
            return ChatOllama(model=model, temperature=0, num_ctx=8192, num_predict=1024)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=0)
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0)
    except ImportError as error:
        raise RuntimeError(f"'{provider}' için paket eksik. Kur: pip install -e \".[{provider}]\"") from error
