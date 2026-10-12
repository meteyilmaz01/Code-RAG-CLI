"""RagEngine: hibrit arama (anlam + kelime), bulunan kodu LLM'e bağlam olarak verip referanslı yanıt üretir."""
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from config import CANDIDATES, DEFAULT_LLM_MODELS, LLM_MODEL, LLM_PROVIDER, RERANK_CANDIDATES, TOP_K
from core.chunker import Chunk
from core.reranker import Reranker
from core.vector_store import VectorStore, embedding_text

RRF_K = 60

SYSTEM_PROMPT = """You are a code assistant answering questions about a codebase.
You are given code snippets retrieved from the project, each headed by a location such as `src/app.py:10-25`.

Rules:
- Answer using only the provided snippets. If they are not enough, say so plainly instead of guessing.
- Cite the places your answer relies on by copying the location from the snippet header, e.g. `src/app.py:10-25`.
- Reply in the same language as the question."""

REWRITE_PROMPT = """Rewrite the user's question as a short English search query for a code search engine.
Include likely function, class or variable names (for example: index_directory, VectorStore).
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
    ) -> None:
        self.store = store
        self.llm = llm  # verilmezse ilk soruda ayarlardan oluşturulur
        self.rerank = rerank  # verilmezse birleştirilmiş sıralama olduğu gibi kullanılır
        self.rerank_with_english = rerank_with_english  # False ise reranker orijinal soruyla puanlar

    def retrieve(self, question: str) -> list[Chunk]:
        queries = [question]
        english = self.rewrite_query(question)
        if english and english != question:
            queries.append(english)

        rankings = []
        for query in queries:
            rankings.append(self.store.search_dense(query, CANDIDATES))
            rankings.append(self.store.search_bm25(query, CANDIDATES))
        fused = reciprocal_rank_fusion(rankings)

        if self.rerank is None:
            return fused[:TOP_K]
        rerank_query = queries[-1] if self.rerank_with_english else question
        return rerank_chunks(self.rerank, rerank_query, fused[:RERANK_CANDIDATES])[:TOP_K]

    def rewrite_query(self, question: str) -> str:
        """Soruyu kod aramasına uygun kısa bir İngilizce sorguya çevirir (kod İngilizce, soru Türkçe olabilir)."""
        reply = self.get_llm().invoke([SystemMessage(REWRITE_PROMPT), HumanMessage(question)])
        lines = message_text(reply.content).strip().splitlines()
        return lines[0].strip(" `\"'") if lines else ""

    def get_llm(self):
        if self.llm is None:
            self.llm = build_llm()
        return self.llm

    def ask(self, question: str) -> Answer:
        if self.store.collection.count() == 0:
            return Answer(NOT_FOUND, [])

        sources = self.retrieve(question)
        reply = self.get_llm().invoke(build_messages(question, sources))
        return Answer(message_text(reply.content), sources)


def reciprocal_rank_fusion(rankings: list[list[Chunk]]) -> list[Chunk]:
    """Birkaç sıralamayı tek listede birleştirir: iki listede de üst sıradaki parça en öne çıkar."""
    scores: dict[tuple, float] = {}
    by_key: dict[tuple, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = (chunk.file_path, chunk.start_line, chunk.end_line)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_key[key] = chunk
    return [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)]


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
    symbol = f" ({chunk.symbol})" if chunk.symbol else ""
    header = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}{symbol}"
    return f"### {header}\n```{chunk.language}\n{chunk.text}\n```"


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
