import pytest
from langchain_core.messages import AIMessage

from core.rag_engine import RagEngine, build_llm, format_snippet, message_text, reciprocal_rank_fusion
from core.vector_store import VectorStore
from tests.test_vector_store import CHUNKS, fake_embed, make_chunk


class RecordingLLM:
    def __init__(self, reply="Cevap [users.py:1-2]"):
        self.reply = reply
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(self.reply)


def make_engine(tmp_path, llm):
    store = VectorStore(tmp_path, embed=fake_embed)
    store.add(CHUNKS)
    return RagEngine(store, llm=llm)


def test_rrf_prefers_chunks_ranked_high_in_both_lists():
    a, b, c = make_chunk("a", start=1), make_chunk("b", start=2), make_chunk("c", start=3)
    merged = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert merged[0] == b
    assert len(merged) == 3


def test_ask_sends_relevant_code_to_llm_and_returns_sources(tmp_path):
    llm = RecordingLLM()
    engine = make_engine(tmp_path, llm)
    answer = engine.ask("get_user_by_id nasıl çalışıyor?")

    assert answer.text == "Cevap [users.py:1-2]"
    assert answer.sources[0].symbol == "get_user_by_id"
    system, human = llm.messages
    assert "users.py:1-2" in human.content
    assert "db.users[user_id]" in human.content
    assert "get_user_by_id nasıl çalışıyor?" in human.content


def test_non_english_question_is_rewritten_for_retrieval(tmp_path):
    class TranslatingLLM(RecordingLLM):
        def invoke(self, messages):
            if "search query" in messages[0].content:
                return AIMessage("send_email smtp")
            return super().invoke(messages)

    engine = make_engine(tmp_path, TranslatingLLM())
    answer = engine.ask("e-posta gönderen fonksiyon hangisi?")
    assert "send_email" in [c.symbol for c in answer.sources]


def test_reranker_reorders_results_and_uses_english_query(tmp_path):
    seen = {}

    def fake_rerank(query, texts):
        seen["query"] = query
        return [10.0 if "parse_config" in text else 0.0 for text in texts]

    class TranslatingLLM(RecordingLLM):
        def invoke(self, messages):
            if "search query" in messages[0].content:
                return AIMessage("send_email smtp")
            return super().invoke(messages)

    store = VectorStore(tmp_path, embed=fake_embed)
    store.add(CHUNKS)
    engine = RagEngine(store, llm=TranslatingLLM(), rerank=fake_rerank)

    sources = engine.retrieve("e-posta gönderen fonksiyon hangisi?")
    assert sources[0].symbol == "parse_config"  # reranker'ın en yükseğe koyduğu
    assert seen["query"] == "send_email smtp"


def test_progress_reports_each_step_in_order(tmp_path):
    steps = []
    store = VectorStore(tmp_path, embed=fake_embed)
    store.add(CHUNKS)
    engine = RagEngine(store, llm=RecordingLLM(), rerank=lambda q, texts: [0.0] * len(texts), progress=steps.append)

    engine.ask("get_user_by_id nedir?")

    assert len(steps) == 5
    assert steps[0].startswith("Soru İngilizce")
    assert steps[1].startswith("Kod aranıyor")
    assert steps[2].startswith("Sonuçlar RRF")
    assert steps[3].startswith("Reranker")
    assert steps[4].startswith("LLM cevabı")


def test_reranker_step_is_skipped_when_reranker_is_off(tmp_path):
    steps = []
    store = VectorStore(tmp_path, embed=fake_embed)
    store.add(CHUNKS)
    RagEngine(store, llm=RecordingLLM(), progress=steps.append).ask("get_user_by_id nedir?")
    assert not any(step.startswith("Reranker") for step in steps)


def test_reranker_can_use_original_question(tmp_path):
    seen = {}

    def fake_rerank(query, texts):
        seen["query"] = query
        return [0.0] * len(texts)

    class TranslatingLLM(RecordingLLM):
        def invoke(self, messages):
            if "search query" in messages[0].content:
                return AIMessage("send_email smtp")
            return super().invoke(messages)

    store = VectorStore(tmp_path, embed=fake_embed)
    store.add(CHUNKS)
    engine = RagEngine(store, llm=TranslatingLLM(), rerank=fake_rerank, rerank_with_english=False)
    engine.retrieve("e-posta gönderen fonksiyon hangisi?")
    assert seen["query"] == "e-posta gönderen fonksiyon hangisi?"


def test_without_reranker_order_comes_from_fusion(tmp_path):
    engine = make_engine(tmp_path, RecordingLLM())
    assert engine.rerank is None
    assert engine.retrieve("get_user_by_id")[0].symbol == "get_user_by_id"


def test_rewrite_prompt_lists_real_symbols(tmp_path):
    llm = RecordingLLM("send_email smtp")
    engine = make_engine(tmp_path, llm)
    engine.rewrite_query("e-posta gönderen fonksiyon hangisi?")

    prompt = llm.messages[0].content
    assert "mail.py: send_email" in prompt
    assert "users.py: get_user_by_id" in prompt
    assert "Never invent" in prompt


def test_rewrite_prompt_uses_only_nearby_symbols_when_index_is_large(tmp_path, monkeypatch):
    monkeypatch.setattr("core.rag_engine.MAX_SYMBOLS_IN_PROMPT", 2)  # 3 sembol var: sınırı aşar
    monkeypatch.setattr("core.rag_engine.SYMBOL_HINT_NEIGHBOURS", 1)
    llm = RecordingLLM("x")
    engine = make_engine(tmp_path, llm)
    engine.rewrite_query("send email smtp")

    prompt = llm.messages[0].content
    assert "send_email" in prompt
    assert "parse_config" not in prompt


def test_snippet_lines_carry_their_real_line_numbers():
    chunk = make_chunk("a = 1\nb = 2\nc = 3", "config.py", start=9, symbol=None)
    snippet = format_snippet(chunk)
    assert "### config.py:9-11" in snippet
    assert " 9 | a = 1\n10 | b = 2\n11 | c = 3" in snippet


def test_ask_on_empty_index_does_not_call_llm(tmp_path):
    llm = RecordingLLM()
    engine = RagEngine(VectorStore(tmp_path, embed=fake_embed), llm=llm)
    answer = engine.ask("herhangi bir şey")
    assert answer.sources == []
    assert "coderag index" in answer.text
    assert llm.messages is None


def test_message_text_handles_part_lists():
    assert message_text("düz") == "düz"
    assert message_text([{"type": "text", "text": "a"}, "b"]) == "ab"


def test_build_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_llm("bilinmeyen")
