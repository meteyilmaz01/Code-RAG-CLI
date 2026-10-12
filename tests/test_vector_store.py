import zlib

from coderag.core.chunker import Chunk
from coderag.core.vector_store import VectorStore, tokenize


def fake_embed(texts):
    """Gerçek model yerine kelimeleri 64 boyutlu bir torbaya atan sahte embedding."""
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        vector[0] = 0.001  # tamamen sıfır vektör olmasın
        for word in tokenize(text):
            vector[zlib.crc32(word.encode()) % 64] += 1.0
        vectors.append(vector)
    return vectors


def make_chunk(text, file_path="a.py", start=1, symbol=None):
    return Chunk(
        text=text,
        file_path=file_path,
        start_line=start,
        end_line=start + text.count("\n"),
        language="python",
        symbol=symbol,
    )


def make_store(tmp_path):
    return VectorStore(tmp_path, embed=fake_embed)


CHUNKS = [
    make_chunk("def get_user_by_id(user_id):\n    return db.users[user_id]", "users.py", 1, "get_user_by_id"),
    make_chunk("def send_email(to, body):\n    smtp.send(to, body)", "mail.py", 1, "send_email"),
    make_chunk("def parse_config(path):\n    return json.load(open(path))", "config.py", 1, "parse_config"),
]


def test_tokenize_splits_identifiers():
    assert tokenize("getUserById") == ["get", "user", "by", "id"]
    assert tokenize("get_user_by_id") == ["get", "user", "by", "id"]


def test_empty_store_returns_nothing(tmp_path):
    store = make_store(tmp_path)
    assert store.search_dense("anything", 3) == []
    assert store.search_bm25("anything", 3) == []


def test_dense_search_finds_relevant_chunk(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS)
    (best, *_) = store.search_dense("send email to user", 2)
    assert best.symbol == "send_email"
    assert best.file_path == "mail.py"


def test_bm25_search_matches_identifier(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS)
    (best, *_) = store.search_bm25("getUserById", 3)
    assert best.symbol == "get_user_by_id"


def test_chunk_roundtrip_keeps_metadata(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS)
    found = store.search_bm25("parse_config", 1)[0]
    assert found == CHUNKS[2]


def test_symbols_by_file_groups_and_skips_unnamed_chunks(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS + [make_chunk("X = 1", "users.py", 50, None)])
    assert store.symbols_by_file() == {
        "config.py": ["parse_config"],
        "mail.py": ["send_email"],
        "users.py": ["get_user_by_id"],
    }


def test_delete_file_removes_its_chunks(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS)
    store.search_bm25("send_email", 1)  # BM25 indeksini oluştur
    store.delete_file("mail.py")
    assert all(c.file_path != "mail.py" for c in store.search_dense("send email", 5))
    assert all(c.file_path != "mail.py" for c in store.search_bm25("send_email", 5))


def test_data_persists_on_disk(tmp_path):
    make_store(tmp_path).add(CHUNKS)
    reopened = make_store(tmp_path)
    assert reopened.collection.count() == 3


def test_adding_same_chunk_twice_does_not_duplicate(tmp_path):
    store = make_store(tmp_path)
    store.add(CHUNKS)
    store.add(CHUNKS)
    assert store.collection.count() == 3
