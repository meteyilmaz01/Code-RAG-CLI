from core.indexer import index_directory
from core.vector_store import VectorStore
from tests.test_vector_store import fake_embed


def make_store(tmp_path):
    return VectorStore(tmp_path / ".coderag", embed=fake_embed)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_first_run_indexes_everything(tmp_path):
    write(tmp_path, "a.py", "def a():\n    return 1\n")
    write(tmp_path, "b.md", "# Başlık\n\nmetin\n")
    store = make_store(tmp_path)
    stats = index_directory(tmp_path, store)
    assert (stats.added, stats.updated, stats.removed, stats.unchanged) == (2, 0, 0, 0)
    assert stats.chunks >= 2
    assert set(store.indexed_files()) == {"a.py", "b.md"}


def test_second_run_skips_unchanged_files(tmp_path):
    write(tmp_path, "a.py", "def a():\n    return 1\n")
    store = make_store(tmp_path)
    index_directory(tmp_path, store)
    stats = index_directory(tmp_path, store)
    assert (stats.added, stats.updated, stats.unchanged, stats.chunks) == (0, 0, 1, 0)


def test_changed_file_is_reindexed_without_duplicates(tmp_path):
    write(tmp_path, "a.py", "def a():\n    return 1\n")
    store = make_store(tmp_path)
    index_directory(tmp_path, store)
    write(tmp_path, "a.py", "def renamed():\n    return 2\n")
    stats = index_directory(tmp_path, store)
    assert stats.updated == 1
    symbols = [c.symbol for c in store.search_bm25("renamed", 5)]
    assert symbols == ["renamed"]
    assert store.collection.count() == 1


def test_deleted_file_is_removed(tmp_path):
    write(tmp_path, "a.py", "def a():\n    pass\n")
    write(tmp_path, "b.py", "def b():\n    pass\n")
    store = make_store(tmp_path)
    index_directory(tmp_path, store)
    (tmp_path / "b.py").unlink()
    stats = index_directory(tmp_path, store)
    assert stats.removed == 1
    assert set(store.indexed_files()) == {"a.py"}
    assert store.collection.count() == 1


def test_hashes_survive_reopening_store(tmp_path):
    write(tmp_path, "a.py", "def a():\n    pass\n")
    index_directory(tmp_path, make_store(tmp_path))
    stats = index_directory(tmp_path, make_store(tmp_path))
    assert stats.unchanged == 1
