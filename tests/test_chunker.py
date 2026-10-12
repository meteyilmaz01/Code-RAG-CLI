from pathlib import Path

from coderag.core.chunker import chunk_file
from coderag.core.scanner import SourceFile


def make_source(code: str, language: str = "python") -> SourceFile:
    return SourceFile(path=Path("src/app.py"), language=language, content=code, sha256="x")


def test_functions_become_separate_chunks():
    code = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    chunks = chunk_file(make_source(code))
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [("a", 1, 2), ("b", 5, 6)]
    assert chunks[0].file_path == "src/app.py"
    assert chunks[0].text == "def a():\n    return 1"


def test_imports_and_constants_are_grouped():
    code = "import os\nimport sys\n\nX = 1\n\n\ndef f():\n    pass\n"
    chunks = chunk_file(make_source(code))
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [(None, 1, 4), ("f", 7, 8)]


def test_decorator_is_part_of_the_chunk():
    code = "@cache\ndef f():\n    pass\n"
    (chunk,) = chunk_file(make_source(code))
    assert chunk.symbol == "f"
    assert chunk.start_line == 1
    assert chunk.text.startswith("@cache")


def test_short_class_stays_in_one_chunk():
    code = "class A:\n    def m(self):\n        pass\n"
    (chunk,) = chunk_file(make_source(code))
    assert chunk.symbol == "A"
    assert (chunk.start_line, chunk.end_line) == (1, 3)


def test_long_class_is_split_by_method(monkeypatch):
    monkeypatch.setattr("coderag.core.chunker.MAX_CHUNK_CHARS", 40)
    code = (
        "class A:\n"
        '    """Belge."""\n'
        "\n"
        "    def one(self):\n"
        "        return 1\n"
        "\n"
        "    def two(self):\n"
        "        return 2\n"
    )
    chunks = chunk_file(make_source(code))
    assert [(c.symbol, c.start_line, c.end_line) for c in chunks] == [
        ("A", 1, 2),
        ("A.one", 4, 5),
        ("A.two", 7, 8),
    ]


def test_other_languages_use_splitter_with_line_numbers(monkeypatch):
    monkeypatch.setattr("coderag.core.chunker.MAX_CHUNK_CHARS", 60)
    code = (
        "public class A {\n"
        "    public int One() { return 1; }\n"
        "}\n"
        "\n"
        "public class B {\n"
        "    public int Two() { return 2; }\n"
        "}\n"
    )
    chunks = chunk_file(make_source(code, language="csharp"))
    assert len(chunks) >= 2
    assert chunks[0].start_line == 1
    lines = code.split("\n")
    for chunk in chunks:
        assert chunk.symbol is None
        assert lines[chunk.start_line - 1:chunk.end_line] == chunk.text.split("\n")


def test_unknown_language_still_gets_chunked():
    (chunk,) = chunk_file(make_source("hello\nworld", language="whatever"))
    assert (chunk.start_line, chunk.end_line) == (1, 2)
