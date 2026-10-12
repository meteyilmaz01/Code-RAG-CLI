from pathlib import Path

from coderag.core.scanner import scan


def _write(path: Path, content: str | bytes = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _paths(root: Path) -> list[str]:
    return sorted(s.path.as_posix() for s in scan(root))


def test_collects_target_extensions_only(tmp_path):
    _write(tmp_path / "a.py")
    _write(tmp_path / "b.cs")
    _write(tmp_path / "notes.md", "# hi")
    _write(tmp_path / "image.png", b"\x89PNG")
    _write(tmp_path / "data.txt")
    assert _paths(tmp_path) == ["a.py", "b.cs", "notes.md"]


def test_skips_always_ignored_dirs(tmp_path):
    _write(tmp_path / "src" / "a.py")
    _write(tmp_path / "node_modules" / "lib.js")
    _write(tmp_path / "bin" / "x.cs")
    _write(tmp_path / ".git" / "hook.py")
    assert _paths(tmp_path) == ["src/a.py"]


def test_respects_root_gitignore(tmp_path):
    _write(tmp_path / ".gitignore", "generated/\n*.tmp.py\n")
    _write(tmp_path / "keep.py")
    _write(tmp_path / "generated" / "g.py")
    _write(tmp_path / "scratch.tmp.py")
    assert _paths(tmp_path) == ["keep.py"]


def test_respects_nested_gitignore(tmp_path):
    _write(tmp_path / "pkg" / ".gitignore", "secret.py\n")
    _write(tmp_path / "pkg" / "secret.py")
    _write(tmp_path / "pkg" / "ok.py")
    _write(tmp_path / "secret.py")  # kök seviyesinde etkilenmez
    assert _paths(tmp_path) == ["pkg/ok.py", "secret.py"]


def test_skips_binary_and_oversized(tmp_path, monkeypatch):
    _write(tmp_path / "bin_like.py", b"abc\x00def")
    _write(tmp_path / "big.py", "a" * 50)
    _write(tmp_path / "small.py")
    monkeypatch.setattr("coderag.core.scanner.MAX_FILE_BYTES", 20)
    assert _paths(tmp_path) == ["small.py"]


def test_source_file_fields(tmp_path):
    _write(tmp_path / "m.py", "print('merhaba')\n")
    (source,) = scan(tmp_path)
    assert source.language == "python"
    assert source.content == "print('merhaba')\n"
    assert len(source.sha256) == 64
