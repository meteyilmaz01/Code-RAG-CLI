import cli
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from core.vector_store import VectorStore
from tests.test_vector_store import fake_embed

runner = CliRunner()


class FakeLLM:
    def invoke(self, messages):
        return AIMessage("`create_user` kullanıcıyı kaydeder (users.py).")


def setup_project(tmp_path, monkeypatch):
    (tmp_path / "users.py").write_text(
        "def create_user(name):\n    db.insert(name)\n\n\ndef delete_user(name):\n    db.remove(name)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "open_store", lambda root: VectorStore(root / ".coderag", embed=fake_embed))
    monkeypatch.setattr("core.rag_engine.build_llm", lambda: FakeLLM())


def test_index_then_ask(tmp_path, monkeypatch):
    setup_project(tmp_path, monkeypatch)

    result = runner.invoke(cli.app, ["index", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "1 yeni" in result.output

    result = runner.invoke(cli.app, ["ask", "create_user ne yapar?", "--path", str(tmp_path), "--show-code"])
    assert result.exit_code == 0, result.output
    assert "kullanıcıyı kaydeder" in result.output
    assert "users.py" in result.output
    assert "db.insert(name)" in result.output


def test_second_index_run_reports_unchanged(tmp_path, monkeypatch):
    setup_project(tmp_path, monkeypatch)
    runner.invoke(cli.app, ["index", str(tmp_path)])
    result = runner.invoke(cli.app, ["index", str(tmp_path)])
    assert "1 değişmeyen" in result.output
    assert "0 parça" in result.output


def test_ask_without_index_fails_with_hint(tmp_path):
    result = runner.invoke(cli.app, ["ask", "soru", "--path", str(tmp_path)])
    assert result.exit_code == 1
    assert "coderag index" in result.output


def test_index_missing_directory(tmp_path):
    result = runner.invoke(cli.app, ["index", str(tmp_path / "yok")])
    assert result.exit_code == 1
