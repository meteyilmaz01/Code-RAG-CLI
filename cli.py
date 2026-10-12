"""Terminal giriş noktası: `coderag index` ve `coderag ask`."""
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from config import INDEX_DIR, RERANK_MODEL, RERANK_WITH_ENGLISH
from core.indexer import index_directory
from core.rag_engine import RagEngine
from core.reranker import fastembed_reranker
from core.vector_store import VectorStore

app = typer.Typer(help="Kod tabanı üzerinde RAG ile soru-cevap.", no_args_is_help=True)
console = Console()


def open_store(root: Path) -> VectorStore:
    return VectorStore(root / INDEX_DIR)


def open_reranker():
    return fastembed_reranker(RERANK_MODEL)


@app.command()
def index(path: Path = typer.Argument(Path("."), help="İndekslenecek dizin")) -> None:
    """Dizini tara, parçala ve vektör veritabanına kaydet. Sadece değişen dosyalar yeniden işlenir."""
    root = path.resolve()
    if not root.is_dir():
        console.print(f"[red]Dizin bulunamadı:[/red] {root}")
        raise typer.Exit(1)

    store = open_store(root)
    with console.status("Taranıyor...") as status:
        stats = index_directory(root, store, on_file=lambda file: status.update(f"İndeksleniyor: {file}"))

    console.print(
        f"[green]Bitti.[/green] {stats.added} yeni, {stats.updated} güncellenen, {stats.removed} silinen, "
        f"{stats.unchanged} değişmeyen dosya. {stats.chunks} parça eklendi."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Kod tabanına sorulacak soru"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="İndekslenmiş dizin"),
    show_code: bool = typer.Option(False, "--show-code", "-c", help="Kaynak kod parçalarını da göster"),
    rerank: bool = typer.Option(
        False, "--rerank", help="Sonuçları reranker ile yeniden sırala (daha iyi sıralama, soru başına ~20 sn ek süre)"
    ),
) -> None:
    """Soruyu yanıtla; dosya ve satır referanslarını göster."""
    root = path.resolve()
    if not (root / INDEX_DIR).exists():
        console.print(f"[red]İndeks yok.[/red] Önce çalıştır: coderag index {path}")
        raise typer.Exit(1)

    engine = RagEngine(
        open_store(root),
        rerank=open_reranker() if rerank else None,
        rerank_with_english=RERANK_WITH_ENGLISH,
    )
    try:
        with console.status("Düşünüyorum..."):
            answer = engine.ask(question)
    except (RuntimeError, ValueError) as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1)

    console.print(Markdown(answer.text))
    if not answer.sources:
        return

    table = Table(title="Kaynaklar")
    table.add_column("Dosya")
    table.add_column("Satırlar")
    table.add_column("Sembol")
    for chunk in answer.sources:
        table.add_row(chunk.file_path, f"{chunk.start_line}-{chunk.end_line}", chunk.symbol or "")
    console.print(table)

    if show_code:
        for chunk in answer.sources:
            console.rule(f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
            console.print(Syntax(chunk.text, chunk.language, line_numbers=True, start_line=chunk.start_line))


if __name__ == "__main__":
    app()
