"""Terminal giriş noktası: `coderag index` ve `coderag ask`."""
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
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


class StepPrinter:
    """Çalışan adımı canlı gösterir, biten adımı süresiyle birlikte kalıcı satır olarak yazar."""

    def __init__(self, console: Console, status) -> None:
        self.console = console
        self.status = status
        self.step: str | None = None
        self.started = 0.0

    def __call__(self, step: str) -> None:
        self.finish()
        self.step = step
        self.started = time.perf_counter()
        self.status.update(f"{escape(step)}...")

    def note(self, text: str) -> None:
        """Çalışan adımın canlı satırını günceller; kalıcı satır yazmaz."""
        self.status.update(f"{escape(text)}...")

    def rename(self, text: str) -> None:
        """Adım bitince yazılacak kalıcı satırın metnini sonuçla değiştirir."""
        self.step = text

    def finish(self) -> None:
        if self.step is None:
            return
        elapsed = time.perf_counter() - self.started
        duration = f" [dim]({elapsed:.1f} sn)[/dim]" if elapsed >= 0.1 else ""
        self.console.print(f"[green]✓[/green] {escape(self.step)}{duration}", highlight=False)
        self.step = None


def update_index(root: Path, store: VectorStore, steps: StepPrinter) -> None:
    """Soru öncesi indeksi değişen dosyalara göre günceller. Değişiklik yoksa sadece hash karşılaştırır."""
    steps("İndeks değişiklikler için kontrol ediliyor")
    stats = index_directory(root, store, on_file=lambda file: steps.note(f"İndeksleniyor: {file}"))
    if stats.added + stats.updated + stats.removed:
        steps.rename(
            f"İndeks güncellendi ({stats.added} yeni, {stats.updated} güncellenen, {stats.removed} silinen dosya)"
        )
    else:
        steps.rename("İndeks güncel")


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
    update: bool = typer.Option(
        True, "--update/--no-update", help="Soru öncesi indeksi değişen dosyalara göre güncelle"
    ),
) -> None:
    """Soruyu yanıtla; dosya ve satır referanslarını göster."""
    root = path.resolve()
    if not (root / INDEX_DIR).exists():
        console.print(f"[red]İndeks yok.[/red] Önce çalıştır: coderag index {path}")
        raise typer.Exit(1)

    try:
        with console.status("Başlıyor...") as status:
            steps = StepPrinter(console, status)
            store = open_store(root)
            if update:
                update_index(root, store, steps)
            engine = RagEngine(
                store,
                rerank=open_reranker() if rerank else None,
                rerank_with_english=RERANK_WITH_ENGLISH,
                progress=steps,
            )
            answer = engine.ask(question)
            steps.finish()
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
