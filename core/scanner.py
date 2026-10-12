"""Scanner: bir klasörü gezer ve içindeki kaynak kod dosyalarını okuyup verir."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pathspec import GitIgnoreSpec

from config import ALWAYS_IGNORE_DIRS, LANGUAGE_BY_EXTENSION, MAX_FILE_BYTES


@dataclass
class SourceFile:
    path: Path  
    language: str  
    content: str  
    sha256: str  


def scan(root: Path) -> Iterator[SourceFile]:
    """root altındaki kod dosyalarını tek tek SourceFile olarak verir."""
    root = root.resolve()
    gitignores = {}  

    for folder, subfolders, files in os.walk(root):
        folder = Path(folder)

        spec = read_gitignore(folder)
        if spec:
            gitignores[folder] = spec

        
        subfolders[:] = sorted(
            name for name in subfolders
            if name not in ALWAYS_IGNORE_DIRS and not is_ignored(folder / name, gitignores, is_folder=True)
        )

        for name in sorted(files):
            path = folder / name
            language = LANGUAGE_BY_EXTENSION.get(path.suffix.lower())
            if language is None or is_ignored(path, gitignores, is_folder=False):
                continue

            source = read_file(path, root, language)
            if source:
                yield source


def read_gitignore(folder: Path) -> GitIgnoreSpec | None:
    """Klasörde .gitignore varsa kurallarını döndürür, yoksa None."""
    try:
        lines = (folder / ".gitignore").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return GitIgnoreSpec.from_lines(lines)


def is_ignored(path: Path, gitignores: dict, is_folder: bool) -> bool:
    """Üst klasörlerdeki .gitignore kurallarından biri bu yolu yasaklıyor mu?"""
    for base, spec in gitignores.items():
        if not path.is_relative_to(base):
            continue
        relative = path.relative_to(base).as_posix()
        if is_folder:
            relative += "/"  
        if spec.match_file(relative):
            return True
    return False


def read_file(path: Path, root: Path, language: str) -> SourceFile | None:
    """Dosyayı okur. Çok büyük, ikili veya okunamayan dosyalar için None döner."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None

    if b"\0" in raw[:8192]: 
        return None

    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")

    return SourceFile(
        path=path.relative_to(root),
        language=language,
        content=text,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
