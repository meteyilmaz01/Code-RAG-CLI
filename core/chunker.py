"""Chunker: bir kod dosyasını fonksiyon/sınıf sınırlarında parçalara böler (şimdilik sadece Python)."""
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tree_sitter_language_pack import get_parser

from config import MAX_CHUNK_CHARS
from core.scanner import SourceFile

SPLITTER_LANGUAGES = {
    "csharp": Language.CSHARP,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "go": Language.GO,
    "markdown": Language.MARKDOWN,
}


@dataclass
class Chunk:
    text: str
    file_path: str
    start_line: int  # 1 tabanlı
    end_line: int
    language: str
    symbol: str | None = None  


def chunk_file(source: SourceFile) -> list[Chunk]:
    """Dosyayı parçalara böler. Python fonksiyon/sınıf sınırlarında, diğer diller metin ayracıyla bölünür."""
    if source.language == "python":
        tree = get_parser("python").parse(source.content.encode("utf-8"))
        lines = source.content.split("\n")
        return chunk_nodes(source, lines, tree.root_node.children, prefix="")
    return chunk_with_splitter(source)


def chunk_with_splitter(source: SourceFile) -> list[Chunk]:
    """Python dışındaki dosyalar için LangChain splitter: dilin sınıf/fonksiyon anahtar kelimelerinde keser."""
    language = SPLITTER_LANGUAGES.get(source.language)
    if language:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=MAX_CHUNK_CHARS, chunk_overlap=0, add_start_index=True
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=0, add_start_index=True)

    chunks = []
    for part in splitter.create_documents([source.content]):
        start_line = source.content.count("\n", 0, part.metadata["start_index"]) + 1
        chunks.append(
            Chunk(
                text=part.page_content,
                file_path=source.path.as_posix(),
                start_line=start_line,
                end_line=start_line + part.page_content.count("\n"),
                language=source.language,
            )
        )
    return chunks


def chunk_nodes(source: SourceFile, lines: list[str], nodes: list, prefix: str) -> list[Chunk]:
    """Bir düğüm listesini (dosyanın üst seviyesi ya da sınıfın gövdesi) parçalara böler.

    Fonksiyon ve sınıflar kendi parçalarını alır. Aralarındaki düz satırlar
    (import, sabit, yorum...) peş peşe geliyorsa tek parçada toplanır.
    """
    chunks = []
    loose = []  

    for node in nodes:
        definition = unwrap_decorator(node)
        if definition.type not in ("function_definition", "class_definition"):
            loose.append(node)
            continue

        if loose:
            chunks.append(make_chunk(source, lines, loose[0], loose[-1], prefix.rstrip(".") or None))
            loose = []
        chunks += chunk_definition(source, lines, node, definition, prefix)

    if loose:
        chunks.append(make_chunk(source, lines, loose[0], loose[-1], prefix.rstrip(".") or None))
    return chunks


def chunk_definition(source: SourceFile, lines: list[str], node, definition, prefix: str) -> list[Chunk]:
    """Tek bir fonksiyonu ya da sınıfı parçalara böler."""
    symbol = prefix + definition.child_by_field_name("name").text.decode("utf-8")
    whole = make_chunk(source, lines, node, node, symbol)

  
    if definition.type == "function_definition" or len(whole.text) <= MAX_CHUNK_CHARS:
        return [whole]


    members = definition.child_by_field_name("body").children
    first_def = next(
        (i for i, m in enumerate(members) if unwrap_decorator(m).type in ("function_definition", "class_definition")),
        None,
    )
    if first_def is None:  
        return [whole]

    header_end_row = members[first_def].start_point.row - 1
    header = make_chunk_from_rows(source, lines, node.start_point.row, header_end_row, symbol)
    return [header] + chunk_nodes(source, lines, members[first_def:], prefix=symbol + ".")


def unwrap_decorator(node):
    """@dekoratörlü tanımlarda asıl fonksiyon/sınıf düğümünü döndürür."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition")
    return node


def make_chunk(source: SourceFile, lines: list[str], first_node, last_node, symbol: str | None) -> Chunk:
    return make_chunk_from_rows(source, lines, first_node.start_point.row, last_node.end_point.row, symbol)


def make_chunk_from_rows(source: SourceFile, lines: list[str], start_row: int, end_row: int, symbol: str | None) -> Chunk:
    """start_row..end_row (0 tabanlı, ikisi de dahil) satırlarından parça üretir."""
    while end_row > start_row and not lines[end_row].strip():  
        end_row -= 1
    return Chunk(
        text="\n".join(lines[start_row : end_row + 1]),
        file_path=source.path.as_posix(),
        start_line=start_row + 1,
        end_line=end_row + 1,
        language=source.language,
        symbol=symbol,
    )
