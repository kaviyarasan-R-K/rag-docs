from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}

@dataclass
class Document:
    source: str
    page: int| None
    text: str

def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return [(i, page.extract_text()) for i, page in enumerate(reader.pages, 1)]

def _load_file(path: Path, base: Path) -> list[Document]:
    name = str(path.relative_to(base)) if base != path else path
    if path.suffix == ".pdf":
        return [Document(name, n, t) for n, t in _pdf_pages(path) if t.strip()]
    text = path.read_text(encoding = "utf-8", errors = "ignore")
    return [Document(name, None, text)] if text.strip() else []

def load_path(path :Path | str) -> list[Document]:
    p = Path(path)
    if p.is_dir():
        files = sorted(
            f
            for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED
            and not any(part.startswith(".") for part in f.relative_to(p).parts)
        )
        base = p
    elif p.is_file():
        if p.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported file type '{p.suffix}'. Use: {', '.join(sorted(SUPPORTED))}")
        files, base = [p], p.parent
    else:
        raise FileNotFoundError(f"Path does not exist: {p}")
    
    docs: list[Document] = []
    for f in files:
        docs.extend(_load_file(f, base))
    return docs
