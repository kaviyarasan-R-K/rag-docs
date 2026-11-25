import hashlib
from dataclasses import dataclass

SEPARATORS = ["\n\n","\n",". "," "]

@dataclass
class Chunk:
    id: str
    source: str
    page: int| None
    text: str

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if not 0 <= overlap < size // 2:
        raise ValueError("Overlap must be between 0 and half the chunk size")

    text = text.strip()
    chunks: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + size, n)
        if end < n:

            window_start = start + int(size * 0.6)
            for sep in SEPARATORS:
                i = text.rfind(sep, window_start, end)
                if i != -1:
                    end = i + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks

def chunk_id(source: str, page: int| None, text: str) -> str:
    raw = f"{source}|{page}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
    