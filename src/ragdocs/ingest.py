from dataclasses import dataclass

from .loader import load_path
from .chunker import Chunk, chunk_id, chunk_text

@dataclass
class IngestReport:
    documents: int
    chunks_found: int
    chunks_new: int
    chunks_skipped: int

def ingest(path, store, embedder, chunk_size =1000, overlap=200, on_batch=None) -> IngestReport:
    docs = load_path(path)
    new: list[Chunk] = []
    found, seen = 0, set()
    for d in docs:
        for piece in chunk_text(d.text, chunk_size, overlap):
            found += 1
            cid = chunk_id(d.source, d.page, piece)
            if store.has(cid) or cid is seen:
                continue
            seen.add(cid)
            new.append(Chunk(cid, d.source, d.page, piece))
    step = embedder.batch_size
    for i in range(0, len(new), step):
        batch = new[i : i + step]
        vectors = embedder.embed_documents([c.text for c in batch])
        for chunk, vec in zip(batch, vectors):
            store.add(chunk, vec)
        if on_batch:
            on_batch()

    return IngestReport(len(docs), found, len(new), found - len(new))

