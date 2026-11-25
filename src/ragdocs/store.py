import heapq
import json
import os
from pathlib import Path

from .chunker import Chunk

class IndexMisMatch(RuntimeError):
    pass

class VectorStore:
    def __init__(self, model: str, dim: int):
        self.model, self.dim = model,dim
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def __len__(self):
        return len(self._chunks)

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def add(self, chunk: Chunk, vector: list[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"Vector {len(vector)} stores except {self.dim}")
        self._chunks[chunk.id] = chunk
        self._vectors[chunk.id] = vector

    def search(self, query: list[float], k: int = 5) -> list[tuple[float, Chunk]]:
        scored = (
            (sum(a * b for a, b in zip(query, vec)), cid)
            for cid, vec in self._vectors.items()
        )
        return [(score, self._chunks[cid]) for score, cid in heapq.nlargest(k, scored)]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._chunks.values():
            counts[c.source] = counts.get(c.source, 0) + 1
        return dict(sorted(counts.items()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "dim": self.dim,
            "items": [
                {
                    "id": c.id, "source": c.source, "page": c.page, "text": c.text,
                    "vector": [round(x, 6) for x in self._vectors[c.id]],
                }
                for c in self._chunks.values()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written index

    @classmethod
    def load(cls, path: str | Path, model: str, dim: int) -> "VectorStore":
        store = cls(model, dim)
        path = Path(path)
        if not path.is_file():
            return store
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["model"] != model or data["dim"] != dim:
            raise IndexMisMatch(
                f"Index was built with {data['model']} ({data['dim']} dims) but config says "
                f"{model} ({dim} dims). Delete {path} and re-ingest, or fix the config."
            )
        for it in data["items"]:
            store.add(Chunk(it["id"], it["source"], it["page"], it["text"]), it["vector"])
        return store