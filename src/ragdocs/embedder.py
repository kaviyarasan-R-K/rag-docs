import math
import time

from .retry import with_retry

class EmbeddingError(Exception):
    pass

def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return list(v)
    return [x / norm for x in v]  

class GeminiEmbedder:
    def __init__(self, client, model= "gemini-embedding-01", dim = 768,
                 batch_size = 32, pause = 0.0, sleep =time.sleep):
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.pause = pause
        self.sleep = sleep 

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        result = with_retry(
            lambda: self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config={"task_type": task_type, "output_dimensionality": self.dim},
        ),
            sleep=self.sleep,
        )
        vectors = [normalize(list(e.values)) for e in result.embeddings]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i : i + self.batch_size], "RETRIEVAL_DOCUMENT"))
            if self.pause:
                self.sleep(self.pause)  # gentle on free-tier per-minute limits
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]