import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    chat_model: str = "gemini-3.8-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int= 5
    min_score: float= 0.0
    index_path: str = ".ragdocs/index.json"

def load_config(cwd: Path | None = None) -> Config:
    path = (cwd or Path.cwd()) / "ragdocs.toml"
    if not path.is_file():
        return Config()
    with path.open('rb') as f:
        data = tomllib.load(f)
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in known})


