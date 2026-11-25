import click
from .config import load_config
from .embedder import GeminiEmbedder
from .ingest import ingest as run_ingest
from .llm import generate, make_client
from .rag import EmptyIndexError, answer_question
from .store import IndexMisMatch, VectorStore

def _setup():
    cfg = load_config()
    try: 
        client = make_client()
        store = VectorStore.load(cfg.index_path, cfg.embed_model, cfg.embed_dim)
    except(RuntimeError, IndexMisMatch) as exc:
        raise click.ClickException(str(exc))
    embedder = GeminiEmbedder(client, cfg.embed_model, cfg.embed_dim)
    return cfg, client, store, embedder

def _print_answer(ans, show_scores: bool):
    click.echo("\n" + ans.text + "\n")
    if ans.sources:
        click.echo("sources")
        for s in ans.sources:
            mark = " <- cited " if s.cited else ""
            score = f"  (score {s.score:.3f})" if show_scores else ""
            click.echo(f"  [{s.n}] {s.label}{score}{mark}")
        click.echo("")

def _ask(question, cfg, client, store, embedder, k, show_scores):
    try:
        ans = answer_question(
            question, store, embedder,
            lambda prompt: generate(client, cfg.chat_model, prompt),
            k=k or cfg.top_k, min_score=cfg.min_score,
            )
    except EmptyIndexError as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:  # network, quota, bad key...
        raise click.ClickException(f"Gemini API call failed: {exc}")
    _print_answer(ans, show_scores)

@click.group()
def main():
    """Chat with your own documents."""

@main.command()
@click.argument("path", type=click.Path(exists=True))
def ingest(path):
    cfg, _client, store, embedder= _setup()
    try:
        rep = run_ingest(
            path, store, embedder, cfg.chunk_size, cfg.chunk_overlap,
            on_batch=lambda: store.save(cfg.index_path),
        )
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))
    except Exception as exc:
        raise click.ClickException(
            f"Ingest stopped: {exc}\nProgress was saved. Run the same command again to resume."
        ) 
    store.save(cfg.index_path)
    click.echo(
        f"Read{rep.documents} document(s)/page(s): {rep.chunks_found} chunks, "
        f"{rep.chunks_new} new, {rep.chunks_skipped} already indexed."
    )   
    if rep.documents== 0:
        click.echo("Nothing readable found. Scanned PDFs have no text layer and need OCR.")

@main.command()
@click.argument("question")
@click.option("--k", type=int, default=None, help="How many passages need to retrieve")
@click.option("--scores", is_flag=True, help="Similarity scores")
def ask(question, k, scores):
    cfg, client, store, embedder = _setup()
    _ask(question, cfg, client, store, embedder, k, scores)


@main.command()
@click.option("--scores", is_flag=True)
def chat(scores):
    """Interactive Q&A. Each question is answered independently. Ctrl+C to quit."""
    cfg, client, store, embedder = _setup()
    click.echo(f"{len(store)} chunks indexed. Type a question (empty line to quit).")
    while True:
        q = click.prompt("you", default="", show_default=False).strip()
        if not q:
            break
        _ask(q, cfg, client, store, embedder, None, scores)

@main.command()
def stats():
    """Show what is indexed."""
    cfg = load_config()
    try:
        store = VectorStore.load(cfg.index_path, cfg.embed_model, cfg.embed_dim)
    except IndexMisMatch as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Index: {cfg.index_path}  ({len(store)} chunks, {cfg.embed_model}, {cfg.embed_dim} dims)")
    for name, n in store.sources().items():
        click.echo(f"  {n:>4}  {name}")


if __name__ == "__main__":
    main()