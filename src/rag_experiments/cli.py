from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import typer

from .ask import ask as ask_fn
from .config import load_config
from .index import build_or_update_index

load_dotenv()

app = typer.Typer(no_args_is_help=True)


@app.command()
def index(
    config: Path = typer.Option(Path("configs/baseline.yaml")),
    force: bool = typer.Option(False, help="Delete and rebuild the collection."),
) -> None:
    cfg = load_config(config)
    n = build_or_update_index(cfg, force=force)
    typer.echo(f"Indexed {n} chunks into Chroma at: {cfg.persist_dir}")


@app.command()
def ask(
    question: str = typer.Argument(...),
    config: Path = typer.Option(Path("configs/baseline.yaml")),
    show_context: bool = typer.Option(True),
) -> None:
    cfg = load_config(config)
    out = ask_fn(cfg, question)
    typer.echo(out.text)
    if show_context:
        typer.echo("\n\n--- CONTEXTS ---\n")
        for i, c in enumerate(out.contexts, start=1):
            typer.echo(f"[{i}]\n{c}\n")
