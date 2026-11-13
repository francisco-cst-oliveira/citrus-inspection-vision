from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import Config
from .model import TorchScriptScorer
from .runtime import run_live

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
error_console = Console(stderr=True)


@app.command()
def live(
    model: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    source: str = typer.Option("0", "--source", help="Camera index or local video path."),
    threshold: float | None = typer.Option(None, min=0.0, max=1.0),
    config: Path = typer.Option(
        Path("configs/inference.yaml"), exists=True, file_okay=True, dir_okay=False, readable=True
    ),
) -> None:
    """Open the inspection dashboard for a camera or local video."""
    try:
        cfg = Config.load(config)
        score_threshold = cfg.runtime.threshold if threshold is None else threshold
        scorer = TorchScriptScorer(model, score_threshold, cfg.runtime.device)
        console.print(run_live(cfg, scorer, source).as_dict())
    except KeyboardInterrupt as exc:
        raise typer.Exit(130) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
