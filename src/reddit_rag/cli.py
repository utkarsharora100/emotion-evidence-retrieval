from __future__ import annotations

import json
import typer

from .pipeline import Config, demo_frame, load_csv, load_kaggle, run_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def run(
    demo: bool = typer.Option(False, help="Use the explicit five-row demo fixture."),
    csv: str | None = typer.Option(None, help="Path to a downloaded Kaggle CSV."),
    kaggle_dataset: str | None = typer.Option(None, help="Kaggle dataset identifier OWNER/SLUG."),
    reddit_only: bool = typer.Option(True, help="Filter to Reddit rows when a source column exists."),
    limit: int | None = typer.Option(None, min=1),
    query: str = typer.Option("What do Reddit users say about games?"),
    qdrant_path: str | None = typer.Option(None, help="Persistent local Qdrant path; default is in-memory."),
):
    sources = sum(bool(x) for x in (demo, csv, kaggle_dataset))
    if sources != 1:
        raise typer.BadParameter("Choose exactly one of --demo, --csv, or --kaggle-dataset.")
    frame = demo_frame().head(limit) if demo else (load_csv(csv, reddit_only, limit) if csv else load_kaggle(kaggle_dataset, reddit_only, limit))
    if frame.empty:
        raise typer.BadParameter("The selected data source contains no usable rows.")
    result = run_pipeline(frame, query, Config(), qdrant_path)
    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
