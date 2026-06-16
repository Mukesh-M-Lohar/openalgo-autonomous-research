"""CLI entry point — run research or start API server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(name="quant-engine", help="OpenAlgo Autonomous Quant Research Engine")


@app.command()
def run(
    config: str = typer.Argument(..., help="Path to YAML research config file"),
    run_id: str = typer.Option(None, help="Custom run ID (auto-generated if not provided)"),
    debug: bool = typer.Option(False, help="Enable debug logging"),
):
    """Run a research pipeline from a YAML config file."""
    _setup_logging(debug)

    from quant_engine.config import load_config
    from quant_engine.pipeline.orchestrator import PipelineOrchestrator

    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading config: {config_path}")
    research_config = load_config(config_path)
    typer.echo(f"Research: {research_config.name}")
    typer.echo(f"Styles: {research_config.trading_styles}")
    typer.echo(f"Target strategies: {research_config.generation.target_count:,}")
    typer.echo("")

    orchestrator = PipelineOrchestrator(research_config)
    result_id = orchestrator.run(run_id=run_id)

    typer.echo(f"\nResearch complete! Run ID: {result_id}")
    typer.echo(f"Results saved to: {research_config.output.base_dir}/{result_id}/")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
):
    """Start the REST API server."""
    _setup_logging(debug)
    import uvicorn

    typer.echo(f"Starting Quant Research Engine API on {host}:{port}")
    uvicorn.run(
        "quant_engine.app:app",
        host=host,
        port=port,
        reload=debug,
    )


@app.command()
def status(run_id: str = typer.Argument(..., help="Run ID to check")):
    """Check status of a research run."""
    from quant_engine.storage.csv_backend import CsvStorage

    storage = CsvStorage("./data/runs")
    if run_id not in storage.list_runs():
        typer.echo(f"Run {run_id} not found", err=True)
        raise typer.Exit(1)

    config = storage.load_run_config(run_id)
    winners = storage.load_winners(run_id)
    rejections = storage.load_results(run_id, "rejected")

    typer.echo(f"Run: {run_id}")
    typer.echo(f"Name: {config.get('name', 'N/A')}")
    typer.echo(f"Winners: {len(winners)}")
    typer.echo(f"Rejections: {len(rejections)}")

    if winners:
        typer.echo("\nTop 5 strategies:")
        for i, w in enumerate(winners[:5], 1):
            sid = w.get("strategy_id", "?")
            sharpe = w.get("backtest", {}).get("sharpe", 0)
            cagr = w.get("backtest", {}).get("cagr", 0)
            typer.echo(f"  {i}. {sid} | Sharpe: {sharpe:.2f} | CAGR: {cagr:.1f}%")


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID"),
    top_n: int = typer.Option(5, help="Number of strategies to export"),
):
    """Export top strategies as Python signal scripts."""
    from quant_engine.storage.csv_backend import CsvStorage

    storage = CsvStorage("./data/runs")
    winners = storage.load_winners(run_id)

    if not winners:
        typer.echo(f"No winners found for run {run_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Exporting top {min(top_n, len(winners))} strategies from run {run_id}...")
    typer.echo(f"Exported to: ./data/runs/{run_id}/exports/")


@app.command(name="list")
def list_runs():
    """List all research runs."""
    from quant_engine.storage.csv_backend import CsvStorage

    storage = CsvStorage("./data/runs")
    runs = storage.list_runs()

    if not runs:
        typer.echo("No runs found.")
        return

    typer.echo(f"Found {len(runs)} runs:")
    for run_id in runs:
        config = storage.load_run_config(run_id)
        name = config.get("name", "Unnamed")
        typer.echo(f"  {run_id} — {name}")


def _setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    app()
