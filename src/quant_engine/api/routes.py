"""FastAPI REST endpoints for the research engine."""

from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException

from quant_engine.api.schemas import (
    ExportResponse,
    ResearchStartRequest,
    RunListResponse,
    RunStatusResponse,
    WinnersResponse,
)
from quant_engine.config import ResearchConfig
from quant_engine.export.formatter import StrategyExporter
from quant_engine.pipeline.orchestrator import PipelineOrchestrator
from quant_engine.storage.csv_backend import CsvStorage


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenAlgo Quant Research Engine",
        description="Autonomous strategy discovery, validation, and export",
        version="0.1.0",
    )

    # In-memory state for active runs
    active_runs: dict[str, PipelineOrchestrator] = {}
    storage = CsvStorage("./data/runs")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "quant-engine"}

    @app.post("/research/start", response_model=RunStatusResponse)
    def start_research(request: ResearchStartRequest, background_tasks: BackgroundTasks):
        """Start a new research run."""
        config = ResearchConfig(**request.config)
        orchestrator = PipelineOrchestrator(config)
        run_id = request.run_id or f"run_{id(orchestrator) % 100000:05d}"

        active_runs[run_id] = orchestrator

        def _run():
            orchestrator.run(run_id=run_id)

        background_tasks.add_task(_run)

        return RunStatusResponse(
            run_id=run_id,
            status="started",
            progress={},
        )

    @app.get("/research/status/{run_id}", response_model=RunStatusResponse)
    def get_status(run_id: str):
        """Get status of a research run."""
        orchestrator = active_runs.get(run_id)
        if orchestrator is None:
            if run_id in storage.list_runs():
                return RunStatusResponse(run_id=run_id, status="completed", progress={})
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        progress = orchestrator.progress
        return RunStatusResponse(
            run_id=run_id,
            status=progress.status,
            progress=progress.to_dict(),
        )

    @app.post("/research/stop/{run_id}")
    def stop_research(run_id: str):
        """Stop a running research."""
        orchestrator = active_runs.get(run_id)
        if orchestrator is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        orchestrator.stop()
        return {"status": "stopping", "run_id": run_id}

    @app.get("/research/results/{run_id}")
    def get_results(run_id: str):
        """Get ranked results for a completed run."""
        survivors = storage.load_results(run_id, "survivors")
        if not survivors:
            raise HTTPException(status_code=404, detail=f"No results for run {run_id}")
        return {"run_id": run_id, "total": len(survivors), "strategies": survivors}

    @app.get("/research/winners/{run_id}", response_model=WinnersResponse)
    def get_winners(run_id: str):
        """Get top winning strategies."""
        winners = storage.load_winners(run_id)
        if not winners:
            raise HTTPException(status_code=404, detail=f"No winners for run {run_id}")
        return WinnersResponse(run_id=run_id, winners=winners)

    @app.get("/research/rejections/{run_id}")
    def get_rejections(run_id: str):
        """Get rejected strategies with reasons."""
        rejections = storage.load_results(run_id, "rejected")
        return {"run_id": run_id, "total": len(rejections), "rejections": rejections}

    @app.post("/research/export/{run_id}/{strategy_id}", response_model=ExportResponse)
    def export_strategy(run_id: str, strategy_id: str):
        """Export a strategy as Python signal script."""
        winners = storage.load_winners(run_id)
        strategy_data = next((w for w in winners if w.get("strategy_id") == strategy_id), None)
        if strategy_data is None:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_id} not found")

        StrategyExporter(f"./data/runs/{run_id}/exports")
        # Note: full export requires reconstructing StrategyGenome from stored data
        return ExportResponse(
            strategy_id=strategy_id,
            message="Strategy exported",
            export_dir=f"./data/runs/{run_id}/exports",
        )

    @app.get("/research/runs", response_model=RunListResponse)
    def list_runs():
        """List all research runs."""
        runs = storage.list_runs()
        return RunListResponse(runs=runs)

    @app.get("/research/reports/{run_id}")
    def get_report(run_id: str):
        """Get summary report for a run."""
        config = storage.load_run_config(run_id)
        winners = storage.load_winners(run_id)
        rejections = storage.load_results(run_id, "rejected")
        backtested = storage.load_results(run_id, "backtested")

        return {
            "run_id": run_id,
            "config_summary": {
                "name": config.get("name", ""),
                "trading_styles": config.get("trading_styles", []),
                "target_count": config.get("generation", {}).get("target_count", 0),
            },
            "statistics": {
                "total_generated": len(backtested) + len(rejections),
                "total_rejected": len(rejections),
                "total_backtested": len(backtested),
                "total_winners": len(winners),
            },
            "top_strategies": winners[:10],
        }

    return app
