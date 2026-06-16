"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel


class ResearchStartRequest(BaseModel):
    config: dict
    run_id: str | None = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    progress: dict


class WinnersResponse(BaseModel):
    run_id: str
    winners: list[dict]


class ExportResponse(BaseModel):
    strategy_id: str
    message: str
    export_dir: str


class RunListResponse(BaseModel):
    runs: list[str]
