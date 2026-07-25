"""
Shared evaluation report schema.

Every evaluation script in this package writes an EvalReport instead of
inventing its own report format, so results are directly comparable across
phases (NER, embeddings, retrieval, RAG, KG — see ROADMAP.md Track C/E).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import BaseModel, Field


class EvalReport(BaseModel):
    eval_name: str = Field(description="e.g. 'ner_v1', 'retrieval_v1', 'rag_v1'")
    run_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_size: int = Field(ge=0, description="Number of examples the metrics were computed over")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description='e.g. {"precision": 0.81, "recall": 0.74}'
    )
    notes: Optional[str] = None

    def to_json_dict(self) -> dict:
        payload = self.model_dump()
        payload["run_at"] = self.run_at.isoformat()
        return payload
