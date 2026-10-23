from typing import Literal
from pydantic import BaseModel, Field


class ModelAnswer(BaseModel):
    question_id: str
    status: Literal[
        "entailed",
        "contradicted",
        "both",
        "insufficient",
    ]
    answer: str | None
    evidence_ids: list[str]
    counterevidence_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    explanation: str
