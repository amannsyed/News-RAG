from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from gliner import GLiNER
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("NER_MODEL_NAME", "urchade/gliner_multi-v2.1")
DEVICE = os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="News RAG NER Service")
_model: GLiNER | None = None


class NerRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    threshold: float = 0.5


class NerResponse(BaseModel):
    model: str
    device: str
    results: list[list[dict[str, object]]]


def load_model() -> GLiNER:
    global _model
    if _model is None:
        _model = GLiNER.from_pretrained(MODEL_NAME)
        _model.to(DEVICE)
    return _model


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": _model is not None, "model": MODEL_NAME, "device": DEVICE, "torch_version": torch.__version__}


@app.post("/ner", response_model=NerResponse)
def ner(request: NerRequest) -> NerResponse:
    model = load_model()
    results = []
    for text in request.texts:
        entities = model.predict_entities(text, request.labels, threshold=request.threshold)
        normalized = []
        for entity in entities:
            normalized.append(
                {
                    "text": entity.get("text"),
                    "label": str(entity.get("label", "")).upper(),
                    "score": float(entity.get("score", 0.0)),
                    "start": int(entity.get("start", 0)),
                    "end": int(entity.get("end", 0)),
                }
            )
        results.append(normalized)
    return NerResponse(model=MODEL_NAME, device=DEVICE, results=results)
