from __future__ import annotations

import os
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from news_ingest.ml.text import chunks_from_token_offsets
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")
DEVICE = os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = os.getenv("EMBEDDING_DTYPE", "bfloat16")

app = FastAPI(title="News RAG Embedding Service")
_model: SentenceTransformer | None = None


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    input_type: Literal["document", "query"] = "document"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    device: str
    embeddings: list[list[float]]


class ChunkRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    max_tokens: int = 500
    overlap_tokens: int = 50


class ChunkResponse(BaseModel):
    model: str
    tokenizer: str
    chunks: list[list[dict[str, object]]]


def _torch_dtype():
    if DTYPE == "bfloat16" and DEVICE.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE, model_kwargs={"torch_dtype": _torch_dtype()})
    return _model


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": _model is not None, "model": MODEL_NAME, "device": DEVICE, "dtype": DTYPE, "torch_version": torch.__version__}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    model = load_model()
    if request.input_type == "query" and hasattr(model, "encode_query"):
        embeddings = model.encode_query(request.texts, convert_to_numpy=True, normalize_embeddings=True)
    elif hasattr(model, "encode_document"):
        embeddings = model.encode_document(request.texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        embeddings = model.encode(request.texts, convert_to_numpy=True, normalize_embeddings=True)
    values = embeddings.tolist()
    dim = len(values[0]) if values else 0
    return EmbedResponse(model=MODEL_NAME, dim=dim, device=DEVICE, embeddings=values)


@app.post("/chunk", response_model=ChunkResponse)
def chunk(request: ChunkRequest) -> ChunkResponse:
    model = load_model()
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="loaded embedding model does not expose a tokenizer")

    all_chunks: list[list[dict[str, object]]] = []
    for text in request.texts:
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"tokenizer does not support offset mapping: {exc}") from exc
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise HTTPException(status_code=500, detail="tokenizer response missing offset_mapping")
        chunks = chunks_from_token_offsets(text, list(offsets), max_tokens=request.max_tokens, overlap_tokens=request.overlap_tokens)
        all_chunks.append(
            [
                {
                    "index": item.index,
                    "text": item.text,
                    "char_start": item.char_start,
                    "char_end": item.char_end,
                    "token_count": item.token_count,
                    "content_hash": item.content_hash,
                }
                for item in chunks
            ]
        )

    tokenizer_name = tokenizer.__class__.__name__
    return ChunkResponse(model=MODEL_NAME, tokenizer=tokenizer_name, chunks=all_chunks)
