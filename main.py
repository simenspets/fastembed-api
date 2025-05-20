# main.py
from __future__ import annotations

import os
from typing import List, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from fastembed import TextEmbedding

###############################################################################
# 1. Modell- og embedder-oppsett
###############################################################################

MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

try:
    EMBEDDER = TextEmbedding(model_name=MODEL_NAME, device="cpu")
except Exception as e:  # pragma: no cover
    # Feiler typisk bare hvis modellnavnet er ugyldig
    raise RuntimeError(f"Kunne ikke laste modellen '{MODEL_NAME}': {e}") from e

# fastembed 0.7 har ikke en stabil 'dimension'-property, så vi sjekker dynamisk
try:
    EMBEDDING_DIM = EMBEDDER.embedding_dimension  # fungerer i 0.8+
except AttributeError:  # 0.7
    EMBEDDING_DIM = len(next(EMBEDDER.embed(["dummy"])))

###############################################################################
# 2. Pydantic-schemas
###############################################################################

class EmbedRequest(BaseModel):
    """
    Request-body:
        {
          "text": "én streng"
        }
        – eller –
        {
          "text": ["streng 1", "streng 2", ...]
        }
    """
    text: Union[str, List[str]] = Field(
        ...,
        description="Streng eller liste med strenger som skal embeddes",
        example=["Hva krever NORSOK om sveising?", "Hvordan sveiser jeg P235GH-rør?"],
    )

    # Sørg for at vi ender opp med en liste internt
    @validator("text", pre=True)
    def _ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and all(isinstance(s, str) for s in v):
            return v
        raise ValueError("text må være en streng eller en liste av strenger")


class EmbedResponse(BaseModel):
    """Respons-format – én vektor pr. input-streng i samme rekkefølge."""
    model: str
    dimension: int
    vectors: List[List[float]]

###############################################################################
# 3. FastAPI-app
###############################################################################

app = FastAPI(
    title="fastembed-api",
    version="1.0",
    summary="Lettvekts tekst-embedding som REST-tjeneste",
)

# CORS (valgfritt – åpner for alt; lås gjerne ned domenene dine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health():
    """Rask helsesjekk."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": EMBEDDING_DIM,
    }


@app.post("/embed", response_model=EmbedResponse, tags=["embed"])
def embed(req: EmbedRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text kan ikke være tom")

    # Embedder returnerer en generator med numpy-arrayer; konverter til rene Python-lister
    vectors = [vec.tolist() for vec in EMBEDDER.embed(req.text)]

    return EmbedResponse(
        model=MODEL_NAME,
        dimension=EMBEDDING_DIM,
        vectors=vectors,
    )

###############################################################################
# 4. Lokalkjøring:  uvicorn main:app --reload
###############################################################################
