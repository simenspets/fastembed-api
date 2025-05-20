from fastapi import FastAPI
from fastembed import TextEmbedding
import os

# ── 1. Modell & instans  ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDER   = TextEmbedding(model_name=MODEL_NAME, device="cpu")

# Finn dimensjonen (prøv felt først → ellers beregn).
if hasattr(EMBEDDER, "embedding_dimension"):
    DIM = EMBEDDER.embedding_dimension
elif hasattr(EMBEDDER, "dimension"):
    DIM = EMBEDDER.dimension
else:
    # fallback – embed én dummy-tekst og mål lengden på vektoren
    DIM = len(EMBEDDER.embed(["fastembed"])[0])

# ── 2. FastAPI-app  ─────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
def health():
    """
    En enkel helsesjekk som Render kan pinge.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dim": DIM,
    }


@app.post("/embed")
def embed(texts: list[str]):
    """
    Gi meg en liste tekster → tilbake kommer en liste embeddings.
    """
    return EMBEDDER.embed(texts)
