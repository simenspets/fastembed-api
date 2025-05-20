from fastapi import FastAPI
from fastembed import TextEmbedding
import os

# ── 1. Modell & instans  ────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDER   = TextEmbedding(model_name=MODEL_NAME, device="cpu")

# Finn dimensjon (prøv felt → ellers én dummy-embedding via next())
if hasattr(EMBEDDER, "embedding_dimension"):
    DIM = EMBEDDER.embedding_dimension
elif hasattr(EMBEDDER, "dimension"):
    DIM = EMBEDDER.dimension
else:
    DIM = len(next(EMBEDDER.embed(["dummy"])))


# ── 2. FastAPI-app  ─────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
def health():
    """
    Helsesjekk som Render pinger.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dim": DIM,
    }


@app.post("/embed")
def embed(texts: list[str]):
    """
    Gi en liste tekster → får tilbake liste embeddings.
    """
    return list(EMBEDDER.embed(texts))   # konverter generator → liste
