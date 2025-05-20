from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384 dim – støttes i fastembed 0.7.0
EMBEDDER   = TextEmbedding(model_name=MODEL_NAME, device="cpu")

app = FastAPI(title="FastEmbed-API", version="1.0")


class EmbedRequest(BaseModel):
    text: str
    mode: str | None = "query"          # "query" | "passage"


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    model: str
    mode: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dim": EMBEDDER.dimension,        # 384 når du bruker bge-small-en-v1.5
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    prefix = "query: " if req.mode == "query" else "passage: "
    vec: np.ndarray = EMBEDDER.embed([prefix + req.text])[0]
    return EmbedResponse(
        embedding=vec.tolist(),
        dim=vec.size,
        model=MODEL_NAME,
        mode=req.mode,
    )
