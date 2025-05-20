from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

MODEL_NAME = "Supabase/gte-small"       # 384-dim, flerspråklig
EMBEDDER   = TextEmbedding(model_name=MODEL_NAME, device="cpu")

app = FastAPI(title="FastEmbed-API", version="1.0")


class EmbedRequest(BaseModel):
    text: str
    mode: str | None = "query"          # "query" | "passage"


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int = 384
    model: str = MODEL_NAME
    mode: str


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(400, detail="text must not be empty")

    prefix = "query: " if req.mode == "query" else "passage: "
    vec = EMBEDDER.embed([prefix + req.text])[0]       # np.ndarray
    return EmbedResponse(embedding=vec.tolist(), mode=req.mode)
