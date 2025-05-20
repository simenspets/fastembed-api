from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

MODEL_NAME = "all-MiniLM-L6-v2"          # ✅ gyldig hos fastembed

EMBEDDER = TextEmbedding(
    model_name=MODEL_NAME,
    device="cpu",          # Render-free har ingen GPU
    batch_size=32,         # juster hvis minneproblemer
)

class EmbedIn(BaseModel):
    text: str

class EmbedOut(BaseModel):
    vector: list[float]

@app.post("/embed", response_model=EmbedOut)
def embed(req: EmbedIn):
    vec = next(EMBEDDER.embed([req.text]))
    return {"vector": vec.tolist()}

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}
