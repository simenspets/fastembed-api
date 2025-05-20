from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI(title="FastEmbed MiniLM API")
embedder = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384-dim
    device="cpu"
)

class Inp(BaseModel):
    text: str

@app.post("/embed")
def embed(inp: Inp):
    vec = next(embedder.embed([inp.text]))       # ⇒ list[float]
    return {"embedding": vec}
