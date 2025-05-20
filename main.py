from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding

app = FastAPI()

# -- last ned én gang ved oppstart
EMBEDDER = TextEmbedding(model_name="qdrant/all-MiniLM-L6-v2-onnx",
                         device="cpu",  # Render free-tier har ikke GPU
                         dtype="float32")               # reduserer RAM

class EmbedIn(BaseModel):
    text: str                     # én streng inn

class EmbedOut(BaseModel):
    vector: list[float]           # JSON-vennlig liste ut

@app.post("/embed", response_model=EmbedOut)
def embed(req: EmbedIn):
    # fastembed gir en generator → next(...)
    vec = next(EMBEDDER.embed([req.text]))
    return {"vector": vec.tolist()}        # tolist() gjør den JSON-klar

@app.get("/")
def root():
    return {"status": "ok"}
