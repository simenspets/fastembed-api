# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastembed import TextEmbedding

# ────────────────────────── modellvalg ────────────────────────────
MODEL_NAME = "Supabase/gte-small"      # flerspråklig, 384-dim
EMBEDDER   = TextEmbedding(model_name=MODEL_NAME, device="cpu")

app = FastAPI(title="FastEmbed-API", version="1.0")


# ────────────────────────── datamodeller ──────────────────────────
class EmbedRequest(BaseModel):
    text: str
    mode: str | None = "query"         # "query"  eller "passage"


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int  = 384
    model: str = MODEL_NAME
    mode: str


# ───────────────────────────── routes ─────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse, tags=["embed"])
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(400, detail="text must not be empty")

    # === ❶ «Tekst-prefiks» ======================================
    #
    # GTE / E5-modellene ble trent med et spesifikt start-prefiks:
    #   •  bruker-spørsmål   ->  "query: …"
    #   •  dokument-chunk    ->  "passage: …"
    #
    #  Følger du dette mønsteret får du 5-15 % høyere nøyaktighet.
    #
    prefix = "query: " if req.mode == "query" else "passage: "
    text_for_model = prefix + req.text

    vec = EMBEDDER.embed([text_for_model])[0]          # np.array
    return EmbedResponse(embedding=vec.tolist(), mode=req.mode)
