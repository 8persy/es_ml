# api_for_text_model/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from contextlib import asynccontextmanager
import logging
import anyio

from api_for_text_model.model import SentimentModel

logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    texts: List[str]


class PredictResponseItem(BaseModel):
    text: str
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    device_env = os.environ.get("MODEL_DEVICE", "cpu").lower()
    if device_env in ("cpu", "-1"):
        device = -1
    else:
        try:
            device = int(device_env)
        except Exception:
            device = -1

    try:
        model = await anyio.to_thread.run_sync(lambda: SentimentModel(device=device))
        app.state.model = model
        logger.info("SentimentModel loaded, device=%s", device)
    except Exception:
        logger.exception("Failed to load SentimentModel during startup")
        raise

    try:
        yield
    finally:
        try:
            if hasattr(app.state, "model"):
                del app.state.model
                logger.info("SentimentModel removed from app.state on shutdown")
        except Exception:
            logger.exception("Error while cleaning up SentimentModel")


app = FastAPI(title="Sentiment API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=List[PredictResponseItem])
def predict(req: PredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="`texts` list is empty")
    model = app.state.model
    results = model.predict(req.texts)
    return results
