from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from api_for_text_model.model import SentimentModel


app = FastAPI(title="Sentiment API", version="1.0")


class PredictRequest(BaseModel):
	texts: List[str]


class PredictResponseItem(BaseModel):
	text: str
	label: str
	score: float


@app.on_event("startup")
def load_model():
# Определяем устройство: по умолчанию CPU. Если нужно GPU, установите ENV CUDA_VISIBLE_DEVICES
	device_env = os.environ.get("MODEL_DEVICE", "cpu").lower()
	if device_env in ("cpu", "-1"):
		device = -1
	else:
		try:
			device = int(device_env)
		except Exception:
			device = -1
	app.state.model = SentimentModel(device=device)


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