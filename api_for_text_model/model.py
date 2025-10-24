from transformers import pipeline
import os

MODEL_NAME = os.environ.get("HF_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")


class SentimentModel:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
    # device=-1 -> CPU, device>=0 -> GPU index (if available)
        self.pipeline = pipeline( # type: ignore
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )


    def predict(self, texts):
        # texts: list[str]
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        # pipeline returns list of {'label': str, 'score': float}
        results = self.pipeline(list(texts))
        # Normalize to predictable JSON-friendly format
        out = []
        for t, r in zip(texts, results):
            out.append({
                "text": t,
                "label": r.get("label"),
                "score": float(r.get("score", 0.0)),
            })
        return out