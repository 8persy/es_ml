from transformers import pipeline
import os

MODEL_NAME = os.environ.get("HF_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")


class SentimentModel:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        self.pipeline = pipeline( # type: ignore
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )


    def predict(self, texts):
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        results = self.pipeline(list(texts))
        out = []
        for t, r in zip(texts, results):
            out.append({
                "text": t,
                "label": r.get("label"),
                "score": float(r.get("score", 0.0)),
            })
        return out