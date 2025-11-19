import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean
import time
import numpy as np

from llm import summarize_to_keyword

with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

y_true = []
y_pred = []
latencies = []

for item in data:
    text = item["text"]
    label = item["label"].lower()

    t0 = time.time()
    pred = summarize_to_keyword(text).lower()
    t1 = time.time()

    y_true.append(label)
    y_pred.append(pred)
    latencies.append((t1 - t0) * 1000)

print(y_true)
print(y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

results = {
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted,
    "latency": {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(mean(latencies))
    },
    "n_samples": len(data)
}

import json
print(json.dumps(results, indent=2, ensure_ascii=False))
