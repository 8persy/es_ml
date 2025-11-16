# eval_metrics.py
import requests
import pandas as pd
import time
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    log_loss, mean_absolute_error
)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import re

API_URL = "http://localhost:8000/predict"
BATCH_SIZE = 32
CSV_PATH = "./data.csv"

def parse_label(label_str):
    m = re.search(r'\d+', str(label_str))
    if m:
        return int(m.group(0))
    else:
        raise ValueError(f"Не удалось извлечь число из метки: {label_str}")

df = pd.read_csv(CSV_PATH)
texts = df['text'].astype(str).tolist()
y_true = [parse_label(x) for x in df['label'].tolist()]

pred_labels = []
pred_scores = []

latencies = []
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    t0 = time.time()
    r = requests.post(API_URL, json={"texts": batch})
    t1 = time.time()
    latencies.append((t1 - t0) / len(batch))
    r.raise_for_status()
    resp = r.json()
    for item in resp:
        pred_labels.append(parse_label(item.get('label')))
        pred_scores.append(float(item.get('score', 0.0)))

acc = accuracy_score(y_true, pred_labels)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, pred_labels, average='macro', zero_division=0
)
prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
    y_true, pred_labels, average='weighted', zero_division=0
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, pred_labels, average='micro', zero_division=0
)

labels = sorted(list(set(y_true) | set(pred_labels)))  # числа 1-5
cm = confusion_matrix(y_true, pred_labels, labels=labels)

try:
    Y_true_bin = LabelBinarizer().fit_transform(y_true)
    Y_pred_prob = np.full((len(y_true), len(labels)), 1e-9)
    for i, pl in enumerate(pred_labels):
        j = labels.index(pl)
        Y_pred_prob[i, j] = pred_scores[i] if pred_scores[i] > 0 else 1e-9
    Y_pred_prob = Y_pred_prob / Y_pred_prob.sum(axis=1, keepdims=True)
    ll = log_loss(y_true, Y_pred_prob, labels=labels)
except:
    ll = None

lat_ms = np.array(latencies) * 1000.0
lat_report = {
    "p50_ms": float(np.percentile(lat_ms, 50)),
    "p95_ms": float(np.percentile(lat_ms, 95)),
    "p99_ms": float(np.percentile(lat_ms, 99)),
    "mean_ms": float(np.mean(lat_ms))
}

mae = mean_absolute_error(y_true, pred_labels)

metrics = {
    "accuracy": acc,
    "precision_macro": prec_macro,
    "recall_macro": rec_macro,
    "f1_macro": f1_macro,
    "precision_weighted": prec_weight,
    "recall_weighted": rec_weight,
    "f1_weighted": f1_weight,
    "precision_micro": prec_micro,
    "recall_micro": rec_micro,
    "f1_micro": f1_micro,
    "log_loss": ll,
    "mae_if_ordinal": mae,
    "latency": lat_report,
    "n_samples": len(y_true),
    "labels": labels
}

with open("metrics.json", "w", encoding="utf8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print("Wrote metrics.json and confusion_matrix.png")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
