# tests/test_api_for_text_model.py
import os
import pytest
from fastapi.testclient import TestClient

# Импортируем приложение
from api_for_text_model.api import app

# Установим устройство в CPU для тестов (избежим попыток задействовать GPU)
os.environ.setdefault("MODEL_DEVICE", "-1")


@pytest.fixture(scope="module")
def client():
    # используем TestClient как контекстный менеджер — это гарантирует запуск startup
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_predict_ok(client):
    payload = {"texts": ["I love this!", "Это плохой товар."]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"status {r.status_code}, body: {r.text}"
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item, original in zip(data, payload["texts"]):
        assert item["text"] == original
        assert isinstance(item.get("label"), str)
        assert isinstance(item.get("score"), float)


def test_predict_empty_list(client):
    payload = {"texts": []}
    r = client.post("/predict", json=payload)
    assert r.status_code == 400


@pytest.mark.parametrize("single", ["Hello world", "Плохо"])
def test_predict_single_text(client, single):
    r = client.post("/predict", json={"texts": [single]})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["text"] == single
