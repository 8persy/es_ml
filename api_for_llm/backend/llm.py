import uvicorn
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import os
from dotenv import load_dotenv
from langfuse import observe, Langfuse

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["DISABLE_TF_IMPORT"] = "1"

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

NEWS_API_KEY = os.getenv("API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

NEWS_API_URL = "https://newsapi.org/v2/everything"

class InRequest(BaseModel):
    text: str
    max_new_tokens: Optional[int] = 8
    temperature: Optional[float] = 0.0


@observe(name="summarize-to-keyword", as_type="generation")
def summarize_to_keyword(user_input: str, max_new_tokens: int = 8, temperature: float = 0.0) -> str:
    """Сокращает текст до одного ключевого слова"""
    prompt = f"Ты - умный ассистент. Сократи это предложение до одного ключевого слова: {user_input}\nКлючевое слово:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()

    if not generated_text:
        return ""
    generated_text = generated_text.splitlines()[0].strip()
    keyword = generated_text.split()[0].strip(" .,:;\"'()[]")
    return keyword


@observe(name="fetch-news", as_type="generation")
def fetch_news(keyword: str):
    """Возвращает новости по ключевому слову"""
    params = {
        "q": keyword,
        "language": "ru",
        "pageSize": 3,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code != 200:
        return {"error": f"News API error: {response.status_code}", "keyword": keyword}

    data = response.json()
    if not data.get("articles"):
        return {"keyword": keyword, "news": []}

    articles = [
        {
            "title": a["title"],
            "url": a["url"],
            "source": a["source"]["name"]
        }
        for a in data["articles"]
    ]
    return {"news": articles, "keyword": keyword}


@app.post("/summarize")
def api_summarize(req: InRequest):
    """Эндпоинт: получение ключевого слова + новостей"""
    keyword = summarize_to_keyword(req.text, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    news = fetch_news(keyword)
    return news


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
