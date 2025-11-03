from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
from fastapi.middleware.cors import CORSMiddleware

model = GPT4All("gpt4all-falcon-newbpe-q4_0.gguf", device="cpu")
session = model.chat_session()

app = FastAPI(title="Local LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int | None = 200
    temp: float | None = 0.7


@app.post("/chat")
def chat(req: ChatRequest):
    response = model.generate(
        req.prompt,
        max_tokens=req.max_tokens,
        temp=req.temp,
    )
    return {"response": response}


@app.get("/")
def root():
    return {"message": "API для LLM"}
