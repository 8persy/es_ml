from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")

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


@app.post("/chat")
def chat(req: ChatRequest):
    with model.chat_session() as session:
        response = session.generate(
            req.prompt,
            max_tokens=400,
            temp=0.4,
        )
    return {"response": response.strip()}


@app.get("/")
def root():
    return {"message": "API для LLM"}


if __name__ == "__main__":
    uvicorn.run(app, port=8080)