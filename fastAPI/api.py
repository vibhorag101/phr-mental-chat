from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from emotionModel.model import predict_emotion
from llamaModel.model import run

SYSTEM_PROMPT = """\
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

class GenerateInput(BaseModel):
    message: str
    history: List[Tuple[str, str]] = []
    system_prompt: str = SYSTEM_PROMPT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 1
    top_p: float = 0.95
    top_k: int = 50

class EmotionInput(BaseModel):
    message: str

app = FastAPI()

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "description": "LLama-2-Mental-Therapy-Chatbot-API",
    }

@app.post("/generate")
async def generate(data: GenerateInput) -> List[Tuple[str, str]]:
    if data.max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise HTTPException(status_code=400, detail="max_new_tokens is too large")
    generator = run(data.message, data.history, data.system_prompt, data.max_new_tokens, data.temperature, data.top_p, data.top_k)
    last_response = (data.message, '')
    try:
        for response in generator:
            last_response = (data.message, response)
    except StopIteration:
        pass
    return data.history + [last_response]

@app.post("/emotion")
async def emotion(data: EmotionInput) -> str:
    return predict_emotion(data.message)