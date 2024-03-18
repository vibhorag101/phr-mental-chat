from typing import List, Tuple, Dict
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from modelGuards.emotionModel import predict_emotion
from modelGuards.suicideModel import predictSuicide
from modelGuards.threatModel import predictThreat

SYSTEM_PROMPT = """\
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096

class EmotionInput(BaseModel):
    message: str

app = FastAPI()

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "description": "LLama-2-Mental-Therapy-Chatbot-API",
    }

# If using the API, we dont need to expose our own API.
if(os.getenv("USE_LOCAL_MODEL")=="True"):
    from llamaModel.model import get_LLAMA_response
    class GenerateInput(BaseModel):
        messages: List[Dict[str, str]]
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
        temperature: float = 1
        top_p: float = 0.95
        top_k: int = 50

    @app.post("/generate")
    async def generate(data: GenerateInput) -> str:
        if data.max_new_tokens > MAX_MAX_NEW_TOKENS:
            raise HTTPException(status_code=400, detail="max_new_tokens is too large")
        return(get_LLAMA_response(data.messages, data.max_new_tokens, data.temperature, data.top_p, data.top_k))

# @app.post("/emotion")
# async def emotion(data: EmotionInput) -> List[str]:
#     return predict_emotion(data.message)

@app.post("/suicide")
async def emotion(data: EmotionInput) -> str:
    return predictSuicide(data.message)

@app.post("/threat")
async def emotion(data: EmotionInput) -> str:
    return predictThreat(data.message)

@app.post("/safety")
async def checkSafety(data:EmotionInput) -> str:
    if os.getenv("PREDICT_SUICIDE")=="True" and predictSuicide(data.message)=='suicide':
        return("I am sorry that you are feeling this way. You need a specialist help. Please consult a nearby doctor.")
    if os.getenv("PREDICT_THREAT")=="True" and predictThreat(data.message)=='threat':
        return("We detected unlawful language and intentions in the conversation.")
    
    return("safe")