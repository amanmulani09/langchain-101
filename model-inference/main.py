from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())

model = init_chat_model(
    model='qwen/qwen3-32b',
    model_provider='groq',
    temperature=0.1
)
class ChatRequest(BaseModel):
    message:str

app = FastAPI()

@app.post('/chat')
async def chat(request:ChatRequest):
    r = model.invoke(request.message)
    return r.json()

@app.post('/chat/stream')
async def stream_chat(request:ChatRequest):

    def generate():
        for chunk in model.stream(request.message):
            if chunk.text:
                yield chunk.text

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )