import requests
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(find_dotenv())

model = init_chat_model(
    model='qwen/qwen3-32b',
    model_provider='groq',
    temperature=0.1
)

for chunks in model.stream("what is python"):
    print(chunks.text,end='',flush=True)