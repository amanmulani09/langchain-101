import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen/qwen3-32b",
    model_provider="groq",
    temperature=0.1
)

r = model.invoke("what is python fella?")

print(r)
print(r.content)