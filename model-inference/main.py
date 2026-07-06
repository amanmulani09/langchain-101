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

conversation = [
    SystemMessage('You are a helpful assistant who answers for programming questions'),
    HumanMessage('what is python'),
    AIMessage('pythono is a interpreted programming language.'),
    HumanMessage('when was it released?')
]

r = model.invoke(conversation)

print(r)
print(r.content)