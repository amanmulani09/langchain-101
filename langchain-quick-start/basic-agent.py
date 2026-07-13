import requests
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


@tool('get_weather',description="get weather for a city")
def get_weather(city:str):
    """ get weather for a given city"""
    r = requests.get(f'https://wttr.in/{city}?format=j1')
    return r.json()


model = init_chat_model(
    model="llama-3.3-70b-versatile",
    model_provider="groq"
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="you are a helpful weather assistant.",
)

# invoke agent 
r = agent.invoke({
    'messages': [
        {
            'role':'user',
            'content':'what is the weather in pune'
        }
    ]
})

print(r)
print(r['messages'][-1].content)