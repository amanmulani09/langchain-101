import requests
from dotenv import load_dotenv,find_dotenv

from langchain.agents import create_agent # function to create a agent 
from langchain.tools import tool # to define the agent tools 
from langchain.chat_models import init_chat_model # to define the inference model config

load_dotenv(find_dotenv()) # loading env

# defining tool get_weather 
@tool('get_weather',description="return weather information for given city")
def get_weather(city:str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

# defining model config
model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    max_tokens=1024
)

# creating agent 
agent = create_agent(
    model=model, # API KEY in env & langchain[model] needs to installed
    tools=[get_weather],
    system_prompt='you are a weather assistant, who always cracks jokes and is humourous while remaining helpful.'
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