import requests
from dotenv import load_dotenv, find_dotenv

from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(find_dotenv())

class Context(BaseModel):
    user_id:str

class ResponseFormat(BaseModel):
    summary:str
    temperature_celsius:float
    temperature_fahrenheit:float
    humadity:float

@tool('get_weather',description="returns current weather from users query")
def get_weather(city:str):
    
    r = requests.get(f'https://wttr.in/{city}?format=j1')
    return r.json()

@tool('locate_user',description='look up the users city based on context')
def locate_user(runtime:ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'user_1':
            return 'pune'
        case 'user_2':
            return "hyd"
        case "user_3":
            return "mumbai"
        case _:
            return "Unknown"

model = init_chat_model(
    model='llama-3.3-70b-versatile',
    model_provider="groq",
    temperature=0.3
)

checkpointer = InMemorySaver()

agent = create_agent(
    name="weather agent",
    model=model,
    system_prompt="you're a helpful weather assistant agent, who cracks the joke and is humorous",
    tools=[get_weather,locate_user],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {
    'configurable':{
        'thread_id':1
    }
}

r = agent.invoke(
    {
        'messages':[
        {'role':'user','content':'what is the weather like'}
        ]
    },
    config=config,
    context=Context(user_id='user_1'))

print(r['structured_response'])


r = agent.invoke(
    {
        'messages':[
        {'role':'user','content':'and is this usual?'}
        ]
    },
    config=config,
    context=Context(user_id='user_1'))

print(r['structured_response'].summary)
