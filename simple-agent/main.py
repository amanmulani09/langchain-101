import requests
from dotenv import load_dotenv,find_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

@tool('get_weather',description="return weather information for given city")
def get_weather(city:str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()
