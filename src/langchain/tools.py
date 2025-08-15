from pydantic import BaseModel, Field
from langchain.agents import tool

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import *

class CityInput(BaseModel):
    city_name: str = Field(..., description="Name of the city to get coordinates for")

@tool(args_schema=CityInput)
def get_weather_by_city(city_name: str) -> str:
    """Get current weather for a city by name. This function combines city lookup with weather data."""

    return get_weather_by_city_func(city_name)
    
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""

    return search_wikipedia_func(query)

if __name__ == "__main__":
    # result = get_weather_by_city.invoke({"city_name":"shanghai"})
    # print(result)
    result = search_wikipedia_func({"query":"langchain"})
    print(result[:100])