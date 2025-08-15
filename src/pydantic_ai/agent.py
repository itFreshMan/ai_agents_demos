from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from funcs import *

# Load environment variables from .env file
load_dotenv()

model = GeminiModel(
    model_name="gemini-2.0-flash",
)

agent = Agent(model,
              tools=[get_weather_by_city_func,search_wikipedia_func],
              system_prompt="You are helpful but sassy assistant")

def main():
    history = []
    while True:
        user_input = input("Input:")
        if(user_input == "EXIT"):
            print("BYE BYE ...")
            break
        resp = agent.run_sync(user_input,message_history=history)
        history = list(resp.all_messages())
        print(resp.output)

if __name__ == "__main__":
    main()