from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages


import os
from dotenv import load_dotenv
from tools import *


# Load environment variables from .env file
load_dotenv()

# Configure Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tools = [get_weather_by_city, search_wikipedia]
functions = [convert_to_openai_function(f) for f in tools]

# Bind functions to model
model = model.bind(functions=functions)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
chain = prompt | model | OpenAIFunctionsAgentOutputParser()

def run_agent(user_input):
    intermediate_steps = [] # 记录中间步骤
    while True:
        result = chain.invoke({
            "input": user_input, 
            "agent_scratchpad": format_to_openai_function_messages(intermediate_steps)  # 正确格式化中间步骤
        })
        if isinstance(result, AgentFinish):
            return result.return_values['output']
        tool = {
            "search_wikipedia": search_wikipedia, 
            "get_weather_by_city": get_weather_by_city
        }[result.tool]
        # 如果是function call,将`result和observation`元组追加至中间步骤中
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))



if __name__ == "__main__":
    # Test the chain
    result1 = run_agent({
        "input": "what is the weather is ShangHai?"
    })
    print(result1)