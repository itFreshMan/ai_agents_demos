from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough


from dotenv import load_dotenv
import os
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
])

from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_weather_by_city": get_weather_by_city,
        }
        return tools[result.tool].run(result.tool_input)
    

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route


if __name__ == "__main__":
    # Test the chain
    result1 = chain.invoke({
        "input": "what is the weather is ShangHai?"
    })
    print(result1)