from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor



import os
from dotenv import load_dotenv
from tools import *


# Load environment variables from .env file
_=load_dotenv()

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
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_function_messages(x["intermediate_steps"])
) | chain

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)


if __name__ == "__main__":
    # Test the chain
    result1 = agent_executor.invoke({
        "input": "what is the weather is ShangHai?"
    })
    print(result1)