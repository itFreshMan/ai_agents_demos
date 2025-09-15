from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish,AgentAction
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_openai import AzureChatOpenAI

from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import StateGraph, END


import json
import os
from dotenv import load_dotenv
from tools import *


# Load environment variables from .env file
_=load_dotenv()

##Configure Gemini model
model = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=1
)

# define tools
tools = [get_weather_by_city, search_wikipedia]
tools_map = {
    "get_weather_by_city": get_weather_by_city,
    "search_wikipedia": search_wikipedia
}

# Convert tools to OpenAI function specs (list[dict]) to avoid PydanticSerializationError
functions = [convert_to_openai_function(t) for t in tools]

## Bind functions metadata to model
llm = model.bind(functions=functions)

llm = llm | OpenAIFunctionsAgentOutputParser()

## define prompt_template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

##### 

class AgentState(TypedDict, total=False):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# ---------- 节点 1: Agent ----------
def node_agent(state: AgentState) -> AgentState:
    input = state["input"]
    intermediate_steps = state.get("intermediate_steps", [])
    chat_history = state.get("chat_history", [])

    prompt = prompt_template.format_messages(
        agent_scratchpad=format_to_openai_function_messages(intermediate_steps),
        chat_history=chat_history,
        input=input
    )
    response = llm.invoke(prompt)
    state["agent_outcome"] = response
    # print("LLM response:", type(response))
    return state
    
    

# ---------- 节点 2: tools ----------
def node_tools(state: AgentState) -> AgentState:
    action = state["agent_outcome"] 
    if isinstance(action, AgentAction):
        # print("Invoking tool:", action.tool, "with input:", action.tool_input)
        tool_fn = tools_map[action.tool]
        observation = tool_fn(action.tool_input)
        return {"intermediate_steps": [(action, observation)]}
    return {}    
    
# ---------- 路由 ----------
def should_continue(state: AgentState) -> str:
    action = state.get("agent_outcome", None)
    if not action or isinstance(action, AgentFinish):
        return END
    return "node_tools"

# ---------- 构建图 ----------
graph = StateGraph(AgentState,recursion_limit=5)

graph.add_node("node_agent", node_agent)
graph.add_node("node_tools", node_tools)

graph.set_entry_point("node_agent")
graph.add_conditional_edges("node_agent", should_continue, {"node_tools": "node_tools", END: END})

app = graph.compile()

# ---------- 运行 ----------
result = app.invoke({"input": "今天上海的天气", })
print("input:", result["input"]) # input: 今天上海的天气
lastResult:tuple[AgentAction, str] = result["intermediate_steps"][-1]
print("output:", lastResult[-1]) # output: Weather for 上海市, 中国: The current temperature is 31.1°C