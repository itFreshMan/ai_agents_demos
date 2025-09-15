# Project Description
This project is designed to test agents using LangChain and Pydantic AI.  
The model used is Google Gemini.

# funcs.py
Two functions are exposed for use as tools:
- `get_weather_by_city_func`: Retrieves the weather for a given city.
- `search_wikipedia_func`: Searches Wikipedia and returns the results.

# LangChain
- `tools.py`: Converts functions from `funcs.py` into `LangChain` tools (`langchain.agents.tools`).
- Various versions of LangChain agents:
  - **v1**: `chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route`
  - **v2**: Uses `run_agent` with `agent_scratchpad`
  - **v3**: Uses `agent_chain`
  - **v4**: Uses `AgentExecutor`
  - **v5**: Uses `AgentExecutor` with memory support
  - **v5**: Use `Langgraph`

# Pydantic AI
- Build agents using `pydantic_ai` with the exposed functions as tools.