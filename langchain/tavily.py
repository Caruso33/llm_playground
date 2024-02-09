import getpass
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# os.environ["TAVILY_API_KEY"] = getpass.getpass()


tool = TavilySearchResults()

response = tool.invoke({"query": "What happened in the latest burning man floods"})

print(response)

###

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = OllamaFunctions(temperature=0, model="llama2-uncensored")
tavily_tool = TavilySearchResults()
tools = [tavily_tool]
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke(
    {"input": "What happened in the latest burning man floods?"}
)

print(response)
