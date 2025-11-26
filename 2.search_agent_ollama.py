import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama  # Use Ollama for local LLM
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class Source(BaseModel):
    """Scehema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response of the agent with answer and sources"""
    answer: str = Field(description="The agent's answer to the user's question")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent to answer the question")

@tool
def search(query: str) -> str:
    """
    Search the web for information about the query.
    Args:
        query: The query to search for.
    Returns:
        The search results.
    """
    print(f"Searching for {query}")
    return tavily.search(query)

llm = ChatOllama(model="llama3.2:3b", temperature=0)  # Local model via Ollama
tools = [search]
# tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({'messages': [HumanMessage(content="Search for 3 job postings for ai engineer on linkedin from us or europe that allow international hires from brazil")]})
    print(result)

if __name__ == "__main__":
    main()
