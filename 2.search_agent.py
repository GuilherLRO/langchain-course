import os
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# @tool
# def search(query: str) -> str:
#     """
#     Search the web for information about the query.
#     Args:
#         query: The query to search for.
#     Returns:
#         The search results.
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query)
    
llm = ChatOpenAI(model="gpt-5")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({'messages': [HumanMessage(content="Search for 3 job postings for ai engineer on linkedin from us or europe that allow international hires from brazil")]})
    print(result)

if __name__ == "__main__":
    main()
