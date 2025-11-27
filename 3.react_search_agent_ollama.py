import os
from dotenv import load_dotenv

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

load_dotenv()

tools = [TavilySearch()]
llm = ChatOllama(model="llama3.1:8b", temperature=0)
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    print("Hello from langchain-course!")
    # result = agent_executor.invoke({"input": "Search for 3 job postings for ai engineer on linkedin from us or europe that allow international hires from brazil"})
    result = agent_executor.invoke(
        {
            "input": "What are the latest and most relevant news on the Data Analystics as Artificial Intelligence? I also want links to the articles"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
