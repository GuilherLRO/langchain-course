from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableLambda

set_debug(True)
# ---------- Pydantic schema ----------


class Source(BaseModel):
    url: str = Field(description="URL of a source used to answer the question")


class AgentResponse(BaseModel):
    answer: str = Field(description="Natural language answer to the user's question")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to answer the question",
    )


# ---------- LLM + Tool ----------

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tavily tool (langchain-tavily)
tavily_tool = TavilySearch(max_results=5)
tools = [tavily_tool]


# ---------- Parser + format instructions ----------

parser = PydanticOutputParser(pydantic_object=AgentResponse)
format_instructions = parser.get_format_instructions()


# ---------- Build the agent ----------

agent_executor = create_agent(
    model=llm,
    tools=tools,
)

# Create a chain: Agent -> Extract Content -> Parse
chain = agent_executor | RunnableLambda(lambda x: x["messages"][-1].content) | parser


# ---------- Run & parse ----------

if __name__ == "__main__":
    print("Hello from langchain-course!")

    question = "search for linkedin roles available for ai engineer in us or europe that allow international hires from brazil. Try multiple ways to get the answer untill you find the answer"

    # Strong system message with explicit schema instructions
    system_message = f"""
You are an assistant that MUST ALWAYS respond in pure JSON.

You will answer the user's question using the available tools.
Your response MUST be a single JSON object that matches exactly
the following schema:

{format_instructions}

Do not include any extra keys. Do not include markdown.
Do not include explanations before or after the JSON.
Just return the JSON.
"""

    # Call the chain
    parsed_result = chain.invoke(
        {
            "messages": [
                ("system", system_message),
                ("human", question),
            ]
        },
        # optional verbose config if you want to see tool calls
        config={"configurable": {"verbose": True}},
    )

    print("\nParsed as AgentResponse:\n", parsed_result)

    print("\nAnswer:\n", parsed_result.answer)
    print("\nSources:")
    for s in parsed_result.sources:
        print("  -", s.url)
