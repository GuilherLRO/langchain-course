from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    tools = [TavilySearch(), multiply]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent_executor = create_agent(llm, tools)

    res = agent_executor.invoke(
        {
            "messages": [("user", "what is the weather in dubai right now? compare it with San Francisco, output should be in celsius")]
        }
    )

    # Parse the response
    final_message = res['messages'][-1].content
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    
    # Handle different response formats
    if isinstance(final_message, list):
        # Extract text from list of content blocks
        for block in final_message:
            if isinstance(block, dict) and block.get('type') == 'text':
                print(block['text'])
    elif isinstance(final_message, str):
        print(final_message)
    else:
        print(final_message)
