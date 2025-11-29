from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

LANGSMITH_PROJECT="react_implemented"

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"Getting text length for: {text}")
    text = text.strip("'\n'").strip('"')
    return len(text)


if __name__ == "__main__":
    llm = ChatOpenAI()
    tools = [get_text_length]
    
    agent_executor = create_agent(llm, tools)
    
    result = agent_executor.invoke({"messages": [("user", "What is the length of the text 'Hello, world!'?")]})
    
    # Display the conversation flow
    print("\n" + "="*60)
    print("AGENT EXECUTION TRACE")
    print("="*60)
    
    for i, message in enumerate(result['messages'], 1):
        print(f"\nStep {i}: {message.__class__.__name__}")
        print("-" * 60)
        
        if hasattr(message, 'content') and message.content:
            print(f"Content: {message.content}")
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("Tool Calls:")
            for tool_call in message.tool_calls:
                print(f"  - {tool_call['name']}({tool_call['args']})")
        
        if message.__class__.__name__ == 'ToolMessage':
            print(f"Tool Result: {message.content}")
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(result['messages'][-1].content)
