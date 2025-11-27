from typing import Union
from dotenv import load_dotenv
from langchain_core.tools import render_text_description, tool, Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"Getting text length for: {text}")
    text = text.strip("'\n'").strip('"')
    return len(text)


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

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
    """

    def find_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation:", "Observation:"])

    agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'Hello, world!'?"})
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        print(f"\n[INFO] Agent decided to use tool: '{agent_step.tool}'")
        print(f"[INFO] Tool input: {agent_step.tool_input}")

        tool_to_use = find_tool_by_name(tools, agent_step.tool)
        tool_input = agent_step.tool_input

        print(f"[INFO] Invoking tool '{tool_to_use.name}' with input: '{tool_input}'")
        observation = tool_to_use.func(str(tool_input))

        print(f"[RESULT] Tool '{tool_to_use.name}' returned: {observation}")

