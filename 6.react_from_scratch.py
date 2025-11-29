from typing import Union
from dotenv import load_dotenv
from langchain_core.tools import render_text_description, tool, Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

load_dotenv()

from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("*********")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*********")

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"Getting text length for: {text}")
    text = text.strip("'\n'").strip('"')
    return len(text)


def format_log_to_str(intermediate_steps):
    """Format the intermediate steps for ReAct agent scratchpad as a string."""
    thoughts = []
    for action, observation in intermediate_steps:
        thoughts.append(
            f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}"
        )
    return "\n".join(thoughts)


if __name__ == "__main__":
    print("Hello ReAct LangChain!")

    tools = [get_text_length]
    tool_names = ", ".join([t.name for t in tools])

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
    Thought: {agent_scratchpad}
    """

    def find_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Tool with name {tool_name} not found")

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=tool_names,
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation:", "Observation:"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []

    agent = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: x["agent_scratchpad"]} | prompt | llm | ReActSingleInputOutputParser()

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'Hello, world!'?", 
        "agent_scratchpad": format_log_to_str(intermediate_steps)})
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            print(f"\n[INFO] Agent decided to use tool: '{agent_step.tool}'")
            print(f"[INFO] Tool input: {agent_step.tool_input}")

            tool_to_use = find_tool_by_name(tools, agent_step.tool)
            tool_input = agent_step.tool_input

            print(f"[INFO] Invoking tool '{tool_to_use.name}' with input: '{tool_input}'")
            observation = tool_to_use.func(str(tool_input))

            print(f"[RESULT] Tool '{tool_to_use.name}' returned: {observation}")
            intermediate_steps.append((agent_step, str(observation)))   

if isinstance(agent_step, AgentFinish):
    print(f"[INFO] Final answer: {agent_step.return_values['output']}")