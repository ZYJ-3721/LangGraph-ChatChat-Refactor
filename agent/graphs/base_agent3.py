from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from agent.tools import AGETN_TOOLS


def create_base_agent3_graph(llm: ChatOpenAI, tool_names: list[str]):
    tool_list = [AGETN_TOOLS[name] for name in tool_names]
    return create_deep_agent(
        model=llm,
        tools=tool_list,
        system_prompt="",
        checkpointer=InMemorySaver()
    )
