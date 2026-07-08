from .base_agent import create_base_agent_graph
from .base_agent2 import create_base_agent2_graph
from .base_agent3 import create_base_agent3_graph


AGENT_GRAPHS = {
    "基础Agent(langgraph)": create_base_agent_graph,
    "基础Agent(langchain)": create_base_agent2_graph,
    "基础Agent(deepagents)": create_base_agent3_graph,
}

def create_agent_graph(llm, selected_agent_graph, selected_tool_names):
    return AGENT_GRAPHS[selected_agent_graph](llm, selected_tool_names)
