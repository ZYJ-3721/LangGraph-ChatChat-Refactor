from .base_agent import create_base_agent_graph


AGENT_GRAPHS = {
    "基础Agent": create_base_agent_graph,
}

def create_agent_graph(llm, selected_agent_graph, selected_tool_names):
    return AGENT_GRAPHS[selected_agent_graph](llm, selected_tool_names)
