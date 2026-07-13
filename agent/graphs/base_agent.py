from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from agent.tools import AGETN_TOOLS


def create_base_agent_graph(llm: ChatOpenAI, tool_names: list[str]):
    tool_list = [AGETN_TOOLS[name] for name in tool_names]
    tool_node = ToolNode(tools=tool_list)

    def llm_node(state: MessagesState):
        llm_with_tools = llm.bind_tools(tools=tool_list)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("model", llm_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    # checkpointer可以换成数据库保存实现持久化记忆
    return graph.compile(checkpointer=InMemorySaver())
