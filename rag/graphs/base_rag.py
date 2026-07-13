from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
@tool
def query_db(input:str):
    """查询数据库信息

    args:
        input: 输入问题
    """
    import time
    # time.sleep(30)
    return "查询结果：你叫小zhi"

def create_base_rag_graph(llm: ChatOpenAI, kb_names: list[str]):
    tool_node = ToolNode(tools=[query_db])

    def llm_node(state: MessagesState):
        llm_with_tools = llm.bind_tools(tools=[query_db])
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("model", llm_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    # checkpointer可以换成数据库保存实现持久化记忆
    return graph.compile(checkpointer=InMemorySaver())
