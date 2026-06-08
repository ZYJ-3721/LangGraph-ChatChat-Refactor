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
    time.sleep(30)
    return "查询结果：你叫小zhi"
@tool
def multiply(a: int, b: int) -> int:
    """计算 a 乘 b
    Args:
        a: 乘数
        b: 乘数
    """
    import time
    time.sleep(20)
    return a * b
@tool
def add(a: int, b: int) -> int:
    """计算 a 加 b"""
    return a + b

def create_base_rag_graph(llm: ChatOpenAI, kb_names: list[str]):
    tool_node = ToolNode(tools=[query_db,multiply,add])

    def call_llm(state: MessagesState):
        llm_with_tools = llm.bind_tools(tools=[query_db,multiply,add])
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", tools_condition)
    graph.add_edge("tools", "call_llm")
    # checkpointer可以换成数据库保存实现持久化记忆
    return graph.compile(checkpointer=InMemorySaver())
