from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

def create_rag_graph(llm):
    graph = StateGraph(MessagesState)
    def chatbot(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    # checkpointer可以换成数据库保存实现持久化记忆
    return graph.compile(checkpointer=MemorySaver())
