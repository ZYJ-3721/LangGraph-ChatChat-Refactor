from .base_rag import create_base_rag_graph


RAG_GRAPHS = {
    "基础RAG": create_base_rag_graph,
}

def create_rag_graph(llm, selected_rag_graph, selected_kb_names):
    return RAG_GRAPHS[selected_rag_graph](llm, selected_kb_names)
