import os
import importlib

LOADERS_FILE_TYPES = {
    "TextLoader": [".txt"],
    "PDFLoader": [".pdf"],
    "DOCLoader": [".docx"],
    "PPTLoader": [".pptx"],
    "UnstructuredMarkdownLoader": [".md"],
    "UnstructuredFileLoader": [".doc", ".ppt"],
}
FILE_TYPES_LOADERS = {type: loader for loader, types in LOADERS_FILE_TYPES.items() for type in types}

def load_document(file_path, **loader_kwargs):
    file_type = os.path.splitext(file_path)[-1]
    loader_name = FILE_TYPES_LOADERS.get(file_type)
    loaders_module = importlib.import_module("rag.loaders")
    if hasattr(loaders_module, loader_name):
        loader_kwargs.setdefault("method", "rapidocr")
        DocumentLoader = getattr(loaders_module, loader_name)
    else:
        loaders_module = importlib.import_module(
            "langchain_community.document_loaders")
        DocumentLoader = getattr(loaders_module, loader_name)
    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader.load()


SPLITTERS = {
    "递归字符分割": "RecursiveCharacterTextSplitter",
}

def split_text(documents, splitter_type, **splitter_kwargs):
    splitter_name = SPLITTERS.get(splitter_type)
    if splitter_name == "RecursiveCharacterTextSplitter":
        splitter_kwargs.setdefault("is_separator_regex", True)
        splitter_kwargs.setdefault("keep_separator", "end")
        splitter_kwargs.setdefault("separators", [
            r'[。！？]+\s*', r'[；，]+\s*', r'[.!?]+\s*', r'[;,]+\s*', r'\s+'])
    splitters_module = importlib.import_module("rag.splitters")
    if hasattr(splitters_module, splitter_name):
        TextSplitter = getattr(splitters_module, splitter_name)
    else:
        splitters_module = importlib.import_module(
            "langchain_text_splitters") # langchain 分割器
        TextSplitter = getattr(splitters_module, splitter_name)
    splitter = TextSplitter(**splitter_kwargs)
    return splitter.split_documents(documents)


VECTORSTORES = {
    "Milvus": "Milvus",
    "Chroma": "Chroma",
    "Faiss": "Faiss",
}

def get_vectorstore(kb_name, kb_type, embedding, vectorstores_path, **vectorstore_kwargs):
    if kb_type == "Milvus":
        from langchain_milvus import Milvus
        vectorstore_kwargs.setdefault("connection_args", {"uri": os.path.join(vectorstores_path, f'{kb_name}.db')})
        vectorstore_kwargs.setdefault("index_params", {"index_type": "AUTOINDEX", "metric_type": "L2"})
        return Milvus(embedding_function=embedding, auto_id=True, **vectorstore_kwargs)
    elif kb_type == "Chroma":
        from langchain_chroma import Chroma
        vectorstore_kwargs.setdefault("persist_directory", vectorstores_path)
        return Chroma(embedding_function=embedding, **vectorstore_kwargs)
    elif kb_type == "Faiss":
        from rag.vectorstores import Faiss
        vectorstore_kwargs.setdefault("allow_dangerous_deserialization", True)
        vectorstore_kwargs.setdefault("faiss_directory", vectorstores_path)
        return Faiss(embedding_function=embedding, **vectorstore_kwargs)
