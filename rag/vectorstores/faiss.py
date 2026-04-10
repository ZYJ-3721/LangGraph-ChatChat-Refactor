import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

class Faiss(FAISS):
    def __init__(
            self,
            embedding_function,
            index=None,
            docstore=None, 
            index_to_docstore_id=None,
            index_name: str="index",
            faiss_directory: str=None,
            allow_dangerous_deserialization: bool=False, **kwargs):
        # 初始化父类参数
        super().__init__(embedding_function, index, docstore, index_to_docstore_id, **kwargs)
        # 初始化子类参数
        self.index_name = index_name
        self.faiss_directory = faiss_directory
        self.allow_dangerous_deserialization = allow_dangerous_deserialization

        # 检查本地FAISS数据库文件是否存在
        if os.path.exists(os.path.join(self.faiss_directory, f'{self.index_name}.faiss')):
            local_faiss = FAISS.load_local(
                index_name=self.index_name,
                folder_path=self.faiss_directory, 
                embeddings=self.embedding_function, 
                allow_dangerous_deserialization=self.allow_dangerous_deserialization)
            self.__dict__.update(local_faiss.__dict__)
        else:
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(self.embedding_function.embed_query("你好")))
            if self.docstore is None:
                self.docstore = InMemoryDocstore()
            if self.index_to_docstore_id is None:
                self.index_to_docstore_id = {}
            self.save_local(self.faiss_directory, self.index_name)
    
    def add_documents(self, documents, **kwargs):
        ids = super().add_documents(documents, **kwargs)
        self.save_local(self.faiss_directory, self.index_name)
        return ids
    
