import os
from langchain_community.vectorstores import FAISS
from typing import List

from langchain_core.documents import Document
import config

class VectorManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vector_db_path = os.path.join(config.VECTORSTORE_DIR, "faiss_index")

    def create_and_save_index(self, chunks: List[Document]):
        print(" Building FAISS index...")
        vector_db = FAISS.from_documents(chunks, self.embedding_model)
        vector_db.save_local(self.vector_db_path)
        print(f" Vector store saved to {self.vector_db_path}")
        return vector_db

    def load_index(self):
        if os.path.exists(self.vector_db_path):
            print(" Loading existing FAISS index...")
            return FAISS.load_local(
                self.vector_db_path, 
                self.embedding_model, 
                allow_dangerous_deserialization=True
            )
        else:
            print(" No existing FAISS index found.")
            return None

    def search(self, query: str, k: int = config.TOP_K):
        vector_db = self.load_index()
        if vector_db:
            return vector_db.similarity_search(query, k=k)
        return []
