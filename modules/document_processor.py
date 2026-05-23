import os
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, directory_path: str) -> List[Document]:
        txt_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        documents = txt_loader.load() + pdf_loader.load()
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def process_directory(self, directory_path: str) -> List[Document]:
        docs = self.load_documents(directory_path)
        chunks = self.split_documents(docs)
        print(f"Processed {len(docs)} documents into {len(chunks)} chunks.")
        return chunks
