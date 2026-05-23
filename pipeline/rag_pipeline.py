import os
import config
from modules.document_processor import DocumentProcessor
from modules.embedding_engine import EmbeddingEngine
from modules.vector_manager import VectorManager
from modules.llm_engine import LLMEngine

class RAGPipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.vector_manager = VectorManager(self.embedding_engine.get_embeddings())
        self.llm_engine = LLMEngine()

    def initialize_system(self, force_rebuild=False):
        if force_rebuild or not os.path.exists(os.path.join(config.VECTORSTORE_DIR, "faiss_index")):
            print("Initializing system (First time setup)...")
            chunks = self.doc_processor.process_directory(config.CORPUS_DIR)
            self.vector_manager.create_and_save_index(chunks)
        else:
            print("System already initialized.")

    def run(self, query: str):
        relevant_docs = self.vector_manager.search(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if not context:
            return "No relevant information found in the documents.", []
            
        response = self.llm_engine.get_response(query, context)
        
        return response, relevant_docs

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.initialize_system()
    
    test_query = "What skills do I need for a Spring Boot internship?"
    answer, docs = pipeline.run(test_query)
    
    print("\n--- TEST RESULT ---")
    print(f"Query: {test_query}")
    print(f"Answer: {answer}")
