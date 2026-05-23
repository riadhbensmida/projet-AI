from langchain_huggingface import HuggingFaceEmbeddings
import config

class EmbeddingEngine:
    def __init__(self):
        print(f"Initializing Embedding Model: {config.EMBEDDING_MODEL}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embeddings(self):
        return self.embeddings
