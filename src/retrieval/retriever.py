from typing import List, Dict, Any
import numpy as np

from src.models.embedding_model import IndonesianEmbeddingModel
from src.data.vector_store import VectorStore
from src.utils.logger import app_logger

class Retriever:
    def __init__(self, embedding_model: IndonesianEmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        # Pastikan vector store siap digunakan
        self._ensure_retriever_ready()
        
        app_logger.info("Retriever initialized")
    
    def _ensure_retriever_ready(self):
        """Pastikan retriever siap untuk operasi retrieval"""
        try:
            # Force collection initialization
            if not hasattr(self.vector_store, 'collection'):
                self.vector_store._ensure_collection_initialized()
            
            app_logger.info("Retriever is ready for operations")
        except Exception as e:
            app_logger.error(f"Error preparing retriever: {str(e)}")
            raise
    
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        app_logger.info(f"Retrieving documents for query: '{query}'")
        
        try:
            # Pastikan collection tersedia
            if not hasattr(self.vector_store, 'collection'):
                app_logger.warning("Collection not found, reinitializing...")
                self.vector_store._ensure_collection_initialized()
            
            # Encode query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search vector store
            results = self.vector_store.search(query_embedding.tolist(), n_results=n_results)
            
            app_logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            app_logger.error(f"Error during retrieval: {str(e)}")
            return []