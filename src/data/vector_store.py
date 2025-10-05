import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import os

from src.utils.logger import app_logger
from src.utils.progress_bar import ProgressBar

class VectorStore:
    def __init__(self, persist_directory: str = "data/processed/vector_db", use_tqdm: bool = True):
        self.persist_directory = persist_directory
        self.use_tqdm = use_tqdm
        
        # Buat direktori jika belum ada
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Inisialisasi koleksi - gunakan get_or_create_collection
        self._ensure_collection_initialized()
        
        app_logger.info(f"Vector store initialized at {persist_directory}")
    
    def _ensure_collection_initialized(self):
        """Pastikan koleksi sudah diinisialisasi dengan approach yang benar"""
        try:
            # Gunakan get_or_create_collection untuk menghindari error
            self.collection = self.client.get_or_create_collection(
                name="indonesian_documents",
                metadata={"description": "Indonesian documents for RAG system"}
            )
            app_logger.info("Collection ensured and ready")
        except Exception as e:
            app_logger.error(f"Error ensuring collection: {str(e)}")
            # Fallback: coba create collection dengan approach berbeda
            self._create_collection_fallback()
    
    def _create_collection_fallback(self):
        """Fallback method untuk create collection jika method utama gagal"""
        try:
            # Coba list collections dulu untuk melihat yang ada
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            if "indonesian_documents" in collection_names:
                self.collection = self.client.get_collection("indonesian_documents")
                app_logger.info("Retrieved existing collection")
            else:
                # Create collection dengan parameter minimal
                self.collection = self.client.create_collection(name="indonesian_documents")
                app_logger.info("Created new collection with fallback method")
                
        except Exception as e:
            app_logger.error(f"Fallback collection creation also failed: {str(e)}")
            raise
    
    def collection_exists(self, collection_name: str = "indonesian_documents") -> bool:
        """Check if collection exists in vector database"""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            exists = collection_name in collection_names
            
            if exists and not hasattr(self, 'collection'):
                # Jika koleksi ada tapi belum di-set di instance ini
                self.collection = self.client.get_collection(collection_name)
            
            return exists
        except Exception as e:
            app_logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def get_collection_count(self, collection_name: str = "indonesian_documents") -> int:
        """Get number of documents in collection"""
        try:
            if not hasattr(self, 'collection') or self.collection is None:
                self._ensure_collection_initialized()
            
            return self.collection.count()
        except Exception as e:
            app_logger.error(f"Error getting collection count: {str(e)}")
            return 0

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to vector store - versi yang diperbaiki"""
        try:
            # Pastikan collection tersedia
            if not hasattr(self, 'collection') or self.collection is None:
                self._ensure_collection_initialized()
            
            app_logger.info(f"Adding {len(documents)} documents to vector store")
            
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            documents_content = [doc['content'] for doc in documents]
            metadatas = []
            
            for doc in documents:
                metadata = {
                    'source': doc.get('source', 'unknown'),
                    'chunk_id': doc.get('chunk_id', 0),
                    'language': doc.get('language', 'indonesian'),
                    'timestamp': datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            progress_bar = ProgressBar(
                total=len(documents), 
                desc="Adding to vector store", 
                use_tqdm=self.use_tqdm
            )
            
            # Add in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                
                batch_ids = ids[i:end_idx]
                batch_embeddings = embeddings[i:end_idx].tolist()
                batch_documents = documents_content[i:end_idx]
                batch_metadatas = metadatas[i:end_idx]
                
                try:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    
                    progress_bar.update(len(batch_ids), postfix={
                        "added": end_idx,
                        "total": len(documents)
                    })
                    
                except Exception as e:
                    app_logger.error(f"Error adding batch {i//batch_size}: {str(e)}")
                    progress_bar.update(len(batch_ids))
            
            progress_bar.close()
            app_logger.info(f"Successfully added {len(documents)} documents to vector store")
            
        except Exception as e:
            app_logger.error(f"Error in add_documents: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search in vector store"""
        try:
            # Pastikan collection tersedia
            if not hasattr(self, 'collection') or self.collection is None:
                self._ensure_collection_initialized()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            app_logger.info(f"Retrieved {len(formatted_results)} results from vector store")
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"Error searching vector store: {str(e)}")
            return []