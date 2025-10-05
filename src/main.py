import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import time

from src.config.settings import settings
from src.data.data_loader import IndonesianTextLoader
from src.data.text_splitter import IndonesianTextSplitter
from src.models.embedding_model import IndonesianEmbeddingModel
from src.data.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import IndonesianGenerator
from src.utils.logger import app_logger, setup_logger

class IndonesianRAGPipeline:
    def __init__(self, model_type: str = "qa", use_tqdm: bool = True):
        self.model_type = model_type
        self.use_tqdm = use_tqdm
        self.settings = settings
        
        # Update model type in settings
        self.settings.model_config.model_type = model_type
        
        # Initialize components
        self.data_loader = IndonesianTextLoader(use_tqdm=use_tqdm)
        self.text_splitter = IndonesianTextSplitter(
            chunk_size=settings.data_config.chunk_size,
            chunk_overlap=settings.data_config.chunk_overlap,
            use_tqdm=use_tqdm
        )
        self.embedding_model = IndonesianEmbeddingModel(
            model_name=settings.model_config.embedding_model_name,
            use_tqdm=use_tqdm
        )
        self.vector_store = VectorStore(use_tqdm=use_tqdm)
        self.retriever = Retriever(self.embedding_model, self.vector_store)
        self.generator = IndonesianGenerator(
            model_config=settings.model_config,
            use_tqdm=use_tqdm
        )
        
        app_logger.info(f"Indonesian RAG Pipeline initialized with {model_type} model")
    
    def build_index(self, data_directory: str) -> bool:
        """Build the search index from documents"""
        app_logger.info(f"Building index from data directory: {data_directory}")
        
        try:
            # PERBAIKAN: Cek apakah data directory ada
            data_path = Path(data_directory)
            if not data_path.exists():
                app_logger.error(f"Data directory tidak ditemukan: {data_directory}")
                return False
            
            # Load documents
            documents = self.data_loader.load_from_directory(data_directory)
            if not documents:
                app_logger.error("No documents loaded. Exiting.")
                return False
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            if not chunks:
                app_logger.error("No chunks created. Exiting.")
                return False
            
            # Generate embeddings
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # PERBAIKAN: Tidak perlu panggil create_collection() lagi
            # karena sudah dihandle di constructor VectorStore
            # Store in vector database
            self.vector_store.add_documents(chunks, embeddings)
            
            app_logger.info("Successfully built search index")
            return True
            
        except Exception as e:
            app_logger.error(f"Error building index: {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if collection exists and has documents"""
        try:
            return self.vector_store.collection_exists() and self.vector_store.get_collection_count() > 0
        except Exception as e:
            app_logger.error(f"Error checking collection: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """Get number of documents in collection"""
        try:
            return self.vector_store.get_collection_count()
        except Exception as e:
            app_logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        app_logger.info(f"Processing query: {question}")
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(question, n_results=n_results)
            
            if not relevant_docs:
                return {
                    "answer": "Maaf, tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Generate response
            response = self.generator.generate(question, relevant_docs)
            return response
            
        except Exception as e:
            app_logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "confidence": 0.0,
                "sources": []
            }

def main():
    parser = argparse.ArgumentParser(description="Indonesian RAG System")
    parser.add_argument("--data-dir", type=str, help="Directory containing Indonesian text documents")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--model-type", choices=["qa", "generative"], default="qa", 
                       help="Model type: 'qa' for question-answering, 'generative' for text generation")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    use_tqdm = not args.no_tqdm
    
    # Setup logging
    setup_logger("indonesian_rag", "logs/application.log", "INFO")
    
    rag_pipeline = IndonesianRAGPipeline(model_type=args.model_type, use_tqdm=use_tqdm)
    
    # Build index jika data directory diberikan
    if args.data_dir:
        print(f"📦 Building index from: {args.data_dir}")
        success = rag_pipeline.build_index(args.data_dir)
        if not success:
            print("❌ Failed to build index")
            sys.exit(1)
        else:
            doc_count = rag_pipeline.get_document_count()
            print(f"✅ Successfully built index with {doc_count} documents")
    
    # Process single query
    if args.query:
        print(f"❓ Processing query: {args.query}")
        response = rag_pipeline.query(args.query)
        print(f"\nPertanyaan: {args.query}")
        print(f"Jawaban: {response['answer']}")
        print(f"Confidence: {response['confidence']:.3f}")
        
        if response['sources']:
            print("\n📚 Sumber referensi:")
            for i, source in enumerate(response['sources'], 1):
                print(f"  {i}. {source['source']}")
                print(f"     Preview: {source['content_preview']}")
    
    # Interactive mode
    if args.interactive or (not args.data_dir and not args.query):
        print("\n" + "="*50)
        print("🤖 Selamat datang di Sistem RAG Bahasa Indonesia!")
        print("="*50)
        
        # Show current status
        if rag_pipeline.collection_exists():
            doc_count = rag_pipeline.get_document_count()
            print(f"✅ Sistem siap dengan {doc_count} dokumen terindeks")
        else:
            print("⚠️  Sistem berjalan tanpa dokumen. Gunakan --data-dir untuk membangun index.")
        
        print("\nKetik 'quit' untuk keluar")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n🎯 Pertanyaan Anda: ").strip()
                if question.lower() in ['quit', 'keluar', 'exit', 'q']:
                    print("Terima kasih telah menggunakan sistem RAG Indonesia!")
                    break
                
                if not question:
                    continue
                
                start_time = time.time()
                response = rag_pipeline.query(question)
                response_time = time.time() - start_time
                
                print(f"\n✅ Jawaban: {response['answer']}")
                print(f"📊 Confidence: {response['confidence']:.3f}")
                print(f"⏱️  Response time: {response_time:.2f} detik")
                
                if response['sources']:
                    print(f"📖 Menggunakan {len(response['sources'])} sumber referensi")
                
            except KeyboardInterrupt:
                print("\n\nTerima kasih telah menggunakan sistem RAG Indonesia!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()