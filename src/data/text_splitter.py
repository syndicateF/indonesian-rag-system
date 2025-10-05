from typing import List, Dict, Any
import re
from src.utils.logger import app_logger
from src.utils.progress_bar import ProgressBar

class IndonesianTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, use_tqdm: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_tqdm = use_tqdm
        app_logger.info(f"TextSplitter initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def split_sentences(self, text: str) -> List[str]:
        # Pattern untuk memisahkan kalimat dalam bahasa Indonesia
        sentence_endings = r'[.!?।]+|\n\n'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep overlap for context preservation
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        app_logger.info(f"Splitting {len(documents)} documents into chunks")
        all_chunks = []
        
        progress_bar = ProgressBar(
            total=len(documents), 
            desc="Splitting documents", 
            use_tqdm=self.use_tqdm
        )
        
        for doc in documents:
            try:
                sentences = self.split_sentences(doc['content'])
                chunks = self.create_chunks(sentences)
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = {
                        'content': chunk,
                        'source': doc['source'],
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'language': 'indonesian',
                        'original_doc_size': len(doc['content'])
                    }
                    all_chunks.append(chunk_doc)
                
                progress_bar.update(1, postfix={
                    "original_docs": len(documents),
                    "chunks_created": len(all_chunks)
                })
                
            except Exception as e:
                app_logger.error(f"Error splitting document {doc.get('source', 'unknown')}: {str(e)}")
                progress_bar.update(1)
        
        progress_bar.close()
        app_logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks