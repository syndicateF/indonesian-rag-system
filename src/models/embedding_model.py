import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import torch

from src.utils.logger import app_logger
from src.utils.progress_bar import ProgressBar

class IndonesianEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 device: str = None, use_tqdm: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tqdm = use_tqdm
        
        app_logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            app_logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            app_logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        app_logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        progress_bar = ProgressBar(
            total=len(texts), 
            desc="Generating embeddings", 
            use_tqdm=self.use_tqdm
        )
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                    all_embeddings.extend(batch_embeddings.cpu().numpy())
                
                progress_bar.update(len(batch), postfix={
                    "completed": min(i + batch_size, len(texts)),
                    "embedding_dim": batch_embeddings.shape[1]
                })
                
            except Exception as e:
                app_logger.error(f"Error encoding batch {i//batch_size}: {str(e)}")
                progress_bar.update(len(batch))
        
        progress_bar.close()
        app_logger.info(f"Successfully generated embeddings for {len(all_embeddings)} texts")
        return np.array(all_embeddings)