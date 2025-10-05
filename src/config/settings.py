
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qa_model_name: str = "Wikidepia/indobert-lite-squad"
    # generative_model_name: str = "IzzulGod/GPT2-Indo-Instruct-Tuned"
    generative_model_name: str = "cahya/gpt2-small-indonesian-522M"
    max_length: int = 512
    temperature: float = 0.7
    model_type: str = "qa"

@dataclass
class DataConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 32

@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_file: str = "logs/training.log"
    use_tqdm: bool = True

class Settings:
    def __init__(self):
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.logging_config = LoggingConfig()
        
    def get_vector_db_path(self) -> str:
        return "data/processed/vector_db"

settings = Settings()
