import os
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import pandas as pd

from src.utils.logger import app_logger
from src.utils.progress_bar import ProgressBar

class BaseDataLoader(ABC):
    @abstractmethod
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        pass

class IndonesianTextLoader(BaseDataLoader):
    def __init__(self, use_tqdm: bool = True):
        self.use_tqdm = use_tqdm
        app_logger.info("IndonesianTextLoader initialized")
    
    def load_from_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        directory = Path(directory_path)
        documents = []
        
        if not directory.exists():
            app_logger.error(f"Directory not found: {directory_path}")
            return documents
        
        text_files = list(directory.glob("**/*.txt"))
        app_logger.info(f"Found {len(text_files)} text files in {directory_path}")
        
        progress_bar = ProgressBar(
            total=len(text_files), 
            desc="Loading text files", 
            use_tqdm=self.use_tqdm
        )
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'language': 'indonesian',
                        'file_size': len(content)
                    })
                    
                progress_bar.update(1, postfix={"current_file": file_path.name})
                
            except Exception as e:
                app_logger.error(f"Error loading file {file_path}: {str(e)}")
                progress_bar.update(1)
        
        progress_bar.close()
        app_logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        return self.load_from_directory(source)