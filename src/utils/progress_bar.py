from tqdm import tqdm
import time
from typing import Optional, Dict, Any
from src.utils.logger import app_logger

class ProgressBar:
    def __init__(self, total: int, desc: str = "Processing", use_tqdm: bool = True):
        self.total = total
        self.desc = desc
        self.use_tqdm = use_tqdm
        self.current = 0
        
        if self.use_tqdm:
            self.pbar = tqdm(
                total=total,
                desc=desc,
                unit="item",
                bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'
            )
        else:
            app_logger.info(f"Starting: {desc}")
    
    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None):
        self.current += n
        
        if self.use_tqdm:
            self.pbar.update(n)
            if postfix:
                self.pbar.set_postfix(**postfix)
        else:
            progress_percent = (self.current / self.total) * 100
            app_logger.info(f"Progress: {progress_percent:.1f}% - {self.current}/{self.total}")
            if postfix:
                app_logger.info(f"Metrics: {postfix}")
    
    def set_description(self, desc: str):
        if self.use_tqdm:
            self.pbar.set_description(desc)
        else:
            app_logger.info(f"Stage: {desc}")
    
    def close(self):
        if self.use_tqdm:
            self.pbar.close()
        app_logger.info(f"Completed: {self.desc}")

def track_operation(iterable, desc: str = "Processing", use_tqdm: bool = True):
    return ProgressBar(total=len(iterable), desc=desc, use_tqdm=use_tqdm)