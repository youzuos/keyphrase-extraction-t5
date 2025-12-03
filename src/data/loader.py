import json
from typing import List, Dict, Iterator
from pathlib import Path


class DataLoader:
    
    def __init__(self, data_dir: str = "data/kp20k"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def load_jsonl(self, filename: str) -> Iterator[Dict]:
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}, skipping line")
                        continue
    
    def load_all(self, filename: str) -> List[Dict]:
        return list(self.load_jsonl(filename))
    
    def get_sample(self, filename: str, n: int = 5) -> List[Dict]:
        samples = []
        for i, item in enumerate(self.load_jsonl(filename)):
            if i >= n:
                break
            samples.append(item)
        return samples
    
    def count_lines(self, filename: str) -> int:
        filepath = self.data_dir / filename
        if not filepath.exists():
            return 0
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
