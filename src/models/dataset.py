from torch.utils.data import Dataset
from typing import Optional
from transformers import T5Tokenizer
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataLoader, DataPreprocessor


class KeyphraseDataset(Dataset):
    
    def __init__(
        self,
        data_loader: DataLoader,
        filename: str,
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128,
        sample_size: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        print(f"Loading data: {filename}")
        self.data = []
        count = 0
        
        for item in data_loader.load_jsonl(filename):
            if sample_size and count >= sample_size:
                break
            
            processed = DataPreprocessor.prepare_for_training(item)
            
            if processed['text'] and processed['keyphrases']:
                self.data.append(processed)
                count += 1
            
            if count % 1000 == 0:
                print(f"  Loaded {count} samples...")
        
        print(f"Total loaded: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item['text']
        keyphrases = item['keyphrases']
        target_text = "; ".join(keyphrases)
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten(),
            'text': input_text,
            'keyphrases': keyphrases
        }
