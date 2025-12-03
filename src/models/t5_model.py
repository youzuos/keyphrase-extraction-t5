from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from typing import List, Optional
import torch
from pathlib import Path


class T5KeyphraseModel:
    
    def __init__(self, model_name: str = "t5-small", device: Optional[str] = None):
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Device: {self.device}")
        print(f"Loading model: {model_name}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "models/checkpoints",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("\nTraining...")
        trainer.train()
        
        final_model_path = Path(output_dir) / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        print(f"\nModel saved to: {final_model_path}")
        
        return trainer
    
    def predict(self, text: str, max_length: int = 128, num_beams: int = 4) -> List[str]:
        self.model.eval()
        
        input_encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_encoding['input_ids'],
                attention_mask=input_encoding['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        keyphrases = [kp.strip() for kp in generated_text.split(';') if kp.strip()]
        
        return keyphrases
    
    def predict_batch(self, texts: List[str], max_length: int = 128, num_beams: int = 4) -> List[List[str]]:
        self.model.eval()
        results = []
        
        encodings = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            keyphrases = [kp.strip() for kp in generated_text.split(';') if kp.strip()]
            results.append(keyphrases)
        
        return results
    
    def save_model(self, save_path: str):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, model_path: str):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")
