"""
Model configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "t5-small"
    max_input_length: int = 512
    max_output_length: int = 128


@dataclass
class TrainingConfig:
    train_file: str = "data/kp20k/train.json"
    val_file: str = "data/kp20k/validation.json"
    train_sample_size: Optional[int] = None
    val_sample_size: Optional[int] = 1000
    
    output_dir: str = "models/checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    
    max_length: int = 128
    num_beams: int = 4


default_model_config = ModelConfig()
default_training_config = TrainingConfig()
