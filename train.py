# T5

import sys
from pathlib import Path
from src.data import DataLoader, DataPreprocessor
from src.models import T5KeyphraseModel, KeyphraseDataset
from src.models.config import ModelConfig, TrainingConfig
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():

    print("T5 Model Training")
    model_config = ModelConfig()
    train_config = TrainingConfig()
    print(f"Model: {model_config.model_name}")


    data_loader = DataLoader("data/kp20k")
    model = T5KeyphraseModel(model_name=model_config.model_name)

    train_dataset = KeyphraseDataset(
        data_loader=data_loader,
        filename=train_config.train_file.split('/')[-1],
        tokenizer=model.tokenizer,
        max_input_length=model_config.max_input_length,
        max_output_length=model_config.max_output_length,
        sample_size=train_config.train_sample_size
    )
    
    eval_dataset = None
    if train_config.val_file:
        eval_dataset = KeyphraseDataset(
            data_loader=data_loader,
            filename=train_config.val_file.split('/')[-1],
            tokenizer=model.tokenizer,
            max_input_length=model_config.max_input_length,
            max_output_length=model_config.max_output_length,
            sample_size=train_config.val_sample_size
        )
    

    trainer = model.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        logging_steps=train_config.logging_steps
    )
    
    print(f"\nModel saved to: {train_config.output_dir}/final_model")
    print(f"Run prediction:python predict.py --model_path {train_config.output_dir}/final_model")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
