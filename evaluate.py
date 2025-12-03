import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import DataLoader
from src.models import T5KeyphraseModel
from src.evaluation import KeyphraseEvaluator
import traceback


def main():
    parser = argparse.ArgumentParser(description="Evaluate keyphrase extraction model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/checkpoints/final_model",
        help="Model path"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test.json",
        help="Test file name"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Sample size"
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=3,
        help="Number of samples to show"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_path}\n")
    model = T5KeyphraseModel()
    model.load_model(args.model_path)
    
    evaluator = KeyphraseEvaluator(model)
    data_loader = DataLoader("data/kp20k")
    
    metrics = evaluator.evaluate_and_report(
        data_loader=data_loader,
        filename=args.test_file,
        sample_size=args.sample_size
    )
    
    if args.show_samples > 0:
        evaluator.show_sample_predictions(metrics, n=args.show_samples)
    
    print("\nEvaluation Complete")
    
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
