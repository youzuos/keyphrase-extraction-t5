import sys
from pathlib import Path
from typing import Dict, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataLoader, DataPreprocessor
from src.models import T5KeyphraseModel
from .metrics import calculate_average_metrics, calculate_f1_precision_recall


class KeyphraseEvaluator:
    
    def __init__(self, model: T5KeyphraseModel):
        self.model = model
    
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        filename: str,
        sample_size: Optional[int] = None
    ) -> Dict:
        print(f"\nEvaluating dataset: {filename}")
        if sample_size:
            print(f"Sample size: {sample_size}")
        
        all_true_keyphrases = []
        all_predicted_keyphrases = []
        all_samples = []
        
        count = 0
        for item in data_loader.load_jsonl(filename):
            if sample_size and count >= sample_size:
                break
            
            processed = DataPreprocessor.prepare_for_training(item)
            
            if not processed['text'] or not processed['keyphrases']:
                continue
            
            predicted = self.model.predict(processed['text'])
            
            all_true_keyphrases.append(processed['keyphrases'])
            all_predicted_keyphrases.append(predicted)
            all_samples.append({
                'id': processed['id'],
                'text': processed['text'][:100] + '...' if len(processed['text']) > 100 else processed['text'],
                'true_keyphrases': processed['keyphrases'],
                'predicted_keyphrases': predicted
            })
            
            count += 1
            if count % 10 == 0:
                print(f"  Evaluated {count} samples...")
        
        print(f"\nTotal evaluated: {len(all_true_keyphrases)} samples")
        
        metrics = calculate_average_metrics(
            all_true_keyphrases,
            all_predicted_keyphrases
        )
        
        metrics['samples'] = all_samples
        
        return metrics
    
    def print_evaluation_report(self, metrics: Dict):
        print("\n" + "=" * 60)
        print("Evaluation Report")
        print("=" * 60)
        print(f"\nSamples: {metrics['num_samples']}")
        print(f"\nAvg Keyphrases:")
        print(f"  True: {metrics['avg_true_keyphrases']:.2f}")
        print(f"  Predicted: {metrics['avg_predicted_keyphrases']:.2f}")
        
        print(f"\nMetrics:")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1:        {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print("=" * 60)
    
    def evaluate_and_report(
        self,
        data_loader: DataLoader,
        filename: str,
        sample_size: Optional[int] = None
    ) -> Dict:
        metrics = self.evaluate_dataset(data_loader, filename, sample_size)
        self.print_evaluation_report(metrics)
        return metrics
    
    def show_sample_predictions(self, metrics: Dict, n: int = 3):
        samples = metrics.get('samples', [])
        if not samples:
            print("No sample data")
            return
        
        sample_scores = []
        for sample in samples:
            true_kps = sample['true_keyphrases']
            pred_kps = sample['predicted_keyphrases']
            _, _, f1 = calculate_f1_precision_recall(true_kps, pred_kps)
            sample_scores.append((f1, sample))
        
        sample_scores.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\nTop {min(n, len(samples))} samples:")
        print("=" * 60)
        
        for i, (f1, sample) in enumerate(sample_scores[:n], 1):
            print(f"\nSample {i} (F1: {f1:.2%}):")
            print(f"  Text: {sample['text']}")
            print(f"  True Keyphrases: {', '.join(sample['true_keyphrases'][:5])}")
            if sample['predicted_keyphrases']:
                print(f"  Predicted Keyphrases: {', '.join(sample['predicted_keyphrases'][:5])}")
            else:
                print(f"  Predicted Keyphrases: (none)")
