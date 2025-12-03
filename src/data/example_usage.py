from loader import DataLoader
from preprocessor import DataPreprocessor


def main():
    print("Initializing data loader...")
    loader = DataLoader("data/kp20k")
    
    print("\nLoading test samples...")
    samples = loader.get_sample("test.json", 5)
    print(f"Loaded {len(samples)} samples")
    
    print("\nPreprocessing data...")
    processed_samples = []
    for item in samples:
        processed = DataPreprocessor.prepare_for_training(item)
        processed_samples.append(processed)
    
    print("\nSample results:")
    for i, item in enumerate(processed_samples[:2], 1):
        print(f"\nSample {i}:")
        print(f"  ID: {item['id']}")
        print(f"  Text length: {len(item['text'])} chars")
        print(f"  Keyphrases: {len(item['keyphrases'])}")
        print(f"  Keyphrases: {', '.join(item['keyphrases'][:5])}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
