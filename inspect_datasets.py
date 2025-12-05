import pandas as pd
from pathlib import Path

# Define your dataset paths
DATASETS = {
    "CICIDS2017_cleaned": "data/CICIDS2017_cleaned.csv",
    "KDDTrain+": "data/KDDTrain+.txt",
    "UNSW-NB15": "data/UNSW-NB15.csv",
    "WSN-DS": "data/WSN-DS.csv",
}

def load_dataset(name, path):
    path = Path(path)
    print(f"\n==================== {name} ====================")

    if not path.exists():
        print(f"❌ FILE NOT FOUND: {path}")
        return

    try:
        # handle .txt (KDD) which is usually comma separated
        if path.suffix.lower() == ".txt":
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"❌ FAILED TO READ {name}: {e}")
        return

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nTop 5 rows:")
    print(df.head())

def main():
    for name, path in DATASETS.items():
        load_dataset(name, path)

if __name__ == "__main__":
    main()
