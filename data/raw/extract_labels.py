import json
import pandas as pd
from pathlib import Path

def main():
    raw_dir = Path("data/raw")
    taxi_file = raw_dir / "nyc_taxi.csv"
    windows_file = raw_dir / "combined_windows.json"
    labels_file = raw_dir / "combined_labels.json"
    
    if not taxi_file.exists():
        print(f"Error: {taxi_file} not found")
        return
        
    df = pd.read_csv(taxi_file)
    print(f"Loaded {len(df)} rows from {taxi_file}")
    print("Columns:", list(df.columns))
    print("First 3 rows:\n", df.head(3))
    print("Summary statistics:\n", df.describe())
    
    windows = {}
    labels = []
    if windows_file.exists():
        with open(windows_file, "r") as f:
            all_windows = json.load(f)
            windows = all_windows.get("realKnownCause/nyc_taxi.csv", [])
    if labels_file.exists():
        with open(labels_file, "r") as f:
            all_labels = json.load(f)
            labels = all_labels.get("realKnownCause/nyc_taxi.csv", [])
            
    print(f"Found {len(windows)} anomaly windows and {len(labels)} anomaly point labels")
    
    # Save a clean labels json specifically for nyc_taxi
    taxi_labels = {
        "dataset": "realKnownCause/nyc_taxi.csv",
        "description": "NYC taxi passenger count (half-hourly). Anomalies: Marathon, Thanksgiving, Christmas, New Years, Blizzard",
        "anomaly_windows": windows,
        "anomaly_timestamps": labels
    }
    with open(raw_dir / "nyc_taxi_labels.json", "w") as f:
        json.dump(taxi_labels, f, indent=2)
    print("Saved data/raw/nyc_taxi_labels.json")

if __name__ == "__main__":
    main()
