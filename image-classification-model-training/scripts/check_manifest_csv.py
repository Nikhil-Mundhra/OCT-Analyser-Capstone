import os
import pandas as pd

manifest_path = "/Users/nikhilmundhra/Documents/Github/OCT-Analyser-Capstone/image-classification-model-training/data/dataset_manifest.csv"
if os.path.exists(manifest_path):
    df = pd.read_csv(manifest_path)
    print("Found manifest:", manifest_path)
    print("Shape:", df.shape)
    print(df.head(3))
else:
    print("Manifest CSV not found at", manifest_path)
