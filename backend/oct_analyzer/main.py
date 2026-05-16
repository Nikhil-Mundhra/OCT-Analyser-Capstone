from .runtime import configure_runtime

configure_runtime()

from .data_loader import load_oct_volume
from .pre_processing import get_preprocessing_pipeline
from .anatomical_flattener import flatten_volume_to_rpe

def main():
    try:
        # 1. Load the raw data
        # Note: Ensure "patient_001_baseline.vol" is in this directory 
        # or provide the full path.
        print("--- Loading OCT Volume ---")
        raw_volume, original_spacing = load_oct_volume("patient_001_baseline.vol")

        # 2. Setup the MONAI pipeline
        print("--- Running Preprocessing ---")
        pipeline = get_preprocessing_pipeline()
        clean_tensor = pipeline(raw_volume)

        # 3. Anatomical Flattening
        print("--- Flattening Anatomy ---")
        ml_ready_tensor = flatten_volume_to_rpe(clean_tensor)

        print(f"Success! Final Tensor Shape: {ml_ready_tensor.shape}")
        
    except FileNotFoundError:
        print("Error: The .vol file was not found. Check your file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
