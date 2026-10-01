import os
import argparse
import joblib
import pandas as pd
from pandarallel import pandarallel
from text_processor import PersianTextProcessor

DEFAULT_MODEL_PATH = "../models/price_mention_model.pkl"
DEFAULT_INPUT_PATH = "data/test.csv"
DEFAULT_OUTPUT_PATH = "../data/submission.csv"


def run_inference(model_path: str, input_path: str, output_path: str) -> None:
    """Loads serialized pipeline, preprocesses input reviews, and exports predictions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Run train_pipeline.py first."
        )

    print(f"[INFO] Loading serialized pipeline from '{model_path}'...")
    pipeline = joblib.load(model_path)

    print(f"[INFO] Reading test data from '{input_path}'...")
    test_df = pd.read_csv(input_path)
    X_raw = test_df["comment"].fillna("").astype(str)

    print("[INFO] Running parallel Persian text preprocessing...")
    pandarallel.initialize(progress_bar=True)
    processor = PersianTextProcessor()
    X_clean = X_raw.parallel_apply(processor.preprocess_to_string)

    print("\n[INFO] Generating predictions...")
    predictions = pipeline.predict(X_clean)

    submission = pd.DataFrame({"price_value": predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"[INFO] Saved {len(submission)} predictions to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on Persian reviews.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .pkl model.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to input CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output CSV.")
    args = parser.parse_args()

    run_inference(model_path=args.model, input_path=args.input, output_path=args.output)