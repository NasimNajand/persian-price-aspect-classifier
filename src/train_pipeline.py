import os
import argparse
import joblib
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from src.text_processor import PersianTextProcessor

MODEL_PATH = "../models/price_mention_model.pkl"
DATA_PATH = "../data/train.csv"


def train_or_load_model(force_retrain: bool = False) -> Pipeline:
    """Loads cached model artifact if present; otherwise runs preprocessing and training."""
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"[INFO] Cached model found at '{MODEL_PATH}'. Skipping training.")
        return joblib.load(MODEL_PATH)

    print(f"[INFO] Loading training dataset from '{DATA_PATH}'...")
    train_data = pd.read_csv(DATA_PATH)
    X_raw = train_data["comment"].fillna("").astype(str)
    y = train_data["price_value"]

    print("[INFO] Initializing parallel Persian NLP preprocessor...")
    pandarallel.initialize(progress_bar=True)
    processor = PersianTextProcessor()

    X_clean = X_raw.parallel_apply(processor.preprocess_to_string)

    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y, test_size=0.15, random_state=42, stratify=y
    )

    print("\n[INFO] Fitting TF-IDF + LinearSVC pipeline...")
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=25000,
            min_df=3,
            ngram_range=(1, 2)
        )),
        ("classifier", LinearSVC(
            random_state=42,
            class_weight="balanced",
            dual="auto"
        ))
    ])

    pipeline.fit(X_train, y_train)

    print("\n[INFO] Evaluating model on validation split (15%):")
    y_pred = pipeline.predict(X_val)
    print("\n--- Classification Report ---")
    print(classification_report(y_val, y_pred, digits=4))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[INFO] Pipeline serialized successfully to '{MODEL_PATH}'.")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load the Price Aspect Classifier.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if a serialized model artifact already exists."
    )
    args = parser.parse_args()
    train_or_load_model(force_retrain=args.force)