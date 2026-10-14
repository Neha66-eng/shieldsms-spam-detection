"""Train and evaluate the SMS spam classification pipeline."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_PATH = ROOT / "models" / "sms_spam_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

def load_dataset(path: Path) -> pd.DataFrame:
    """Load UCI's tab-separated dataset (label, message)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download UCI SMS Spam Collection "
            "and put the extracted SMSSpamCollection file in the data/ folder."
        )
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""].drop_duplicates(subset=["label", "message"])
    if df.empty:
        raise ValueError("Dataset loaded, but no valid ham/spam rows were found.")
    return df

def main():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))),
        ("classifier", MultinomialNB(alpha=0.5)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=["ham", "spam"], output_dict=True, zero_division=0
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Dataset rows: {len(df)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision (spam): {metrics['precision_spam']:.4f}")
    print(f"Recall (spam):    {metrics['recall_spam']:.4f}")
    print(f"F1-score (spam):  {metrics['f1_spam']:.4f}")
    print("Confusion matrix (rows=true [ham, spam], columns=predicted [ham, spam]):")
    print(metrics["confusion_matrix"])
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
