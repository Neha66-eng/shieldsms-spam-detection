"""Prediction helpers shared by the app and tests."""
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "sms_spam_pipeline.joblib"

def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Trained model not found. Run `python train.py` first.")
    return joblib.load(path)

def predict_message(message: str, model=None) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please enter a non-empty SMS message.")
    model = model or load_model()
    prediction = str(model.predict([message.strip()])[0])
    result = {"label": prediction, "spam_probability": None}
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "spam" in classes:
            result["spam_probability"] = float(model.predict_proba([message.strip()])[0][classes.index("spam")])
    return result


if __name__ == "__main__":
    message = input("Enter an SMS message: ")

    try:
        result = predict_message(message)
        print("Prediction:", result["label"].upper())

        if result["spam_probability"] is not None:
            print(f"Spam probability: {result['spam_probability']:.2%}")

    except (ValueError, FileNotFoundError) as error:
        print("Error:", error)
