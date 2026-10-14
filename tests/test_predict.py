from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from predict import predict_message

@pytest.fixture
def tiny_model():
    messages = [
        "claim your free prize now",
        "win cash call now",
        "are we meeting today",
        "please send me the notes",
        "free entry winner prize",
        "I will call you later",
    ]
    labels = ["spam", "spam", "ham", "ham", "spam", "ham"]
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    model.fit(messages, labels)
    return model

def test_prediction_returns_expected_keys(tiny_model):
    result = predict_message("free prize winner", tiny_model)
    assert result["label"] in {"ham", "spam"}
    assert "spam_probability" in result

def test_empty_message_is_rejected(tiny_model):
    with pytest.raises(ValueError):
        predict_message("   ", tiny_model)
