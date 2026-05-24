import json
from datetime import datetime

import joblib
import pandas as pd

from app.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    ACCURACY_PATH,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    MODEL_INFO_PATH,
    DATASET_PATH
)

from utils.text_processing import (
    clean_text,
    analyze_sentiment_words,
    generate_explanation
)


def load_model_files():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    accuracy = joblib.load(ACCURACY_PATH)

    try:
        classification_report_data = joblib.load(CLASSIFICATION_REPORT_PATH)
    except Exception:
        classification_report_data = None

    try:
        confusion_matrix_data = joblib.load(CONFUSION_MATRIX_PATH)
    except Exception:
        confusion_matrix_data = None

    try:
        with open(MODEL_INFO_PATH, "r", encoding="utf-8") as file:
            model_info = json.load(file)
    except Exception:
        model_info = {}

    return (
        model,
        vectorizer,
        accuracy,
        classification_report_data,
        confusion_matrix_data,
        model_info
    )


def load_dataset():
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception:
        return pd.DataFrame()


def predict_sentiment(comment, model, vectorizer):
    cleaned_comment = clean_text(comment)
    vector = vectorizer.transform([cleaned_comment])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = max(probabilities) * 100

        class_names = list(model.classes_)
        proba_dict = {
            class_names[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(class_names))
        }
    else:
        confidence = 0
        proba_dict = {}

    analysis = analyze_sentiment_words(comment)
    explanation = generate_explanation(prediction, confidence, analysis)

    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "original_text": comment,
        "cleaned_text": cleaned_comment,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": proba_dict,
        "analysis": analysis,
        "explanation": explanation
    }


def predict_many(texts, model, vectorizer):
    results = []

    for text in texts:
        if str(text).strip() == "":
            continue

        result = predict_sentiment(str(text), model, vectorizer)

        results.append({
            "Text": result["original_text"],
            "Cleaned Text": result["cleaned_text"],
            "Prediction": result["prediction"],
            "Confidence (%)": round(result["confidence"], 2),
            "Positive Words": ", ".join(result["analysis"]["positive_words"]),
            "Negative Words": ", ".join(result["analysis"]["negative_words"]),
            "Word Count": result["analysis"]["word_count"]
        })

    return pd.DataFrame(results)