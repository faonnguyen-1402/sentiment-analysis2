import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils.text_processing import clean_text


DATA_PATH = "data/reviews.csv"
MODEL_DIR = "model"


def main():
    print("Đang đọc dataset...")

    data = pd.read_csv(DATA_PATH)

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("File reviews.csv cần có 2 cột: text,label")

    data = data.dropna(subset=["text", "label"])

    data["text"] = data["text"].apply(clean_text)
    data["label"] = data["label"].astype(str)

    print("Số lượng dữ liệu:", len(data))
    print("Phân bố nhãn:")
    print(data["label"].value_counts())

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1,
        sublinear_tf=True
    )

    print("\nĐang vector hóa văn bản bằng TF-IDF...")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        C=2.0,
        class_weight="balanced"
    )

    print("Đang huấn luyện Logistic Regression...")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    print("\nĐộ chính xác:", accuracy)
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(matrix)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, "model/sentiment_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(accuracy, "model/accuracy.pkl")
    joblib.dump(report, "model/classification_report.pkl")
    joblib.dump(matrix, "model/confusion_matrix.pkl")

    model_info = {
        "project_name": "Vietnamese Sentiment Analysis",
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "features": "1-gram and 2-gram",
        "max_features": 10000,
        "accuracy": float(accuracy),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "labels": sorted(list(data["label"].unique()))
    }

    with open("model/model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    print("\nĐã huấn luyện và lưu model thành công!")
    print("Các file đã tạo:")
    print("- model/sentiment_model.pkl")
    print("- model/vectorizer.pkl")
    print("- model/accuracy.pkl")
    print("- model/classification_report.pkl")
    print("- model/confusion_matrix.pkl")
    print("- model/model_info.json")


if __name__ == "__main__":
    main()