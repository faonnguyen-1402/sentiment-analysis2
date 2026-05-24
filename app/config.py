APP_TITLE = "Vietnamese Sentiment Analysis"
APP_SUBTITLE = "Phân loại cảm xúc bình luận tiếng Việt thành Positive hoặc Negative"

MODEL_PATH = "model/sentiment_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
ACCURACY_PATH = "model/accuracy.pkl"
CLASSIFICATION_REPORT_PATH = "model/classification_report.pkl"
CONFUSION_MATRIX_PATH = "model/confusion_matrix.pkl"
MODEL_INFO_PATH = "model/model_info.json"
DATASET_PATH = "data/reviews.csv"

PROJECT_INFO = {
    "Tên đề tài": "Phân loại cảm xúc văn bản",
    "Ngôn ngữ": "Python",
    "Web": "Streamlit",
    "Vectorizer": "TF-IDF",
    "Model": "Logistic Regression",
    "Output": "Positive / Negative"
}