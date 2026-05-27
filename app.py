import json
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from docx import Document

from utils.text_processing import (
    clean_text,
    analyze_sentiment_words,
    generate_explanation,
    get_model_pipeline_description
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model_files():
    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    accuracy = joblib.load("model/accuracy.pkl")

    try:
        classification_report_data = joblib.load("model/classification_report.pkl")
    except Exception:
        classification_report_data = None

    try:
        confusion_matrix_data = joblib.load("model/confusion_matrix.pkl")
    except Exception:
        confusion_matrix_data = None

    try:
        with open("model/model_info.json", "r", encoding="utf-8") as f:
            model_info = json.load(f)
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


@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/reviews.csv")
    except Exception:
        return pd.DataFrame()


model, vectorizer, accuracy, classification_report_data, confusion_matrix_data, model_info = load_model_files()
dataset = load_dataset()


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 34%),
            radial-gradient(circle at top right, rgba(168, 85, 247, 0.12), transparent 32%),
            radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.08), transparent 28%),
            linear-gradient(135deg, #020617 0%, #0f172a 42%, #020617 100%);
        color: #e2e8f0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1320px;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(2, 6, 23, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.16);
        padding: 10px;
        border-radius: 18px;
        backdrop-filter: blur(16px);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        padding: 10px 16px;
        color: #94a3b8;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.24), rgba(168, 85, 247, 0.18));
        color: #e0f2fe !important;
        border: 1px solid rgba(56, 189, 248, 0.32);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: #67e8f9 !important;
        font-weight: 900;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.24);
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.22), rgba(59, 130, 246, 0.14));
        color: #e0f2fe;
        font-weight: 800;
        transition: 0.25s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.7);
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.34), rgba(168, 85, 247, 0.22));
        box-shadow: 0 16px 38px rgba(14, 165, 233, 0.14);
        color: white;
    }

    textarea,
    input {
        border-radius: 16px !important;
    }

    .app-hero {
        position: relative;
        padding: 42px;
        border-radius: 34px;
        overflow: hidden;
        margin-bottom: 28px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(2, 6, 23, 0.82)),
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.28), transparent 32%),
            radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.22), transparent 34%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow:
            0 30px 100px rgba(0, 0, 0, 0.38),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(18px);
    }

    .app-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent);
        pointer-events: none;
    }

    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 16px;
        border-radius: 999px;
        margin-bottom: 18px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #67e8f9;
        font-size: 13px;
        font-weight: 900;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .main-title {
        text-align: center;
        font-size: clamp(42px, 6vw, 76px);
        line-height: 1.03;
        font-weight: 950;
        margin-bottom: 18px;
        background: linear-gradient(90deg, #e0f2fe, #38bdf8, #a78bfa, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 48px rgba(56, 189, 248, 0.16);
    }

    .sub-title {
        max-width: 780px;
        margin: 0 auto 26px auto;
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        line-height: 1.8;
    }

    .hero-pipeline {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
    }

    .hero-chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 750;
    }

    .glass-card {
        position: relative;
        padding: 26px;
        border-radius: 26px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow:
            0 24px 70px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(18px);
        margin-bottom: 22px;
    }

    .glass-card h3,
    .glass-card h2 {
        color: #e0f2fe;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 28px;
        font-weight: 950;
        color: #e0f2fe;
        margin-bottom: 18px;
    }

    .section-subtext {
        color: #94a3b8;
        line-height: 1.8;
        margin-bottom: 18px;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .dashboard-card {
        padding: 22px;
        border-radius: 22px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(30, 41, 59, 0.62));
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.24);
    }

    .dashboard-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 9px;
    }

    .dashboard-value {
        color: #67e8f9;
        font-size: 34px;
        font-weight: 950;
        line-height: 1;
    }

    .dashboard-note {
        color: #cbd5e1;
        font-size: 13px;
        margin-top: 9px;
    }

    .info-card {
        padding: 22px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #cbd5e1;
        margin-bottom: 16px;
        line-height: 1.7;
    }

    .warning-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.32);
        color: #fde68a;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .success-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.32);
        color: #bbf7d0;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .result-shell {
        margin-top: 28px;
        padding: 30px;
        border-radius: 30px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.76);
        box-shadow: 0 26px 75px rgba(0, 0, 0, 0.28);
    }

    .positive-result {
        padding: 34px;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(74, 222, 128, 0.12)),
            radial-gradient(circle at top right, rgba(187, 247, 208, 0.18), transparent 30%);
        color: #bbf7d0;
        text-align: center;
        font-size: 38px;
        font-weight: 950;
        border: 1px solid rgba(34, 197, 94, 0.38);
        box-shadow: 0 0 45px rgba(34, 197, 94, 0.12);
    }

    .negative-result {
        padding: 34px;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(239, 68, 68, 0.22), rgba(248, 113, 113, 0.12)),
            radial-gradient(circle at top right, rgba(254, 202, 202, 0.18), transparent 30%);
        color: #fecaca;
        text-align: center;
        font-size: 38px;
        font-weight: 950;
        border: 1px solid rgba(248, 113, 113, 0.38);
        box-shadow: 0 0 45px rgba(248, 113, 113, 0.12);
    }

    .keyword-positive {
        padding: 15px;
        border-radius: 16px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.24);
        color: #bbf7d0;
        font-weight: 800;
    }

    .keyword-negative {
        padding: 15px;
        border-radius: 16px;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(248, 113, 113, 0.24);
        color: #fecaca;
        font-weight: 800;
    }

    .pipeline-step {
        padding: 16px 18px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-left: 5px solid #38bdf8;
        margin-bottom: 12px;
        font-weight: 850;
        color: #e0f2fe;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.16);
    }

    .footer-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 24px;
    }

    .footer-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: #cbd5e1;
        text-align: center;
        font-weight: 800;
    }

    .small-note {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.7;
    }

    @media (max-width: 1100px) {
        .dashboard-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 720px) {
        .dashboard-grid,
        .footer-grid {
            grid-template-columns: 1fr;
        }

        .app-hero {
            padding: 28px;
        }

        .main-title {
            font-size: 40px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# SESSION STATE
# =========================

if "comment" not in st.session_state:
    st.session_state.comment = ""

if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# HELPER FUNCTIONS
# =========================

def read_txt_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8-sig", errors="ignore")

    return content


def read_docx_file(uploaded_file):
    document = Document(uploaded_file)
    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def split_long_text(text):
    text = str(text).strip()
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?。！？])\s+|(?<=[.!?])", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 5:
                chunks.append(sentence)

    if not chunks and text:
        chunks = [text]

    return chunks


def predict_sentiment(comment):
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


def analyze_long_text(text):
    chunks = split_long_text(text)
    results = []

    for index, chunk in enumerate(chunks, start=1):
        result = predict_sentiment(chunk)

        results.append({
            "STT": index,
            "Đoạn / Câu": chunk,
            "Dự đoán": result["prediction"],
            "Độ tin cậy (%)": round(result["confidence"], 2),
            "Từ tích cực": ", ".join(result["analysis"]["positive_words"]),
            "Từ tiêu cực": ", ".join(result["analysis"]["negative_words"]),
            "Số từ": result["analysis"]["word_count"]
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df, "Unknown", 0, 0, 0

    positive_count = int((result_df["Dự đoán"] == "Positive").sum())
    negative_count = int((result_df["Dự đoán"] == "Negative").sum())
    avg_confidence = float(result_df["Độ tin cậy (%)"].mean())

    if positive_count >= negative_count:
        overall_prediction = "Positive"
    else:
        overall_prediction = "Negative"

    return result_df, overall_prediction, positive_count, negative_count, avg_confidence


def predict_many(texts):
    results = []

    for text in texts:
        if str(text).strip() == "":
            continue

        result = predict_sentiment(str(text))

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


def set_example(text):
    st.session_state.comment = text


def render_dashboard_card(label, value, note):
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="dashboard-label">{label}</div>
            <div class="dashboard-value">{value}</div>
            <div class="dashboard-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline():
    pipeline_steps = get_model_pipeline_description()

    for index, step in enumerate(pipeline_steps, start=1):
        st.markdown(
            f"<div class='pipeline-step'>{index}. {step}</div>",
            unsafe_allow_html=True
        )


def render_overall_result(overall_prediction, positive_message, negative_message):
    if overall_prediction == "Positive":
        st.markdown(
            f"<div class='positive-result'>😊 Tổng thể: Positive<br><span style='font-size:18px;'>{positive_message}</span></div>",
            unsafe_allow_html=True
        )
    elif overall_prediction == "Negative":
        st.markdown(
            f"<div class='negative-result'>😞 Tổng thể: Negative<br><span style='font-size:18px;'>{negative_message}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Không đủ dữ liệu để phân tích.")


def render_long_analysis_result(result_df, overall_prediction, positive_count, negative_count, avg_confidence, download_name):
    st.markdown("<div class='result-shell'>", unsafe_allow_html=True)
    st.markdown("## 📊 Kết quả tổng quan")

    render_overall_result(
        overall_prediction,
        "Nội dung nghiêng về cảm xúc tích cực",
        "Nội dung nghiêng về cảm xúc tiêu cực"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tổng câu/đoạn", len(result_df))

    with col2:
        st.metric("Positive", positive_count)

    with col3:
        st.metric("Negative", negative_count)

    with col4:
        st.metric("Độ tin cậy TB", f"{avg_confidence:.2f}%")

    if not result_df.empty:
        chart_df = pd.DataFrame({
            "Label": ["Positive", "Negative"],
            "Count": [positive_count, negative_count]
        })

        st.markdown("### 📈 Biểu đồ phân bố cảm xúc")
        st.bar_chart(chart_df, x="Label", y="Count")

        st.markdown("### 🔎 Chi tiết từng câu/đoạn")
        st.dataframe(result_df, width="stretch", hide_index=True)

        result_csv = result_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 Tải kết quả phân tích",
            data=result_csv,
            file_name=download_name,
            mime="text/csv",
            width="stretch"
        )

    with st.expander("📌 Cách hệ thống xử lý"):
        st.write(
            """
            Hệ thống tách nội dung thành các câu/đoạn nhỏ.
            Mỗi câu/đoạn được đưa qua pipeline:
            
            Clean Text → TF-IDF Vectorization → Logistic Regression → Positive / Negative
            
            Sau đó hệ thống tổng hợp số lượng Positive và Negative để đưa ra kết quả tổng thể.
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# DATA SUMMARY
# =========================

total_rows = len(dataset) if not dataset.empty else 0

if not dataset.empty and "label" in dataset.columns:
    total_positive = int((dataset["label"] == "Positive").sum())
    total_negative = int((dataset["label"] == "Negative").sum())
else:
    total_positive = 0
    total_negative = 0

train_size = model_info.get("train_size", "N/A")
test_size = model_info.get("test_size", "N/A")
max_features = model_info.get("max_features", "N/A")


# =========================
# HERO
# =========================

st.markdown(
    """
    <div class="app-hero">
        <div class="hero-content">
            <div class="hero-badge">AI NLP PROJECT</div>
            <h1 class="main-title">🎬 Vietnamese Sentiment Analysis</h1>
            <p class="sub-title">
                Ứng dụng phân tích cảm xúc bình luận tiếng Việt bằng Machine Learning.
                Hệ thống hỗ trợ nhập một bình luận, phân tích đoạn dài, đọc file TXT/Word,
                phân tích hàng loạt bằng CSV, xem dataset và đánh giá mô hình trực quan.
            </p>
            <div class="hero-pipeline">
                <span class="hero-chip">Clean Text</span>
                <span class="hero-chip">TF-IDF Vectorization</span>
                <span class="hero-chip">Logistic Regression</span>
                <span class="hero-chip">Long Text Analysis</span>
                <span class="hero-chip">TXT / DOCX</span>
                <span class="hero-chip">CSV Batch</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("## 📌 Project Control")

    st.markdown(
        """
        <div class="info-card">
            <b>Tên đề tài:</b><br>
            Phân loại cảm xúc văn bản tiếng Việt
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("**Ngôn ngữ:** Python")
    st.write("**Web:** Streamlit")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Model:** Logistic Regression")
    st.write("**Output:** Positive / Negative")
    st.write("**File hỗ trợ:** TXT, DOCX, CSV")

    st.markdown("---")

    st.metric("Accuracy", f"{accuracy:.2f}")
    st.write("**Train size:**", train_size)
    st.write("**Test size:**", test_size)
    st.write("**Max features:**", max_features)

    st.markdown("---")

    if accuracy < 0.6:
        st.warning("Accuracy đang thấp. Nên tăng dataset để mô hình dự đoán tốt hơn.")
    else:
        st.success("Model đang có accuracy khá ổn cho bản demo.")

    st.markdown("---")

    st.subheader("⚡ Test nhanh")

    examples = [
        "Bộ phim này rất hay và cảm động",
        "Diễn viên diễn quá tệ",
        "Nội dung phim hấp dẫn và ý nghĩa",
        "Phim nhàm chán và phí thời gian",
        "Tôi rất thích bộ phim này",
        "Cốt truyện khó hiểu và không hấp dẫn",
        "Phim hơi dài nhưng diễn viên rất xuất sắc",
        "Âm thanh kém và nội dung nhạt nhẽo"
    ]

    for example in examples:
        st.button(
            example,
            width="stretch",
            on_click=set_example,
            args=(example,)
        )

    st.markdown("---")

    if st.button("🧹 Xóa lịch sử", width="stretch"):
        st.session_state.history = []
        st.success("Đã xóa lịch sử.")


# =========================
# DASHBOARD
# =========================

st.markdown("## 🚀 Project Dashboard")

st.markdown("<div class='dashboard-grid'>", unsafe_allow_html=True)

render_dashboard_card("Accuracy", f"{accuracy:.2f}", "Độ chính xác trên tập kiểm tra")
render_dashboard_card("Dataset", total_rows, "Tổng số bình luận hiện có")
render_dashboard_card("Positive", total_positive, "Số mẫu tích cực")
render_dashboard_card("Negative", total_negative, "Số mẫu tiêu cực")

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MAIN TABS
# =========================

tab_analyze, tab_long_text, tab_file, tab_batch, tab_dataset, tab_model, tab_about = st.tabs(
    [
        "🔍 Phân tích 1 câu",
        "📝 Phân tích đoạn dài",
        "📄 Phân tích TXT / Word",
        "📁 Phân tích CSV",
        "🗂️ Dataset",
        "📊 Đánh giá mô hình",
        "📘 Giới thiệu"
    ]
)


# =========================
# TAB 1: SINGLE ANALYSIS
# =========================

with tab_analyze:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>🔍 Phân tích cảm xúc một bình luận</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-subtext'>Nhập một bình luận tiếng Việt, hệ thống sẽ làm sạch văn bản, vector hóa bằng TF-IDF và dự đoán cảm xúc.</p>",
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        comment = st.text_area(
            "Nội dung bình luận",
            value=st.session_state.comment,
            placeholder="Ví dụ: Phim hơi dài nhưng nội dung rất hay và cảm động...",
            height=190
        )

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            analyze_btn = st.button("🚀 Phân tích cảm xúc", width="stretch")

        with button_col2:
            clear_btn = st.button("🧹 Xóa nội dung", width="stretch")

        if clear_btn:
            st.session_state.comment = ""
            st.rerun()

    with right_col:
        st.markdown("### 🧠 Quy trình xử lý")
        render_pipeline()

    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_btn:
        if comment.strip() == "":
            st.warning("Vui lòng nhập bình luận trước khi phân tích.")
        else:
            result = predict_sentiment(comment)
            st.session_state.history.insert(0, result)

            prediction = result["prediction"]
            confidence = result["confidence"]
            analysis = result["analysis"]

            st.markdown("<div class='result-shell'>", unsafe_allow_html=True)
            st.markdown("## 📊 Kết quả phân tích")

            if prediction == "Positive":
                st.markdown(
                    "<div class='positive-result'>😊 Positive<br><span style='font-size:18px;'>Bình luận mang cảm xúc tích cực</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='negative-result'>😞 Negative<br><span style='font-size:18px;'>Bình luận mang cảm xúc tiêu cực</span></div>",
                    unsafe_allow_html=True
                )

            st.markdown("### 🎯 Độ tin cậy dự đoán")
            st.progress(int(confidence))
            st.write(f"Độ tin cậy: **{confidence:.2f}%**")

            if result["probabilities"]:
                proba_df = pd.DataFrame(
                    {
                        "Label": list(result["probabilities"].keys()),
                        "Probability (%)": list(result["probabilities"].values())
                    }
                )

                st.bar_chart(
                    proba_df,
                    x="Label",
                    y="Probability (%)"
                )

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Từ tích cực", analysis["positive_score"])

            with metric_col2:
                st.metric("Từ tiêu cực", analysis["negative_score"])

            with metric_col3:
                st.metric("Số từ", analysis["word_count"])

            with metric_col4:
                st.metric("Số ký tự", analysis["char_count"])

            st.markdown("### 🔎 Phân tích từ khóa")

            keyword_col1, keyword_col2 = st.columns(2)

            with keyword_col1:
                st.subheader("✅ Từ tích cực")
                if analysis["positive_words"]:
                    st.markdown(
                        "<div class='keyword-positive'>" + ", ".join(analysis["positive_words"]) + "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Không tìm thấy từ tích cực rõ ràng.")

            with keyword_col2:
                st.subheader("❌ Từ tiêu cực")
                if analysis["negative_words"]:
                    st.markdown(
                        "<div class='keyword-negative'>" + ", ".join(analysis["negative_words"]) + "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Không tìm thấy từ tiêu cực rõ ràng.")

            st.markdown("### 🤖 Giải thích của AI")
            st.write(result["explanation"])

            with st.expander("🧹 Văn bản sau khi xử lý"):
                st.code(result["cleaned_text"])

            with st.expander("📚 Giải thích kỹ thuật"):
                st.write(
                    """
                    Quy trình kỹ thuật:
                    - Làm sạch văn bản đầu vào
                    - Biến văn bản thành vector số bằng TF-IDF
                    - Đưa vector vào mô hình Logistic Regression
                    - Trả về nhãn Positive hoặc Negative kèm độ tin cậy
                    """
                )

            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>🕒 Lịch sử phân tích</div>", unsafe_allow_html=True)

        history_data = []

        for item in st.session_state.history:
            history_data.append({
                "Thời gian": item["time"],
                "Bình luận": item["original_text"],
                "Dự đoán": item["prediction"],
                "Độ tin cậy": f"{item['confidence']:.2f}%"
            })

        history_df = pd.DataFrame(history_data)

        st.dataframe(history_df, width="stretch", hide_index=True)

        csv = history_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 Tải lịch sử phân tích CSV",
            data=csv,
            file_name="sentiment_history.csv",
            mime="text/csv",
            width="stretch"
        )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 2: LONG TEXT ANALYSIS
# =========================

with tab_long_text:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📝 Phân tích đoạn nhận xét dài</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class='section-subtext'>
        Dùng cho review dài, nhiều câu hoặc nhiều đoạn. Hệ thống sẽ tự tách văn bản thành từng câu/đoạn nhỏ,
        phân tích từng phần và tổng hợp kết quả cảm xúc chung.
        </p>
        """,
        unsafe_allow_html=True
    )

    sample_long_text = (
        "Bộ phim có phần mở đầu hơi chậm và một số cảnh chưa thật sự cần thiết. "
        "Tuy nhiên càng về sau nội dung càng hấp dẫn, diễn viên chính thể hiện rất tốt. "
        "Âm nhạc cảm động, hình ảnh đẹp và thông điệp phim khá ý nghĩa."
    )

    long_text = st.text_area(
        "Nhập đoạn nhận xét dài",
        placeholder=sample_long_text,
        height=280
    )

    long_col1, long_col2 = st.columns(2)

    with long_col1:
        analyze_long_btn = st.button("🚀 Phân tích đoạn dài", width="stretch")

    with long_col2:
        paste_sample_btn = st.button("✨ Hiện ví dụ mẫu", width="stretch")

    if paste_sample_btn:
        st.info(sample_long_text)

    if analyze_long_btn:
        if long_text.strip() == "":
            st.warning("Vui lòng nhập đoạn văn bản trước khi phân tích.")
        else:
            long_result_df, overall_prediction, positive_count, negative_count, avg_confidence = analyze_long_text(long_text)

            render_long_analysis_result(
                long_result_df,
                overall_prediction,
                positive_count,
                negative_count,
                avg_confidence,
                "long_text_sentiment_result.csv"
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 3: TXT / WORD FILE ANALYSIS
# =========================

with tab_file:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📄 Phân tích cảm xúc từ file TXT / Word</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class='section-subtext'>
        Tải lên file <b>.txt</b> hoặc <b>.docx</b>. 
        Hệ thống sẽ đọc nội dung file, tách thành các câu/đoạn nhỏ,
        phân tích từng phần và đưa ra kết quả cảm xúc tổng thể.
        </p>
        """,
        unsafe_allow_html=True
    )

    uploaded_text_file = st.file_uploader(
        "Tải file cần phân tích",
        type=["txt", "docx"]
    )

    st.markdown(
        """
        <div class='info-card'>
        <b>Định dạng hỗ trợ:</b><br>
        • TXT: file văn bản thuần<br>
        • DOCX: file Microsoft Word đời mới<br><br>
        <b>Lưu ý:</b> File .doc cũ chưa được hỗ trợ. Hãy lưu lại thành .docx trước khi upload.
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded_text_file is not None:
        file_name = uploaded_text_file.name.lower()

        try:
            if file_name.endswith(".txt"):
                file_content = read_txt_file(uploaded_text_file)
            elif file_name.endswith(".docx"):
                file_content = read_docx_file(uploaded_text_file)
            else:
                file_content = ""

            if file_content.strip() == "":
                st.error("File không có nội dung hoặc không đọc được nội dung.")
            else:
                st.success(f"Đã đọc file thành công: {uploaded_text_file.name}")

                file_words = len(file_content.split())
                file_chars = len(file_content)

                file_col1, file_col2, file_col3 = st.columns(3)

                with file_col1:
                    st.metric("Số từ trong file", file_words)

                with file_col2:
                    st.metric("Số ký tự", file_chars)

                with file_col3:
                    st.metric("Loại file", file_name.split(".")[-1].upper())

                with st.expander("👀 Xem nội dung file"):
                    st.text_area(
                        "Nội dung đọc được",
                        value=file_content,
                        height=300
                    )

                if st.button("🚀 Phân tích nội dung file", width="stretch"):
                    file_result_df, overall_prediction, positive_count, negative_count, avg_confidence = analyze_long_text(file_content)

                    render_long_analysis_result(
                        file_result_df,
                        overall_prediction,
                        positive_count,
                        negative_count,
                        avg_confidence,
                        "file_sentiment_result.csv"
                    )

        except Exception as e:
            st.error(f"Không thể xử lý file. Lỗi: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 4: BATCH CSV ANALYSIS
# =========================

with tab_batch:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📁 Phân tích nhiều bình luận bằng CSV</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p class='section-subtext'>
        Tải lên file CSV có cột <b>text</b>. Hệ thống sẽ phân tích từng dòng và trả về kết quả Positive hoặc Negative.
        </p>
        """,
        unsafe_allow_html=True
    )

    sample_df = pd.DataFrame({
        "text": [
            "Bộ phim này rất hay và cảm động",
            "Diễn viên diễn quá tệ",
            "Nội dung phim hấp dẫn",
            "Phim nhàm chán và phí thời gian"
        ]
    })

    sample_csv = sample_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Tải file CSV mẫu",
        data=sample_csv,
        file_name="sample_comments.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Tải file CSV cần phân tích",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.markdown("### 👀 Dữ liệu đã tải lên")
            st.dataframe(uploaded_df.head(20), width="stretch", hide_index=True)

            if "text" not in uploaded_df.columns:
                st.error("File CSV cần có cột tên là text.")
            else:
                if st.button("🚀 Phân tích toàn bộ file", width="stretch"):
                    batch_result_df = predict_many(uploaded_df["text"].tolist())

                    st.markdown("### 📊 Kết quả phân tích hàng loạt")
                    st.dataframe(batch_result_df, width="stretch", hide_index=True)

                    positive_count = (batch_result_df["Prediction"] == "Positive").sum()
                    negative_count = (batch_result_df["Prediction"] == "Negative").sum()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Tổng bình luận", len(batch_result_df))

                    with col2:
                        st.metric("Positive", positive_count)

                    with col3:
                        st.metric("Negative", negative_count)

                    chart_df = pd.DataFrame({
                        "Label": ["Positive", "Negative"],
                        "Count": [positive_count, negative_count]
                    })

                    st.bar_chart(chart_df, x="Label", y="Count")

                    result_csv = batch_result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="📥 Tải kết quả phân tích",
                        data=result_csv,
                        file_name="batch_sentiment_result.csv",
                        mime="text/csv",
                        width="stretch"
                    )

        except Exception as e:
            st.error(f"Không đọc được file CSV. Lỗi: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 5: DATASET
# =========================

with tab_dataset:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>🗂️ Dataset hiện tại</div>", unsafe_allow_html=True)

    if dataset.empty:
        st.error("Không đọc được file data/reviews.csv")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tổng dữ liệu", total_rows)

        with col2:
            st.metric("Positive", total_positive)

        with col3:
            st.metric("Negative", total_negative)

        if total_rows < 100:
            st.markdown(
                """
                <div class='warning-card'>
                Dataset hiện tại còn khá ít. Model có thể dự đoán chưa ổn định.
                Nên tăng lên ít nhất 100 - 300 câu để kết quả tốt hơn.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class='success-card'>
                Dataset đã có số lượng khá ổn cho bản demo cơ bản.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Xem dữ liệu")
        st.dataframe(dataset, width="stretch", hide_index=True)

        if "label" in dataset.columns:
            label_count = dataset["label"].value_counts().reset_index()
            label_count.columns = ["Label", "Count"]

            st.markdown("### Phân bố nhãn")
            st.bar_chart(label_count, x="Label", y="Count")

        with st.expander("📌 Gợi ý cải thiện dataset"):
            st.write(
                """
                Để model tốt hơn, bạn nên:
                - Tăng số lượng câu Positive
                - Tăng số lượng câu Negative
                - Cân bằng số lượng giữa 2 nhãn
                - Thêm câu thực tế hơn, dài hơn
                - Thêm câu có sắc thái pha trộn, ví dụ: "phim hơi dài nhưng nội dung hay"
                """
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 6: MODEL EVALUATION
# =========================

with tab_model:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📊 Đánh giá mô hình</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")

    with col2:
        st.metric("Train size", train_size)

    with col3:
        st.metric("Test size", test_size)

    if accuracy < 0.6:
        st.markdown(
            """
            <div class='warning-card'>
            Accuracy hiện tại đang thấp. Nguyên nhân chính thường là dataset quá ít hoặc chưa cân bằng.
            Cần tăng dữ liệu để model học tốt hơn.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Cấu hình mô hình")

    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.markdown(
            """
            <div class='info-card'>
            <b>Vectorizer:</b> TF-IDF<br>
            <b>N-gram:</b> 1-gram và 2-gram<br>
            <b>Max features:</b> 10000<br>
            </div>
            """,
            unsafe_allow_html=True
        )

    with config_col2:
        st.markdown(
            """
            <div class='info-card'>
            <b>Classifier:</b> Logistic Regression<br>
            <b>Library:</b> Scikit-learn<br>
            <b>Output:</b> Positive / Negative<br>
            </div>
            """,
            unsafe_allow_html=True
        )

    if classification_report_data:
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(classification_report_data).transpose()
        st.dataframe(report_df, width="stretch")

    if confusion_matrix_data is not None:
        st.markdown("### Confusion Matrix")
        matrix_df = pd.DataFrame(
            confusion_matrix_data,
            columns=["Predicted Negative", "Predicted Positive"],
            index=["Actual Negative", "Actual Positive"]
        )
        st.dataframe(matrix_df, width="stretch")

    st.markdown("### Nhận xét")

    if accuracy < 0.6:
        st.write(
            """
            Model hiện tại đã chạy được nhưng chất lượng chưa cao.
            Nguyên nhân lớn nhất là dataset còn ít nên model chưa học đủ nhiều trường hợp.
            Tuy nhiên, về mặt kỹ thuật project đã có đủ pipeline NLP cơ bản:
            clean text, TF-IDF, train model, đánh giá và demo web.
            """
        )
    else:
        st.write(
            """
            Model có accuracy tương đối ổn cho bản demo.
            Có thể tiếp tục cải thiện bằng cách tăng dataset hoặc thử thêm thuật toán khác như Naive Bayes, LinearSVC.
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# TAB 7: ABOUT PROJECT
# =========================

with tab_about:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-heading'>📘 Giới thiệu project</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-card'>
        <h3>1. Đề tài</h3>
        Xây dựng ứng dụng phân loại cảm xúc văn bản tiếng Việt.
        Input là một bình luận phim, output là cảm xúc Positive hoặc Negative.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='info-card'>
        <h3>2. Mô hình sử dụng</h3>
        Project sử dụng NLP cơ bản kết hợp TF-IDF Vectorization và Logistic Regression.
        Đây là hướng tiếp cận phù hợp với bài toán phân loại văn bản cơ bản,
        dễ triển khai, tốc độ xử lý nhanh và phù hợp với dataset tiếng Việt đơn giản.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 3. Pipeline")
    render_pipeline()

    st.markdown(
        """
        <div class='info-card'>
        <h3>4. Chức năng chính</h3>
        <ul>
            <li>Phân tích một bình luận ngắn</li>
            <li>Phân tích đoạn nhận xét dài bằng cách tách câu/đoạn</li>
            <li>Phân tích nội dung file TXT hoặc DOCX</li>
            <li>Phân tích hàng loạt bằng CSV</li>
            <li>Xem dataset và đánh giá mô hình</li>
            <li>Tải kết quả phân tích về file CSV</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='info-card'>
        <h3>5. Hướng phát triển</h3>
        <ul>
            <li>Tăng dataset lên 100 - 300 câu hoặc nhiều hơn</li>
            <li>Thêm nhãn Neutral</li>
            <li>Thử thuật toán Naive Bayes hoặc LinearSVC</li>
            <li>Phân tích comment Facebook/Youtube</li>
            <li>Deploy web lên Streamlit Cloud</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <div class="footer-grid">
        <div class="footer-card">🧹 NLP: Clean text & keyword analysis</div>
        <div class="footer-card">📐 Feature: TF-IDF Vectorization</div>
        <div class="footer-card">🤖 Model: Logistic Regression</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class='small-note' style='text-align:center; margin-top:20px;'>
    Project phân loại cảm xúc văn bản tiếng Việt với giao diện Streamlit dashboard hiện đại.
    Hệ thống hỗ trợ phân tích bình luận ngắn, đoạn dài, file TXT/DOCX và CSV.
    </p>
    """,
    unsafe_allow_html=True
)