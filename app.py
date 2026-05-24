import json
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

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
    layout="wide"
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
    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        color: #38bdf8;
        margin-bottom: 8px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }

    .hero-card {
        padding: 28px;
        border-radius: 24px;
        background: linear-gradient(135deg, #e0f2fe, #f8fafc);
        border: 1px solid #bae6fd;
        margin-bottom: 24px;
        color: #0f172a;
        font-size: 16px;
        line-height: 1.7;
    }

    .info-card {
        padding: 22px;
        border-radius: 20px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #0f172a;
        margin-bottom: 16px;
        line-height: 1.6;
    }

    .positive-result {
        padding: 34px;
        border-radius: 24px;
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #166534;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        margin-top: 20px;
        border: 1px solid #86efac;
    }

    .negative-result {
        padding: 34px;
        border-radius: 24px;
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        text-align: center;
        font-size: 36px;
        font-weight: 900;
        margin-top: 20px;
        border: 1px solid #fca5a5;
    }

    .pipeline-step {
        padding: 15px 18px;
        border-radius: 14px;
        background-color: #eff6ff;
        border-left: 6px solid #38bdf8;
        margin-bottom: 10px;
        font-weight: 700;
        color: #0f172a;
    }

    .keyword-positive {
        padding: 14px;
        border-radius: 14px;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
    }

    .keyword-negative {
        padding: 14px;
        border-radius: 14px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
    }

    .warning-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        font-weight: 600;
        margin-bottom: 18px;
    }

    .success-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #dcfce7;
        border: 1px solid #22c55e;
        color: #166534;
        font-weight: 600;
        margin-bottom: 18px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #0f172a;
        text-align: center;
        font-weight: 700;
    }

    .small-note {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
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


# =========================
# HEADER
# =========================

st.markdown(
    "<div class='main-title'>🎬 Vietnamese Sentiment Analysis</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Phân loại cảm xúc bình luận tiếng Việt thành Positive hoặc Negative</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='hero-card'>
        <b>Project:</b> Phân loại cảm xúc văn bản đơn giản bằng Machine Learning.
        Người dùng có thể nhập một bình luận hoặc tải file CSV để phân tích hàng loạt.
        <br>
        <b>Pipeline:</b> Clean Text → TF-IDF Vectorization → Logistic Regression → Positive / Negative.
        <br>
        <b>Ưu điểm:</b> Mô hình có tốc độ xử lý nhanh, dễ triển khai, phù hợp với bài toán phân loại cảm xúc văn bản cơ bản và hỗ trợ phân tích nhiều bình luận cùng lúc.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("📌 Thông tin project")

    st.write("**Tên đề tài:** Phân loại cảm xúc văn bản")
    st.write("**Ngôn ngữ:** Python")
    st.write("**Web:** Streamlit")
    st.write("**Vectorizer:** TF-IDF")
    st.write("**Model:** Logistic Regression")
    st.write("**Output:** Positive / Negative")

    st.markdown("---")

    st.metric("Accuracy", f"{accuracy:.2f}")

    train_size = model_info.get("train_size", 0)
    test_size = model_info.get("test_size", 0)
    max_features = model_info.get("max_features", "N/A")

    st.write("**Train size:**", train_size)
    st.write("**Test size:**", test_size)
    st.write("**Max features:**", max_features)

    st.markdown("---")

    if accuracy < 0.6:
        st.warning("Accuracy đang thấp. Nên tăng dataset để model dự đoán tốt hơn.")
    else:
        st.success("Model đang có accuracy khá ổn.")

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
# MAIN TABS
# =========================

tab_analyze, tab_batch, tab_dataset, tab_model, tab_about = st.tabs(
    [
        "🔍 Phân tích 1 câu",
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
    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        st.markdown("### ✍️ Nhập bình luận cần phân tích")

        comment = st.text_area(
            "Nội dung bình luận",
            value=st.session_state.comment,
            placeholder="Ví dụ: Phim hơi dài nhưng nội dung rất hay và cảm động...",
            height=180
        )

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            analyze_btn = st.button("🔍 Phân tích cảm xúc", width="stretch")

        with button_col2:
            clear_btn = st.button("🧹 Xóa nội dung", width="stretch")

        if clear_btn:
            st.session_state.comment = ""
            st.rerun()

    with right_col:
        st.markdown("### 🧠 Quy trình xử lý")

        pipeline_steps = get_model_pipeline_description()

        for index, step in enumerate(pipeline_steps, start=1):
            st.markdown(
                f"<div class='pipeline-step'>{index}. {step}</div>",
                unsafe_allow_html=True
            )

    if analyze_btn:
        if comment.strip() == "":
            st.warning("Vui lòng nhập bình luận trước khi phân tích.")
        else:
            result = predict_sentiment(comment)
            st.session_state.history.insert(0, result)

            prediction = result["prediction"]
            confidence = result["confidence"]
            analysis = result["analysis"]

            st.markdown("---")
            st.markdown("## 📊 Kết quả phân tích")

            if prediction == "Positive":
                st.markdown(
                    "<div class='positive-result'>😊 Positive<br>Bình luận mang cảm xúc tích cực</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='negative-result'>😞 Negative<br>Bình luận mang cảm xúc tiêu cực</div>",
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
                    Ứng dụng này sử dụng Machine Learning nhẹ thay vì TensorFlow.

                    Quy trình:
                    - Làm sạch văn bản đầu vào
                    - Dùng TF-IDF để chuyển văn bản thành vector số
                    - Dùng Logistic Regression để phân loại Positive hoặc Negative

                    Hướng này phù hợp với dataset nhỏ và bài toán phân loại cảm xúc cơ bản.
                    """
                )

    if st.session_state.history:
        st.markdown("---")
        st.markdown("## 🕒 Lịch sử phân tích")

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


# =========================
# TAB 2: BATCH CSV ANALYSIS
# =========================

with tab_batch:
    st.markdown("## 📁 Phân tích nhiều bình luận bằng file CSV")

    st.markdown(
        """
        <div class='info-card'>
        Tải lên file CSV có cột <b>text</b>. 
        Hệ thống sẽ phân tích từng dòng và trả về kết quả Positive hoặc Negative.
        </div>
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


# =========================
# TAB 3: DATASET
# =========================

with tab_dataset:
    st.markdown("## 🗂️ Dataset hiện tại")

    if dataset.empty:
        st.error("Không đọc được file data/reviews.csv")
    else:
        total_rows = len(dataset)
        total_positive = (dataset["label"] == "Positive").sum() if "label" in dataset.columns else 0
        total_negative = (dataset["label"] == "Negative").sum() if "label" in dataset.columns else 0

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


# =========================
# TAB 4: MODEL EVALUATION
# =========================

with tab_model:
    st.markdown("## 📊 Đánh giá mô hình")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")

    with col2:
        st.metric("Train size", model_info.get("train_size", "N/A"))

    with col3:
        st.metric("Test size", model_info.get("test_size", "N/A"))

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
            <b>Deep Learning:</b> Không dùng TensorFlow<br>
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


# =========================
# TAB 5: ABOUT PROJECT
# =========================

with tab_about:
    st.markdown("## 📘 Giới thiệu project")

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
        <h3>2. Lý do chọn TF-IDF + Logistic Regression</h3>
        Đề tài có gợi ý Embedding → LSTM → Dense, nhưng vì TensorFlow khá nặng
        và dataset hiện tại còn nhỏ, project chọn hướng Machine Learning nhẹ hơn.
        TF-IDF + Logistic Regression phù hợp với bài toán phân loại văn bản cơ bản,
        dễ cài đặt, dễ chạy và dễ giải thích khi thuyết trình.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 3. Pipeline")

    pipeline_steps = get_model_pipeline_description()

    for index, step in enumerate(pipeline_steps, start=1):
        st.markdown(
            f"<div class='pipeline-step'>{index}. {step}</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class='info-card'>
        <h3>4. Hướng phát triển</h3>
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


# =========================
# FOOTER
# =========================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.info("**NLP:** Clean text, keyword analysis")

with footer_col2:
    st.info("**Feature:** TF-IDF Vectorization")

with footer_col3:
    st.info("**Model:** Logistic Regression")

st.markdown(
    """
    <p class='small-note'>
    Ghi chú: Project này ưu tiên sự nhẹ, dễ chạy và dễ demo. 
    Khi dataset lớn hơn, có thể mở rộng sang các mô hình nâng cao hơn.
    </p>
    """,
    unsafe_allow_html=True
)