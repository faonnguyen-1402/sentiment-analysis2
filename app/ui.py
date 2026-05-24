import streamlit as st


def load_custom_css():
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

        .small-note {
            color: #94a3b8;
            font-size: 14px;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
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


def render_footer():
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
        Ghi chú: Project này ưu tiên tính đơn giản, dễ triển khai và phù hợp với bài toán phân loại cảm xúc văn bản cơ bản.
        Khi dataset lớn hơn, hệ thống có thể tiếp tục mở rộng sang các mô hình nâng cao hơn.
        </p>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_steps(steps):
    for index, step in enumerate(steps, start=1):
        st.markdown(
            f"<div class='pipeline-step'>{index}. {step}</div>",
            unsafe_allow_html=True
        )