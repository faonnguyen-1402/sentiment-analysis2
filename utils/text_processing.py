import re


POSITIVE_WORDS = [
    "hay", "tốt", "tuyệt", "tuyệt vời", "thích", "hấp dẫn",
    "cuốn hút", "đẹp", "cảm động", "ý nghĩa", "vui", "vui vẻ",
    "ấn tượng", "nhân văn", "đáng xem", "hài lòng", "thú vị",
    "ổn", "xuất sắc", "tích cực", "tự nhiên", "đỉnh", "rất hay",
    "quá hay", "đáng tiền", "mãn nhãn", "sâu sắc", "chất lượng"
]

NEGATIVE_WORDS = [
    "dở", "tệ", "chán", "nhàm chán", "khó hiểu", "kém",
    "xấu", "thất vọng", "không đáng xem", "buồn ngủ",
    "nhạt nhẽo", "thiếu logic", "gượng", "phí thời gian",
    "không hấp dẫn", "khó chịu", "cảnh thừa", "không thích",
    "quá tệ", "rất dở", "dở tệ", "vô lý", "tệ hại", "nhạt",
    "lãng phí", "diễn đơ", "diễn tệ"
]


def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_phrase(text, phrase):
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def analyze_sentiment_words(text):
    cleaned_text = clean_text(text)

    found_positive = []
    found_negative = []

    for word in POSITIVE_WORDS:
        if contains_phrase(cleaned_text, word):
            found_positive.append(word)

    for word in NEGATIVE_WORDS:
        if contains_phrase(cleaned_text, word):
            found_negative.append(word)

    positive_score = len(found_positive)
    negative_score = len(found_negative)

    words = cleaned_text.split()

    return {
        "cleaned_text": cleaned_text,
        "positive_words": found_positive,
        "negative_words": found_negative,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "word_count": len(words),
        "char_count": len(cleaned_text)
    }


def generate_explanation(prediction, confidence, analysis):
    positive_score = analysis["positive_score"]
    negative_score = analysis["negative_score"]

    if prediction == "Positive":
        if positive_score > negative_score:
            return (
                "AI dự đoán Positive vì văn bản có nhiều từ mang sắc thái tích cực. "
                "Ngoài ra, mô hình TF-IDF và Logistic Regression cũng nhận thấy tổng thể câu nghiêng về cảm xúc tích cực."
            )

        if negative_score > positive_score:
            return (
                "Câu có xuất hiện một số từ tiêu cực, nhưng mô hình học máy đánh giá tổng thể văn bản vẫn nghiêng về Positive."
            )

        return (
            "AI dự đoán Positive dựa trên đặc trưng TF-IDF của câu và mô hình Logistic Regression đã được huấn luyện."
        )

    if negative_score > positive_score:
        return (
            "AI dự đoán Negative vì văn bản có nhiều từ mang sắc thái tiêu cực. "
            "Mô hình nhận thấy tổng thể câu nghiêng về cảm xúc tiêu cực."
        )

    if positive_score > negative_score:
        return (
            "Câu có xuất hiện một số từ tích cực, nhưng mô hình học máy đánh giá tổng thể văn bản vẫn nghiêng về Negative."
        )

    return (
        "AI dự đoán Negative dựa trên đặc trưng TF-IDF của câu và mô hình Logistic Regression đã được huấn luyện."
    )


def get_model_pipeline_description():
    return [
        "Nhập bình luận",
        "Làm sạch văn bản",
        "Tách từ cơ bản",
        "Chuyển văn bản thành vector bằng TF-IDF",
        "Dự đoán bằng Logistic Regression",
        "Trả về Positive hoặc Negative"
    ]