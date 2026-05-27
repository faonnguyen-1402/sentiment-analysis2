import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.text_processing import clean_text


# =========================
# CONFIG
# =========================

DATA_PATH = "data/reviews.csv"
MODEL_DIR = "model"

os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# EXTRA DATASET
# =========================

positive_samples = [
    "Bộ phim này rất hay và cảm động",
    "Nội dung phim hấp dẫn từ đầu đến cuối",
    "Diễn viên diễn rất tự nhiên và xuất sắc",
    "Hình ảnh đẹp âm thanh tốt và cốt truyện ý nghĩa",
    "Tôi rất thích bộ phim này",
    "Phim mang lại cảm xúc tích cực và sâu sắc",
    "Kịch bản tốt nhịp phim hợp lý",
    "Đây là một bộ phim đáng xem",
    "Bộ phim làm tôi cảm thấy hài lòng",
    "Cốt truyện mới lạ và cuốn hút",
    "Phim có nhiều cảnh rất đẹp",
    "Diễn xuất của nhân vật chính rất ấn tượng",
    "Âm nhạc trong phim rất cảm động",
    "Thông điệp phim nhân văn và ý nghĩa",
    "Tôi sẽ giới thiệu phim này cho bạn bè",
    "Bộ phim vượt ngoài mong đợi của tôi",
    "Phim rất đáng tiền",
    "Nội dung phim sâu sắc và dễ hiểu",
    "Các nhân vật được xây dựng rất tốt",
    "Phim đem lại trải nghiệm tuyệt vời",
    "Tôi thấy phim này rất ổn",
    "Phim có nhiều điểm sáng đáng khen",
    "Cách kể chuyện hấp dẫn và lôi cuốn",
    "Bộ phim tạo cảm giác vui vẻ và dễ chịu",
    "Tình tiết phim hợp lý và thú vị",
    "Phim có chiều sâu và cảm xúc",
    "Tôi thật sự ấn tượng với bộ phim",
    "Phim hay hơn tôi mong đợi",
    "Màu phim đẹp và rất nghệ thuật",
    "Bộ phim có kết thúc rất thỏa mãn",
    "Nhân vật chính được thể hiện rất tốt",
    "Tôi cảm thấy rất thích sau khi xem phim",
    "Phim có nội dung tích cực",
    "Tổng thể bộ phim rất chất lượng",
    "Phim vừa giải trí vừa ý nghĩa",
    "Cảnh quay đẹp và âm thanh sống động",
    "Bộ phim có tiết tấu tốt",
    "Tôi không thấy phí thời gian khi xem phim",
    "Phim làm tôi xúc động",
    "Một bộ phim rất đáng để xem lại",
    "Diễn viên phụ cũng diễn rất tốt",
    "Câu chuyện phim gần gũi và cảm động",
    "Phim có nhiều khoảnh khắc đáng nhớ",
    "Tôi đánh giá phim này rất cao",
    "Bộ phim có nội dung rõ ràng",
    "Phim truyền tải thông điệp tốt",
    "Cảm xúc phim rất chân thật",
    "Phim giải trí tốt và không nhàm chán",
    "Tôi rất hài lòng với bộ phim này",
    "Bộ phim có chất lượng tốt",
    "Phim làm tôi cảm thấy vui",
    "Phim có nhiều cảnh hài hước dễ thương",
    "Nội dung phim đơn giản nhưng hiệu quả",
    "Phim phù hợp để xem cùng gia đình",
    "Tôi thích cách phim xây dựng nhân vật",
    "Phim có nhiều điểm sáng về diễn xuất",
    "Bộ phim rất cuốn hút",
    "Phim có nội dung đẹp và nhân văn",
    "Tôi cảm thấy bộ phim rất thành công",
    "Một tác phẩm điện ảnh đáng khen",
    "Phim có ý tưởng sáng tạo",
    "Phim khiến tôi suy nghĩ tích cực hơn",
    "Câu chuyện phim rất cảm hứng",
    "Bộ phim có nhiều cảm xúc đẹp",
    "Phim không quá dài và rất dễ xem",
    "Tôi thấy phim này rất thú vị",
    "Phim có cảnh quay mãn nhãn",
    "Nội dung phim được xử lý rất khéo",
    "Diễn viên nhập vai rất tốt",
    "Phim để lại ấn tượng tốt",
    "Tôi rất thích phần kết của phim",
    "Bộ phim có phong cách riêng",
    "Phim đáng để bỏ thời gian ra xem",
    "Tôi cảm thấy phim rất tuyệt",
    "Phim có chất lượng vượt mong đợi",
    "Nội dung phim rất cuốn",
    "Phim có nhiều chi tiết hay",
    "Tôi thích âm nhạc và hình ảnh của phim",
    "Bộ phim tạo cảm giác dễ chịu",
    "Phim có thông điệp rất tích cực",
    "Diễn xuất rất có hồn",
    "Tôi thấy phim này rất đáng tiền",
    "Phim làm tôi cảm thấy thoải mái",
    "Bộ phim có sự đầu tư tốt",
    "Cách dẫn chuyện rất mượt mà",
    "Phim không bị lê thê",
    "Nội dung phim rất lôi cuốn",
    "Một bộ phim đẹp và cảm động",
    "Phim rất phù hợp với người thích cảm xúc nhẹ nhàng",
    "Tôi rất ưng bộ phim này",
    "Phim có giá trị giải trí cao",
    "Kết phim hợp lý và cảm xúc",
    "Bộ phim có nhiều cảnh đáng nhớ",
    "Tôi thấy đây là phim hay",
    "Phim tạo được cảm xúc tốt cho người xem",
    "Nội dung phim có chiều sâu",
    "Phim có diễn xuất rất tốt",
    "Tôi muốn xem lại bộ phim này",
    "Bộ phim rất thành công về mặt cảm xúc",
    "Phim khiến tôi hài lòng"
]

negative_samples = [
    "Bộ phim này rất dở và nhàm chán",
    "Nội dung phim tệ và khó hiểu",
    "Diễn viên diễn quá tệ",
    "Phim làm tôi cảm thấy phí thời gian",
    "Tôi không thích bộ phim này",
    "Cốt truyện rời rạc và thiếu logic",
    "Phim quá dài và lê thê",
    "Âm thanh kém và hình ảnh xấu",
    "Bộ phim không có điểm gì hấp dẫn",
    "Tôi cảm thấy thất vọng sau khi xem",
    "Diễn xuất gượng gạo và thiếu cảm xúc",
    "Nội dung phim rất nhạt nhẽo",
    "Phim không đáng xem",
    "Kịch bản yếu và thiếu chiều sâu",
    "Tình tiết phim quá dễ đoán",
    "Phim khiến tôi buồn ngủ",
    "Tôi thấy phim này rất chán",
    "Bộ phim không để lại ấn tượng",
    "Nhân vật được xây dựng rất hời hợt",
    "Phim có nhiều cảnh thừa",
    "Nội dung phim thiếu hấp dẫn",
    "Tôi không hiểu phim muốn truyền tải gì",
    "Phim có kết thúc rất hụt hẫng",
    "Bộ phim làm tôi thất vọng",
    "Phim không có cảm xúc",
    "Diễn viên chính diễn thiếu tự nhiên",
    "Phim có nhịp quá chậm",
    "Tôi thấy phim rất phí tiền",
    "Câu chuyện phim quá cũ",
    "Phim không có gì mới lạ",
    "Bộ phim thiếu sự đầu tư",
    "Hình ảnh phim không đẹp",
    "Âm nhạc không phù hợp",
    "Phim không tạo được cảm xúc",
    "Cách kể chuyện rất rối",
    "Tôi không muốn xem lại phim này",
    "Bộ phim có nhiều lỗi logic",
    "Phim gây cảm giác khó chịu",
    "Nội dung phim rất tệ",
    "Tôi không hài lòng với bộ phim",
    "Phim quá nhàm và dài dòng",
    "Diễn xuất thiếu thuyết phục",
    "Bộ phim không đạt kỳ vọng",
    "Phim có nhiều đoạn rất chán",
    "Tôi thấy phim này không đáng tiền",
    "Cốt truyện yếu và thiếu hấp dẫn",
    "Phim không phù hợp với tôi",
    "Bộ phim bị kéo dài không cần thiết",
    "Nội dung thiếu điểm nhấn",
    "Tôi cảm thấy mệt khi xem phim",
    "Phim có nhiều cảnh vô nghĩa",
    "Kết phim quá tệ",
    "Bộ phim không có chiều sâu",
    "Diễn viên phụ diễn rất kém",
    "Tôi thấy phim này rất thất vọng",
    "Phim không cuốn hút",
    "Nội dung phim quá đơn điệu",
    "Phim không có thông điệp rõ ràng",
    "Bộ phim làm tôi chán nản",
    "Tình tiết phim thiếu hợp lý",
    "Phim không có giá trị giải trí",
    "Tôi không thích cách xây dựng nhân vật",
    "Phim có nhiều chi tiết dư thừa",
    "Bộ phim rất khó xem",
    "Phim không gây ấn tượng tốt",
    "Nội dung phim quá nhạt",
    "Tôi thấy phim này rất tệ",
    "Phim không đáng để bỏ thời gian",
    "Câu chuyện phim thiếu cảm xúc",
    "Bộ phim bị làm quá sơ sài",
    "Phim có chất lượng thấp",
    "Tôi cảm thấy bị thất vọng",
    "Phim có tiết tấu rất chậm",
    "Nội dung phim không rõ ràng",
    "Bộ phim quá bình thường và nhàm",
    "Phim không có cảnh nào đáng nhớ",
    "Tôi không đánh giá cao phim này",
    "Diễn xuất rất yếu",
    "Phim không có sự sáng tạo",
    "Bộ phim thiếu điểm nhấn",
    "Phim xem rất mệt",
    "Nội dung phim không thuyết phục",
    "Tôi thấy bộ phim không thành công",
    "Phim có nhiều đoạn gây khó hiểu",
    "Bộ phim không hấp dẫn như quảng cáo",
    "Phim khiến tôi mất hứng",
    "Cốt truyện phim quá kém",
    "Tôi thấy phim này không ổn",
    "Phim không đem lại cảm xúc gì",
    "Bộ phim quá dở",
    "Phim làm tôi cảm thấy chán",
    "Nội dung phim rất thất vọng",
    "Phim có phần mở đầu quá tệ",
    "Bộ phim thiếu logic nghiêm trọng",
    "Tôi không muốn giới thiệu phim này",
    "Phim không phù hợp để xem lại",
    "Cách xử lý tình tiết rất kém",
    "Bộ phim không có sức hút",
    "Phim làm tôi thấy phí thời gian",
    "Tôi rất không hài lòng với phim này"
]


# =========================
# LOAD OLD DATA
# =========================

if os.path.exists(DATA_PATH):
    old_df = pd.read_csv(DATA_PATH)
else:
    old_df = pd.DataFrame(columns=["text", "label"])

if "text" not in old_df.columns or "label" not in old_df.columns:
    old_df = pd.DataFrame(columns=["text", "label"])


# =========================
# CREATE BIGGER DATASET
# =========================

new_data = []

for text in positive_samples:
    new_data.append({
        "text": text,
        "label": "Positive"
    })

for text in negative_samples:
    new_data.append({
        "text": text,
        "label": "Negative"
    })

new_df = pd.DataFrame(new_data)

df = pd.concat([old_df, new_df], ignore_index=True)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

df = df[df["text"] != ""]
df = df[df["label"].isin(["Positive", "Negative"])]
df = df.drop_duplicates(subset=["text", "label"])

df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")


# =========================
# CLEAN TEXT
# =========================

df["cleaned_text"] = df["text"].apply(clean_text)

X = df["cleaned_text"]
y = df["label"]


# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# TRAIN MODEL
# =========================

vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=1,
    sublinear_tf=True
)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=2.0,
    random_state=42
)

model.fit(X_train_vector, y_train)


# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, y_pred)
classification_report_data = classification_report(y_test, y_pred, output_dict=True)
confusion_matrix_data = confusion_matrix(y_test, y_pred)

print("===== DATASET INFO =====")
print("Total rows:", len(df))
print(df["label"].value_counts())

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix_data)


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
joblib.dump(accuracy, os.path.join(MODEL_DIR, "accuracy.pkl"))
joblib.dump(classification_report_data, os.path.join(MODEL_DIR, "classification_report.pkl"))
joblib.dump(confusion_matrix_data, os.path.join(MODEL_DIR, "confusion_matrix.pkl"))

model_info = {
    "model": "Logistic Regression",
    "vectorizer": "TF-IDF",
    "ngram_range": "1-2",
    "max_features": 20000,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "accuracy": round(float(accuracy), 4),
    "labels": sorted(df["label"].unique().tolist())
}

with open(os.path.join(MODEL_DIR, "model_info.json"), "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=4)

print("\n✅ Đã cập nhật data/reviews.csv")
print("✅ Đã lưu model/sentiment_model.pkl")
print("✅ Đã lưu model/vectorizer.pkl")
print("✅ Đã lưu model/accuracy.pkl")
print("✅ Đã lưu model/classification_report.pkl")
print("✅ Đã lưu model/confusion_matrix.pkl")
print("✅ Đã lưu model/model_info.json")
print("\n🎯 Accuracy mới:", round(float(accuracy), 4))