import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ===== CẤU HÌNH CƠ BẢN =====
st.set_page_config(page_title="Food Classifier 🍜", layout="centered")
st.title("🍱 Nhận diện món ăn Việt Nam")
st.write("Ứng dụng nhận diện ảnh **Phở**, **Bánh mì**, **Bún** bằng mô hình TensorFlow")

# ===== TẢI MODEL =====
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("food_model.h5")
    with open("classes.json", "r") as f:
        classes = json.load(f)
    return model, classes

model, classes = load_model()
IMG_SIZE = (224, 224)

# ===== HÀM XỬ LÝ ẢNH =====
def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ===== GIAO DIỆN =====
option = st.radio("Chọn phương thức nhập ảnh:", ["📸 Dùng máy ảnh", "🖼️ Tải ảnh lên"])

uploaded_image = None

if option == "📸 Dùng máy ảnh":
    uploaded_image = st.camera_input("Chụp ảnh món ăn của bạn:")
else:
    uploaded_image = st.file_uploader("Tải lên ảnh món ăn", type=["jpg", "jpeg", "png"])

# ===== XỬ LÝ DỰ ĐOÁN =====
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Ảnh bạn đã chọn", use_column_width=True)

    # Dự đoán
    arr = preprocess_image(image)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = classes[str(idx)] if str(idx) in classes else classes[idx]
    confidence = float(preds[idx])
    CONFIDENCE_THRESHOLD = 0.6  # <-- thêm ngưỡng này

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Không thể nhận dạng món ăn này. Hãy thử lại với ảnh rõ hơn hoặc món khác.")
    else:
        st.subheader(f"🍽️ Dự đoán: **{label.upper()}**")
        st.progress(confidence)
        st.write(f"Độ tin cậy: **{confidence:.2%}**")

    # Hiển thị kết quả
    st.subheader(f"🍽️ Dự đoán: **{label.upper()}**")
    st.progress(confidence)
    st.write(f"Độ tin cậy: **{confidence:.2%}**")

    # Thanh xác suất từng class
    st.write("### 🔎 Xác suất từng loại:")
    for i, (k, v) in enumerate(classes.items()):
        st.write(f"- {v}: {preds[i]:.2%}")