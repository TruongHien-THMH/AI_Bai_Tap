import os
import json
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

# ===== CẤU HÌNH GIAO DIỆN =====
st.set_page_config(page_title="🍜 Nhận diện món ăn Việt Nam", layout="centered")
st.title("🍱 Ứng dụng nhận diện món ăn Việt Nam")
st.write("Nhận diện **Phở**, **Bánh mì**, **Bún** bằng mô hình TensorFlow 🧠")

# ===== TẢI MODEL VÀ CLASS =====
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "food_model.h5")
    class_path = os.path.join(base_dir, "classes.json")

    # Kiểm tra tồn tại
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy model: {model_path}")
        st.stop()

    if not os.path.exists(class_path):
        st.error(f"❌ Không tìm thấy classes.json: {class_path}")
        st.stop()

    # Load model & class
    model = tf.keras.models.load_model(model_path)
    with open(class_path, "r") as f:
        classes = json.load(f)

    return model, classes

model, classes = load_model()

# ===== HÀM TIỀN XỬ LÝ ẢNH =====
IMG_SIZE = (150, 150)

def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        st.error(f"❌ Không thể đọc ảnh: {e}")
        return None
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===== CHỌN ẢNH =====
option = st.radio("Chọn phương thức nhập ảnh:", ["🖼️ Tải ảnh lên", "📸 Dùng máy ảnh"])

uploaded_image = None
if option == "📸 Dùng máy ảnh":
    uploaded_image = st.camera_input("Chụp ảnh món ăn của bạn:")
else:
    uploaded_image = st.file_uploader("Tải lên ảnh món ăn", type=["jpg", "jpeg", "png"])

# ===== DỰ ĐOÁN =====
if uploaded_image is not None:
    # Tiền xử lý
    img_array = preprocess_image(uploaded_image)
    if img_array is None:
        st.stop()

    # Hiển thị ảnh
    st.image(uploaded_image, caption="Ảnh bạn đã chọn", use_column_width=True)

    # Dự đoán
    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = classes.get(str(idx), f"Lớp {idx}")

    CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tự tin

    # Hiển thị kết quả
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Không thể nhận dạng món ăn này. Hãy thử lại với ảnh rõ hơn.")
    else:
        st.success(f"🍽️ Dự đoán: **{label.upper()}**")
        st.progress(confidence)
        st.write(f"Độ tin cậy: **{confidence:.2%}**")

    # Biểu đồ xác suất từng class
    st.write("### 🔎 Xác suất từng loại:")
    for i, (k, v) in enumerate(classes.items()):
        st.write(f"- {v}: {preds[i]:.2%}")

else:
    st.info("⬆️ Hãy tải ảnh hoặc chụp ảnh để bắt đầu nhận diện.")