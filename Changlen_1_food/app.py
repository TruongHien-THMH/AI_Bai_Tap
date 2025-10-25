# app.py
import io, json
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load model và classes
model = tf.keras.models.load_model("food_model.h5")
with open("classes.json", "r") as f:
    classes = json.load(f)  # mapping index->class_name

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files['file']
    img_bytes = file.read()
    x = prepare_image(img_bytes)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    label = classes[str(idx)] if str(idx) in classes else classes[idx]
    confidence = float(preds[idx])
    
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({"label": "Không thể nhận dạng món ăn này", "confidence": confidence})
    
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001,  debug=True)