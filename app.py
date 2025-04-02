from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import os
import gdown

app = Flask(__name__)

# تحقق من وجود النموذج، وإن لم يوجد نزّله من Google Drive
MODEL_PATH = 'madar_model.h5'
DRIVE_MODEL_ID = '1-2Jnar9X4rQXlxR1znNBI4rlnGGxsHD1'
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_MODEL_ID}", MODEL_PATH, quiet=False)

# تحميل النموذج والترميز
model = load_model(MODEL_PATH)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

IMG_SIZE = 128

@app.route('/', methods=['GET'])
def home():
    return '✅ Madar API is running on Render!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    image_data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0) / 255.0

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    class_name = label_encoder.classes_[predicted_class]
    confidence = float(prediction[0][predicted_class])

    return jsonify({
        "prediction": class_name,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
