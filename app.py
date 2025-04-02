from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import os
import requests
import h5py

app = Flask(__name__)

MODEL_PATH = 'madar_model.h5'
FILE_ID = '1-2Jnar9X4rQXlxR1znNBI4rlnGGxsHD1'

def download_from_google_drive(file_id, dest_path):
    print("📥 Downloading model from Google Drive...")
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded!")

# تحميل النموذج إذا لم يكن موجودًا
if not os.path.exists(MODEL_PATH):
    download_from_google_drive(FILE_ID, MODEL_PATH)

# إصلاح مشكلة batch_shape في h5 إذا ظهرت
with h5py.File(MODEL_PATH, 'r+') as f:
    if 'keras_version' in f.attrs:
        del f.attrs['keras_version']

model = load_model(MODEL_PATH, compile=False)

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
