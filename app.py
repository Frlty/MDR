from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import os
import requests

app = Flask(__name__)

# معلومات النموذج
MODEL_PATH = 'madar_model.keras'
FILE_ID = '18O94wPYQ4Oa4Waa-eEV3h-O3Y5znZL0I'

# وظيفة لتحميل النموذج من Google Drive
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

# تحميل النموذج
model = load_model(MODEL_PATH)

# تحميل المرمّج (الترميز)
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
    predicted_class = np.argmax_

