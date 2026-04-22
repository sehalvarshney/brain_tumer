from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ✅ Load model (absolute path for EC2 safety)
MODEL_PATH = "/home/ubuntu/brain_tumor_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class labels
classes = ["Mild", "Moderate", "No Tumor", "Very Mild"]

# ✅ Image preprocessing
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Home route (Frontend)
@app.route("/")
def home():
    return render_template("index.html")

# ✅ API health check
@app.route("/api")
def api():
    return jsonify({"message": "Brain Tumor API is Running"})

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Open and preprocess image
        image = Image.open(file)
        img = preprocess_image(image)

        # Predict
        prediction = model.predict(img)
        class_index = int(np.argmax(prediction))
        result = classes[class_index]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Run locally (not used in EC2 gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)