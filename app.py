from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from flask_cors import CORS   # ✅ for frontend connection

app = Flask(__name__)
CORS(app)   # ✅ enable CORS

# Load model (make sure file exists)
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Class labels
classes = ["Mild", "Moderate", "No Tumor", "Very Mild"]

# Preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")        # ✅ FIX: remove alpha channel
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home route
@app.route("/")
def home():
    return "Brain Tumor API is Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get file
        file = request.files["file"]

        # Open image
        image = Image.open(file)

        # Preprocess
        img = preprocess_image(image)

        # Predict
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        result = classes[class_index]

        # Confidence score
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == "__main__":
    app.run(debug=True)