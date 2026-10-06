import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained face recognition model (TensorFlow)
model = load_model("face_recognition_model.h5")  # Replace with your model file

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Predefined user database (ID-to-name mapping)
USER_DATABASE = {
    0: "Alice",
    1: "Bob",
    2: "Charlie"
}

# Function to preprocess face for model input
def preprocess_face(face_image):
    # Resize to the input shape of your model (e.g., 224x224)
    face_resized = cv2.resize(face_image, (224, 224))
    face_normalized = face_resized / 255.0  # Normalize pixel values
    face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension
    return face_expanded

# Recognizer function
def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Extract face region of interest
        processed_face = preprocess_face(face_roi)
        
        # Predict using the trained model
        predictions = model.predict(processed_face)
        predicted_id = np.argmax(predictions)  # ID with the highest probability
        confidence = predictions[0][predicted_id]

        if confidence > 0.8:  # Threshold for a match
            return {"id": predicted_id, "name": USER_DATABASE.get(predicted_id, "Unknown")}
        else:
            return {"id": None, "name": "Unknown"}

    return {"id": None, "name": "No Face Detected"}

# Flask route for real-time recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    # subprocess.run(['python', 'recognizer.py'])  # Comment this line
    return "Recognition Process Triggered!"


# Flask route for granting access
@app.route('/access', methods=['POST'])
def grant_access():
    data = request.json
    if data.get("name") != "Unknown":
        return jsonify({"status": "Access Granted", "user": data.get("name")})
    return jsonify({"status": "Access Denied"})

if __name__ == "__main__":
    app.run(debug=True)
