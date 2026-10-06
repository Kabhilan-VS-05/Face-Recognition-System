# Face Recognition Attendance & Biometric Identification System

A computer vision-based biometric authentication and attendance system built using **OpenCV** and **Flask**. The system captures training faces via live webcam, trains a facial feature model using **Local Binary Patterns Histograms (LBPH)**, and provides real-time multi-person facial recognition and automated attendance logging.

---

## 👁️ Core Capabilities

- **Automated Face Dataset Capture (`capture_faces.py`)**:
  - Live webcam face detection using OpenCV's Haar Feature-based Cascade Classifier (`haarcascade_frontalface_default.xml`).
  - Automatically crops, normalizes, and stores grayscale face samples per user identifier.
- **Model Training Pipeline (`train_model.py`)**:
  - Ingests user face directories, extracts label IDs and pixel arrays.
  - Trains the OpenCV LBPH Face Recognizer and serializes the trained model weights to `configuration`.
- **Real-Time Video Recognition (`recognizer.py`)**:
  - High-framerate face identification with confidence thresholds and bounding box visualizations.
- **Web Attendance Dashboard (`app.py`)**:
  - Flask web server providing a browser UI to trigger recognition and review attendee logs.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.x
- **Computer Vision**: [OpenCV](https://opencv.org/) (`cv2` with `opencv-contrib-python`)
- **Web Server**: [Flask](https://flask.palletsprojects.com/)
- **Classifier**: Haar Cascade Frontal Face Classifier
- **Recognition Algorithm**: LBPH (Local Binary Pattern Histograms)

---

## 📁 Repository Structure

```
face_recognition_project/
├── app.py                             # Flask web application & routes
├── capture_faces.py                   # Webcam face acquisition script
├── train_model.py                     # LBPH model trainer
├── recognizer.py                      # Real-time inference script
├── haarcascade_frontalface_default.xml # Pretrained OpenCV Haar face detector
├── configuration                      # Serialized trained model weights
├── dataset/                           # Raw acquired face images organized by user
├── faces/                             # Cropped facial crops
└── static/ & temlates/                # Flask web styling and HTML views
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install opencv-python opencv-contrib-python flask numpy
```

### 2. Capture Faces for a New User
```bash
python capture_faces.py
```
*Look into the webcam until the script acquires required facial angles.*

### 3. Train the Recognizer Model
```bash
python train_model.py
```

### 4. Run Live Recognition
- Run directly in OpenCV window:
  ```bash
  python recognizer.py
  ```
- Or launch the Web Dashboard:
  ```bash
  python app.py
  ```
  Visit `http://localhost:5000` in your browser.

---

## 📄 License
Developed by [Kabhilan VS](https://github.com/Kabhilan-VS-05).
