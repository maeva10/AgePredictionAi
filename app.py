from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and labels
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_age(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return ['Error: Unable to load image.']

    # Resize if too large
    if image.shape[1] > 1000:
        image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return ['No face detected']

    predictions = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w].copy()
        if face_img.size == 0:
            continue
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            predicted_age = AGE_BUCKETS[age_preds[0].argmax()]
            predictions.append(predicted_age)
        except Exception as e:
            predictions.append(f"Error: {str(e)}")

    return predictions if predictions else ['No valid face regions']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded.")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predictions = predict_age(filepath)
        relative_path = f"uploads/{filename}"

        return render_template('index.html', age=predictions, filename=relative_path)

    return render_template('index.html', error="Invalid file format.")

if __name__ == '__main__':
    app.run(debug=True)
