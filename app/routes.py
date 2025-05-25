# app/routes.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret'

model = load_model('models/fake_face_detector.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image = cv2.imread(save_path)
            image = cv2.resize(image, (128, 128)) / 255.0
            pred = model.predict(np.expand_dims(image, axis=0))[0][0]
            result = {
                'overall_result': 'fake' if pred > 0.5 else 'real',
                'faces': [],
                'num_faces': 1
            }
        else:
            result = {
                'overall_result': 'real',  # Placeholder: implement video frame analysis if needed
                'num_faces': 1,
                'num_frames_analyzed': 1,
                'total_frames': 1
            }

        return render_template('result.html', filename=filename, result=result)
    flash('File type not allowed')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
