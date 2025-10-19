from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
from datetime import datetime

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('models/fake_face_detector.h5')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_upload_history():
    history = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            timestamp = os.path.getmtime(filepath)
            
            # Get the detection result for this image
            image = cv2.imread(filepath)
            image = cv2.resize(image, (128, 128)) / 255.0
            pred = model.predict(np.expand_dims(image, axis=0))[0][0]
            result = 'fake' if pred > 0.5 else 'real'
            
            history.append({
                'filename': filename,
                'timestamp': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'result': result
            })
    return sorted(history, key=lambda x: x['timestamp'], reverse=True)

@app.route('/')
def index():
    history = get_upload_history()
    return render_template('index.html', history=history)

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        image = cv2.resize(image, (128, 128)) / 255.0
        pred = model.predict(np.expand_dims(image, axis=0))[0][0]
        result = {
            'overall_result': 'fake' if pred > 0.5 else 'real',
            'num_faces': 1
        }
        return render_template('result.html', filename=filename, result=result)
    else:
        flash('Unsupported file type')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
