# Fake Face Detector

A deep learning-based application that can detect fake/forged faces in images using forensic analysis.

## Features

- Upload and analyze images for fake face detection
- Real-time prediction using a trained deep learning model
- Web interface for easy interaction
- Support for common image formats (PNG, JPG, JPEG)

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Project Workflow and Execution Steps

Follow these steps in order to set up and run the project:

1. **Download Dataset**
   ```bash
   # Download original videos
   python download_data.py dataset/ -d original -c raw -t videos -n 10 --server EU2
   
   # Download Deepfake videos
   python download_data.py dataset/ -d Deepfakes -c raw -t videos -n 10 --server EU2
   ```

2. **Extract Frames**
   ```bash
   python extract_frames.py
   ```
   This step extracts individual frames from the downloaded videos for training.

3. **Train the Model**
   ```bash
   python train_model.py
   ```
   This will train the fake face detection model using the extracted frames.

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Install all required Python packages.

5. **Run the Application**
   ```bash
   python app.py
   ```
   This starts the Flask web server.

6. **Access the Application**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
FakeFaceDetector/
├── app/
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── result.html
│   │   └── about.html
│   ├── static/uploads/     # Stores uploaded images/videos
│   └── routes.py           # Flask routes and model logic
├── models/
│   └── fake_face_detector.h5  # Trained model
├── dataset/
│   ├── real/              # Extracted frames from original videos
│   └── fake/              # Extracted frames from manipulated videos
├── download_data.py
├── extract_frames.py
├── train_model.py
├── requirements.txt
├── app.py
└── README.md
```

### Directory Descriptions

- `app/`: Contains the web application components
  - `templates/`: HTML templates for the web interface
  - `static/uploads/`: Directory for storing user-uploaded files
  - `routes.py`: Contains Flask route definitions and model logic
- `models/`: Stores the trained deep learning model
- `dataset/`: Contains the training data
  - `real/`: Extracted frames from original videos
  - `fake/`: Extracted frames from manipulated videos
- Core Python files:
  - `download_data.py`: Script for downloading dataset (creates temporary video folders)
  - `extract_frames.py`: Extracts frames from videos and cleans up temporary folders
  - `train_model.py`: Trains the detection model
  - `app.py`: Main application entry point
  - `requirements.txt`: Project dependencies

## Usage

1. On the web interface, click the upload button to select an image
2. The application will analyze the image and display the results
3. Results will show whether the face is real or fake

## Model Training Details

The model training process involves:
1. Downloading both original and deepfake videos
2. Extracting frames from these videos
3. Training a deep learning model to distinguish between real and fake faces
4. Saving the trained model for use in the web application
