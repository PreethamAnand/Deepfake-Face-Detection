import cv2
import os

def extract_frames(video_path, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    count = 0
    for file in os.listdir(video_path):
        if file.endswith('.mp4'):
            full_path = os.path.join(video_path, file)
            print(f"Processing: {full_path}")  # 👈 Print current file
            cap = cv2.VideoCapture(full_path)

            if not cap.isOpened():
                print(f"❌ Cannot open: {full_path}")
                continue

            success, image = cap.read()
            frame_num = 0
            while success:
                filename = f"{label}_{count}.jpg"
                cv2.imwrite(os.path.join(output_folder, filename), image)
                success, image = cap.read()
                count += 1
                frame_num += 1

            cap.release()
            print(f"✅ Done: {file} ({frame_num} frames)\n")

# ✅ Add your corrected paths here
extract_frames('dataset/original_sequences/youtube/raw/videos', 'dataset/real', 'real')
extract_frames('dataset/manipulated_sequences/Deepfakes/raw/videos', 'dataset/fake', 'fake')
