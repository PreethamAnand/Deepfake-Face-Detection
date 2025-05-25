import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMG_SIZE = 128

def load_images(folder, label):
    data = []
    for img in os.listdir(folder):
        try:
            image = cv2.imread(os.path.join(folder, img))
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append((image, label))
        except:
            pass
    return data

real_data = load_images('dataset/real', 0)
fake_data = load_images('dataset/fake', 1)

data = real_data + fake_data
np.random.shuffle(data)

X = np.array([i[0] for i in data]) / 255.0
y = np.array([i[1] for i in data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save('models/fake_face_detector.h5')
