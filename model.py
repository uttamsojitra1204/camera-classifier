from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import os

class Model:
    def __init__(self):
        self.model = LinearSVC()

    def train_model(self, counters):
        img_list = []
        class_list = []

        # Load and preprocess images for class 1
        for i in range(1, counters[0]):
            img = cv.imread(f'1/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (120, 140)).flatten()  # Resizing for consistency
            img_list.append(img)
            class_list.append(1)

        # Load and preprocess images for class 2
        for i in range(1, counters[1]):
            img = cv.imread(f'2/frame{i}.jpg', cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (120, 140)).flatten()
            img_list.append(img)
            class_list.append(2)

        if not img_list:  # Check if data exists
            print("Error: No data available for training.")
            return False

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.model.fit(img_list, class_list)
        print("Model successfully trained!")
        return True

    def predict(self, frame):
        img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)  # Convert frame to grayscale
        img = cv.resize(img, (120, 140)).flatten()   # Resize and flatten

        prediction = self.model.predict([img])  # Predict class
        return prediction[0]

