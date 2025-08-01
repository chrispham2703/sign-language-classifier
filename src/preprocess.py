import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

IMG_SIZE = 64  
DATA_DIR = "data/asl_alphabet_train" 

def load_data(limit_per_class=None):
    images = []
    labels = []

    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        count = 0
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                images.append(img)
                labels.append(label)
                count += 1

                if limit_per_class and count >= limit_per_class:
                    break

    return np.array(images), np.array(labels)

def preprocess(limit_per_class=None, test_size=0.2, random_state=42):
    images, labels = load_data(limit_per_class)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
    )

    # Reshape for CNN (add 1 channel)
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X_train, X_test, y_train, y_test, label_encoder
