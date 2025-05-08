import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

IMG_SIZE = 128
model = load_model('./models/model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print("🧪 Это дипфейк! (%.2f)" % prediction)
    else:
        print("✅ Реальное изображение (%.2f)" % prediction)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image")
    else:
        predict(sys.argv[1])
