import cv2
import numpy as np
from tensorflow.keras.models import load_model

img = np.expand_dims(cv2.resize(cv2.imread('./2.png', 0), (28, 28)), axis=2)/255.0
cv2.imshow('_', img)
cv2.waitKey(0)
imgs = np.array([img])
print(imgs.shape)
model = load_model('test.h5')
pred = model.predict(imgs)
print(f'Prediction: {pred}')
print(f'Prediction: {np.argmax(pred)}')
