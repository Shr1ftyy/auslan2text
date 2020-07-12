#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 

import cv2
import numpy as np 
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np


class Predictor:
    """Returns Predictions from Neural Network"""
    def __init__(self):
        self.model = MobileNetV2(weights='imagenet')
        print(dir(self.model))

    def predict(self, _input):
        self.img = image.load_img(img_path, target_size=input.shape[:2])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

pred = Predictor()