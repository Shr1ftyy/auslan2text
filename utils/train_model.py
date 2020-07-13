from tensorflow.keras.models import 
from model import MobileNet
import numpy as np
import cv2
import argparse
import os

# Parsing Arguments
parser = argparse.ArgumentParser(description='plays images from a selected TXT_DIR')
parser.add_argument('img_dir', metavar='-i', type=str, nargs='?', 
        help='directory to get image data')
parser.add_argument('label_dir', metavar='-l ', type=str, nargs='?', 
        help='directory to text file containing image labels')
args = parser.parse_args()

def imgsort(files, extension):
    """
    Sorts images in ascending order from a directory into a list
    """
    convFiles = []
    for i in range(0, len(files)):
        convFiles.append(int(files[i].split('.')[0]))

    convFiles.sort(reverse=False)

    for num in range(0, len(convFiles)):
        convFiles[num] = (str(convFiles[num]) + f'.{extension}')

    return convFiles

# Sorts images and labels into arrays
imgfiles = imgsort(os.listdir(args.img_dir), extension='png')

with open(args.label_dir, 'r') as f:
    labels = f.read().split(',')
    labels[len(labels)-1] = labels[len(labels)-1][:-1]
    labels = np.expand_dims(np.array(labels), axis=1)

print(imgfiles)
print(labels)
print(labels.shape)

imgs = np.array([cv2.imread(img) for img in imgfiles])
print(img.shape)

model = MobileNet(classes=16)
model.fit()
