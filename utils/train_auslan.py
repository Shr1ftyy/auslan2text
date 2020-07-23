#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 

# BTW, this isn't complete yet, currently focused on 
# researchign about different approaches to solving this
# problemm, and testing on ASL dataset :|

# from tensorflow.keras.models 
import sys
sys.path.append(r'../src')
from model import MobileNet
import numpy as np
import cv2
import argparse
import os

configproto = tf.compat.v1.ConfigProto() 
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

#Constants
EPOCHS = 100
BATCH_SIZE = 3

letters = "ABCDEFGHIJKLMNOP"
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

print("Parsing image files...")
imgs = np.array([cv2.imread(f'{args.img_dir}{img}') for img in imgfiles])/255.0
print(imgs.shape)

print("Loading Model...")
model = MobileNet(classes=16)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(imgs, labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

outputs = model.predict(imgs[:3])

print(f"outputs:\n{[letters(i) for i in outputs]}")
