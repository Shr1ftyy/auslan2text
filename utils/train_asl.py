#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 

# TODO: 
#  - improve conversion loop for converting into one-hot vectors (if possible) 
#  - implement data augmentation?
#  - implement implement GAN to generate diverse, synthetic datasets? 

# from tensorflow.keras.models 
import sys
sys.path.append(r'../src')
# from model import MobileNet
import tensorflow as tf
tf.executing_eagerly()
from tensorflow.keras.applications.mobilenet import MobileNet
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from tensorflow.keras import Sequential as Seq
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import argparse
import os
import seaborn as sns

#Constants
EPOCHS = 200
BATCH_SIZE = 32
CLASSES = 25

letters = "ABCDEFGHIKLMNOPQRSTUVWXY" # Js and Zs are not in this dataset, lol
# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Parsing Arguments
# parser = argparse.ArgumentParser(description='Script for training ASL Recognition Model')
# parser.add_argument('dir', metavar='-i', type=str, nargs='?', 
#         help='directory to get data (in csv, images and labels)')
# args = parser.parse_args()

print("Loading training data")
train_df = pd.read_csv("../train_data/asl/sign_mnist_train/sign_mnist_train.csv")
print("Loading testing data")
test_df = pd.read_csv("../train_data/asl/sign_mnist_test/sign_mnist_test.csv")

test = pd.read_csv("../train_data/asl/sign_mnist_test/sign_mnist_test.csv")
y = test['label']

print(train_df.head())

# plt.figure(figsize = (10,10)) # Label Count
# sns.set_style("darkgrid")
# sns.countplot(train_df['label'])
# plt.show()
# sys.exit()

y_train = train_df['label'].values
print(y_train[:10])

y_test = test_df['label'].values
del train_df['label']
del test_df['label']

one_k = []
for value in y_train:
    temp = [0 for i in range(0,CLASSES)]
    assert len(temp) == CLASSES
    temp[value] = 1
    one_k.append(temp)
    # one_k.append([1 if i is value else 0 for i in range(0,CLASSES)])

y_train = np.array(one_k)
print(y_train[:10])
print(y_train.shape)
# sys.exit()

one_k = []
for value in y_test:
    temp = [0 for i in range(0,CLASSES)]
    assert len(temp) == CLASSES
    temp[value-1] = 1
    one_k.append(temp)

y_test = np.array(one_k)


# label_binarizer = LabelBinarizer()
# y_train = label_binarizer.fit_transform(y_train)
# y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values


# cv2.imwrite('./original.png', x_train[0].reshape(28,28))

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)


# f, ax = plt.subplots(2,5) 
# f.set_size_inches(10, 10)
# k = 0
# for i in range(2):
#     for j in range(5):
#         ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
#         k += 1
#     plt.tight_layout()   

# plt.show()

# shape = (1024, 1, 1)
# model = MobileNet(classes=CLASSES, include_top=False)

# x = model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Reshape(shape, name='reshape_1')(x)
# x = layers.Dropout(0.3, name='dropout')(x)
# x = layers.Conv2D(CLASSES, (1, 1),
#           padding='same',
#           name='conv_preds')(x)
# x = layers.Reshape((CLASSES,), name='reshape_2')(x)
# x = layers.Activation('softmax', name='act_softmax')(x)

model = Seq() 
model.add(layers.Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(layers.Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(layers.Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(layers.Flatten())
model.add(layers.Dense(units = 512 , activation = 'relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(units = CLASSES , activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
model.save('test.h5')
print(model.summary())
print(history.history)

epochs = [i for i in range(EPOCHS)]
fig , ax = plt.subplots(1,2)
train_loss = history.history['loss']
# val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
# ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
