#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 

import tensorflow as tf
import numpy as np
import sys

MODEL = '../models/deepspeech-0.7.4-models.tflite'
print(f'Model location: {MODEL}')

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(f'Input:\n{input_details}')
output_details = interpreter.get_output_details()
print(f'Output:\n{output_details}')

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(f'INPUT SHAPE: {input_shape}')
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# print(f'INPUT DATA: {input_data}')
print('set input data')
print(dir(interpreter))
interpreter.set_tensor(input_details[0]['index'], input_data)
print('set tensor')

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print('\n\n\n\n----------------------OUTPUT----------------------\n\n\n\n')
print(output_data)
print(f'Shape: {np.shape(output_data)}')
print('DONE')
